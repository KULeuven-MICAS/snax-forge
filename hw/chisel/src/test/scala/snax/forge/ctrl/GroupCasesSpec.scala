package snax.forge.ctrl

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, RunLines, SnaxGroupCases}
import snax.forge.stream.{FifoPacker, FifoSlicer}

/** A group control unit with its outer output packed `dOut` results to a 64-bit word and its sliced outer inputs
  * (`dIn`, each its index and pops per word), wired as `shell.ShellWrapper` wires a packed outer output (D174) and a
  * sliced outer input (D180): the group's outer handshake into a `FifoPacker(1, int<64 / dOut>, 1)`, cleared when the
  * start is accepted, `last` the group's; a sliced input's reader into a `FifoSlicer(1, int<64 / d>, 1)`, its pops the
  * group's input handshake, `enable` the group's busy, `clear` the start accepted. `io` is `GroupCtrl`'s with
  * `out.ready` the writer's and a sliced input's valid and ready its reader's; `pk.value` the owner's result pushed,
  * `pk.sent` a word to the writer and `pk.word` that word.
  */
class PackedGroupCtrl(desc: GroupDesc, dOut: Int, dIn: Seq[(Int, Int)] = Seq.empty) extends Module {
  private val ctrl   = Module(new GroupCtrl(desc, "case"))
  private val packer = Module(new FifoPacker(1, s"int${64 / dOut}", 1))
  require(packer.r == dOut)

  val io = IO(new GroupCtrlIO(desc))
  val pk = IO(new Bundle {
    val value = Input(UInt((64 / dOut).W))
    val sent  = Output(Bool())
    val word  = Output(UInt(64.W))
  })

  io <> ctrl.io
  for ((i, d) <- dIn) {
    val slicer = Module(new FifoSlicer(1, s"int${64 / d}", 1))
    require(slicer.r == d)
    slicer.io.in.valid  := io.in(i).valid
    io.in(i).ready      := slicer.io.in.ready
    slicer.io.in.bits   := VecInit(0.U(64.W))
    ctrl.io.in(i).valid := slicer.io.out.valid
    slicer.io.out.ready := ctrl.io.in(i).ready
    slicer.io.enable    := ctrl.io.busy
    slicer.io.clear     := ctrl.io.start.fire
  }
  packer.io.in.valid := ctrl.io.out.valid
  ctrl.io.out.ready   := packer.io.in.ready
  packer.io.in.bits   := pk.value.asTypeOf(packer.io.in.bits)
  packer.io.out.ready := io.out.ready
  packer.io.clear     := ctrl.io.start.fire
  packer.io.last      := ctrl.io.last
  pk.sent             := packer.io.out.fire
  pk.word             := packer.io.out.bits.asUInt
}

/** `GroupCtrl` alone (no datapath, as UnitCasesSpec drives shapes it has none for) through snax-gvsoc's 58 group cases
  * (`SnaxGroupCases`: dot 2-6 at T 2 and 4, J 1 and 2, dot 5 and 6 at W 1, backpressure, an input gap, JOBS 0, the
  * fold's seed: two folds unseeded, the first or both seeded, a late seed on one fold, on a SerialAccumulate and on a
  * Mac (D187); the output packer, `pack_*`; the input slicer, `slice_*`, among them a seeded Mac's broadcast input
  * sliced 8 from a word (D186: wires after the slicer, which the rule does not see); and bicg's one pass, `fork_*`, two
  * chains on a forked input (D189), a JOBS and a ready per chain's output; their data is checked by `tb.cpp`, and
  * dot5_w1_j2's through the wrapper by ShellEndToEndSpec), cycle for cycle: per cycle the outer valids (a forked
  * input's on each of its units' handshakes, as the wrapper wires it) and each output's ready poked; each input's ready
  * (its unit's firing, the per-unit join of CONTRACTS, "Accelerator RTL spec"; a seed's that firing on its job's first
  * beat, the unit's firings counted mod its T), each unit's fire and give, each buffer's occupancy after the cycle and
  * busy compared; at the run's end its line from `cls` per busy cycle and the head's firings. Each case runs twice (the
  * rule does not tell the two apart, the control unit's counters do): with every unit's T fixed (`params.T`), JOBS on
  * its wire; and with every unit of T != 1 open, its T a JOB_BEATS read live from its wire, so dot 2's and dot 3's mul
  * counts against or borrows from a sum whose T is the csr's. A `pack_*` case (`dout`) runs through `PackedGroupCtrl`:
  * the case's ready is the writer's, the owner's n-th push in a run is -(n + 1) at the lane's width, and per busy cycle
  * `sent` is the packer's word going out; at the end the words sent are the case's `words`. A `slice_*` case (`din`)
  * runs through it too, its sliced inputs through slicers: their valids are the readers', and their ready, the word
  * taken, is compared with the case's `took`. Without `SNAX_GVSOC_GROUP_CASES` the spec cancels.
  */
class GroupCasesSpec extends AnyFlatSpec with ForgeTester {

  /** The case's group: a linked input after the unit's outer ones (to_input = its outer inputs, as `tb.cpp`); its forks
    * by unit; with `open`, every unit of T != 1 reads its T from its JOB_BEATS wire. The case's outputs are its sinks
    * in unit order, `GroupDesc.outputs`.
    */
  def desc(c: SnaxGroupCases.Case, open: Boolean = false): GroupDesc = {
    def at(name: String) = c.units.indexWhere(_.name == name)
    val d = GroupDesc(
      "dot",
      c.units.map { u =>
        val fed = c.links.exists(_.to == u.name)
        val t   = if (open && u.t != 1) None else Some(u.t)
        UnitDesc(u.name, u.latency, u.ii, u.span, u.inputs, t, if (fed) Seq(u.inputs) else Nil, u.seed)
      },
      c.links.map(l => Link(at(l.from), at(l.to), c.units(at(l.to)).inputs, l.depth)),
      c.forks.map(_.map(m => at(m._1)))
    )
    require(
      c.outputs.isEmpty || c.outputs.map(at) == d.outputs,
      s"${c.name}: outputs ${c.outputs}, sinks in unit order"
    )
    d
  }

  /** The packer's side of a pack case: its value poked, what it sent read. */
  case class Pk(value: UInt, sent: Bool, word: UInt)

  private def run(c: SnaxGroupCases.Case, open: Boolean): Unit = {
    val d = desc(c, open)
    if (c.dOut > 1 || c.dIn.nonEmpty) test(new PackedGroupCtrl(d, c.dOut, c.dIn)) { dut =>
      cycles(c, open, d, dut.io, dut.clock, Option.when(c.dOut > 1)(Pk(dut.pk.value, dut.pk.sent, dut.pk.word)))
    }
    else test(new GroupCtrl(d, "case"))(dut => cycles(c, open, d, dut.io, dut.clock, None))
  }

  private def cycles(
    c:     SnaxGroupCases.Case,
    open:  Boolean,
    d:     GroupDesc,
    io:    GroupCtrlIO,
    clock: Clock,
    pk:    Option[Pk]
  ): Unit = {
    val beats  = d.units.zip(c.units).collect { case (u, cu) if u.fixedT.isEmpty => cu.t }
    val place  = c.place.flatten              // per unit input, its column among the case's valids
    val runs   = new RunLines
    val fired  = Array.fill(c.units.size)(0L) // each unit's firings so far: its job's first beat at 0 mod T
    val mask   = (BigInt(1) << (64 / c.dOut)) - 1
    var pushes = 0                            // the owner's pushes in the run, for the packer's values
    val words  = scala.collection.mutable.ArrayBuffer.empty[BigInt]
    for ((cyc, i) <- c.cycles.zipWithIndex) {
      val clue  = s"${c.name}${if (open) " (T from JOB_BEATS)" else ""}: cycle $i"
      val offer = i == c.start._1
      io.start.valid.poke(offer.B)
      io.settings.jobs.poke(c.jobs.U) // the csr's wires, the same through the case
      for {
        more   <- io.settings.more
        (p, j) <- more.zip(c.jobsAll.tail)
      } p.poke(j.U)
      io.settings.beats.zip(beats).foreach { case (p, t) => p.poke(t.U) }
      if (offer) {
        val got = if (!high(io.start.ready)) "refused" else if (c.jobsAll.contains(0)) "ignored" else "taken"
        assert(got == c.start._2, s"$clue: start $got")
        if (got == "taken") pushes = 0
      }
      place.zipWithIndex.foreach { case (col, k) => io.in(k).valid.poke(cyc.valids(col).B) }
      cyc.readies.zipWithIndex.foreach { case (r, o) => io.output(o).ready.poke(r.B) }
      pk.foreach(_.value.poke((BigInt(-(pushes + 1)) & mask).U))
      val busy = high(io.busy)
      assert(busy == cyc.busy, s"$clue: busy $busy")
      runs.cycle(busy, int(io.cls), cyc.fire.head)
      if (busy) {
        for (k   <- c.units.indices) {
          val e = io.units(k)
          assert(high(e.fire) == cyc.fire(k), s"$clue: ${c.units(k).name} fire")
          assert(high(e.give) == cyc.give(k), s"$clue: ${c.units(k).name} give")
        }
        var at = 0
        for {
          (u, k) <- c.units.zipWithIndex
          j      <- 0 until u.inputs
        } {
          val seed = u.seed && j == u.inputs - 1
          c.dIn.indexWhere(_._1 == place(at)) match {
            case -1 =>
              val want = cyc.fire(k) && (!seed || fired(k) % u.t == 0)
              assert(high(io.in(at).ready) == want, s"$clue: input $at's ready, ${u.name}'s firing")
            case s  => // a sliced input: its reader's word taken
              val took = high(io.in(at).ready) && cyc.valids(place(at))
              assert(took == cyc.took(s), s"$clue: input $at's slicer took $took")
          }
          at += 1
        }
        for (k   <- c.units.indices if cyc.fire(k)) fired(k) += 1
        val occ = c.depths.filter(_ > 0).indices.map(i => io.links(i).occupancy.peek().litValue.toInt)
        assert(occ == cyc.occupancy, s"$clue: occupancy $occ")
        for (p <- pk) {
          val sent = high(p.sent)
          assert(sent == cyc.sent, s"$clue: sent $sent")
          if (sent) words += p.word.peek().litValue
        }
        if (high(io.units(d.output).give)) pushes += 1
      }
      clock.step()
    }
    val lines = runs.done
    assert(lines == c.runs, s"${c.name}: runs $lines, the case ${c.runs}")
    if (pk.nonEmpty) assert(words == c.words, s"${c.name}: words ${words.map(_.toString(16))}")
  }

  behavior of "GroupCtrl on snax-gvsoc's group cases"

  SnaxGroupCases.all match {
    case None      =>
      it should "give every group case's cycles and runs" in
        cancel(s"${SnaxGroupCases.Env} is not set (snax-gvsoc's tests/snax/group/cases)")
    case Some(all) =>
      it should "find the 58 group cases" in assert(all.size == 58, s"${all.size} cases")
      for (c <- all) {
        it should s"give ${c.name}'s cycles and runs" in run(c, open = false)
        it should s"give ${c.name}'s cycles and runs with its T from JOB_BEATS" in run(c, open = true)
      }
  }
}
