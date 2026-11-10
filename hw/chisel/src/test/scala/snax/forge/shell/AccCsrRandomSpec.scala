package snax.forge.shell

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, LaneBits, Rand, Seeds, Sink, Source}
import snax.forge.shell.AcceleratorSpec.{CsrSpec, GroupSpec, LinkSpec, PortSpec, UnitSpec}

/** `<tag>_csr` (`AccCsr`) under random requests, against a cycle-level model of the protocol in its docstring
  * (`Model`).
  *
  * Per seed, `Layouts` CSR layouts of 1..3 groups, each a fold (with JOB_BEATS) or not, the offsets in CSR order, at a
  * random job_width (1, 5, 8, 16 or 32) and busy_counter_width (1, 2, 3, 8 or 32; D152), each driven for `Cycles`
  * cycles: a core that offers a request after a random gap and holds it until taken (reads, writes and starts of a
  * random group, in range, just past the last CSR and anywhere in 32 bits), `rsp.ready` and each group's `starts` ready
  * stalled at random, each group's busy random and its done pulse now and then. Checked every cycle: `req.ready`,
  * `rsp.valid`, `rsp.bits` while valid (START 0, a register its written value's low bits, STATUS the sticky done and
  * busy, BUSY_COUNTER saturating), each group's `starts.valid` and its JOBS and each JOB_BEATS in unit order (a group
  * of 0, 1 or 2 folds: JOBS and 1, or a JOB_BEATS per fold). Each unit may hold a scalar after its JOB_BEATS (`K1` of
  * 8, 16 or 32 bits, or `K1_LO` and `K1_HI` at 64, each with the pin `k1`): `scalars` must give each one's register
  * every cycle.
  */
class AccCsrRandomSpec extends AnyFlatSpec with ForgeTester {

  private val Layouts = 3
  private val Cycles  = 300

  /** A layout of `folds.size` groups, group g of `folds(g)` folds (0: one unit without a JOB_BEATS; 1: one fold; 2: two
    * folds linked by a wire, a JOB_BEATS each).
    */
  private def layout(folds: Seq[Int], jw: Int, cw: Int, r: Rand): AcceleratorSpec = {
    var at     = 0
    def next() = { at += 1; at - 1 }
    val groups = folds.zipWithIndex.map { case (n, g) =>
      val start  = CsrSpec("START", "rw", next(), 1)
      val jobs   = CsrSpec("JOBS", "rw", next(), jw)
      val ports  = Seq("a" -> PortSpec("in", "in", "int64", 1), "out" -> PortSpec("out", "out", "int64", 1))
      val units  = (0 until (n max 1)).map { k =>
        val beats  = if (n > 0) Seq(CsrSpec("JOB_BEATS", "rw", next(), jw)) else Seq.empty
        val bits   = r.pick(Seq(0, 8, 16, 32, 64))
        val scalar = bits match {
          case 0  => Seq.empty
          case 64 => Seq("K1_LO", "K1_HI").map(CsrSpec(_, "rw", next(), 32, Some("k1")))
          case w  => Seq(CsrSpec("K1", "rw", next(), w, Some("k1")))
        }
        val dtype  = s"int${bits max 8}"
        UnitSpec(
          s"u${g}_$k",
          "accumulate",
          "TreeAccumulate",
          ports,
          None,
          Map.empty,
          1,
          1,
          1,
          beats ++ scalar,
          if (bits == 0) Seq.empty else Seq("k1" -> dtype)
        )
      }
      val status = CsrSpec("STATUS", "ro", next(), 2)
      val count  = CsrSpec("BUSY_COUNTER", "ro", next(), cw)
      val links  = units
        .sliding(2)
        .collect { case Seq(a, b) =>
          LinkSpec(s"l$g", a.name -> "out", b.name -> "a", 0)
        }
        .toSeq
      GroupSpec(s"g$g", Seq(start, jobs, status, count), units, links)
    }
    AcceleratorSpec(
      "p",
      "t",
      groups.map(g => g.units.head.name -> "a"),
      groups.map(g => g.units.last.name -> "out"),
      groups
    )
  }

  /** The protocol as `AccCsr`'s docstring states it. */
  private class Model(spec: AcceleratorSpec) {
    val csrs    = spec.csrs
    val regs    = Array.fill(csrs.size)(BigInt(0)) // START's stays 0: a strobe
    val done    = Array.fill(spec.groups.size)(false)
    val counter = Array.fill(spec.groups.size)(BigInt(0))
    var kept    = Option.empty[BigInt]             // a read's answer not yet taken
    def at(g: Int, n: String): Option[Int] = spec.groups(g).csr(n).map(_.offset)
    def config(g: Int): (BigInt, Seq[BigInt]) = {
      val beats = spec.groups(g).jobBeats.map(c => regs(c.offset))
      (regs(at(g, "JOBS").get), if (beats.isEmpty) Seq(BigInt(1)) else beats)
    }

    /** One cycle: req.ready, the response, which group starts; then the registers, each group's done and counter, and
      * the kept answer.
      */
    def cycle(
      req:        Option[(BigInt, BigInt, Boolean)],
      rspReady:   Boolean,
      startReady: Seq[Boolean],
      busy:       Seq[Boolean],
      pulse:      Seq[Boolean]
    ) = {
      val start  = req.flatMap { case (addr, data, write) =>
        spec.groups.indices.find(g => write && addr == at(g, "START").get && data.testBit(0))
      }
      val ready  = kept.isEmpty && start.forall(startReady)
      val fired  = req.filter(_ => ready)
      val answer = fired.collect { case (addr, _, false) =>
        csrs
          .find(_.offset == addr)
          .map { c =>
            val g = spec.groups.indexWhere(_.window.contains(c))
            c.name match {
              case "STATUS"       => (if (done(g)) BigInt(2) else BigInt(0)) + (if (busy(g)) 1 else 0)
              case "BUSY_COUNTER" => counter(g)
              case _              => regs(c.offset)
            }
          }
          .getOrElse(BigInt(0))
      }
      val rsp    = answer.orElse(kept)
      val starts = fired.flatMap(_ => start)
      fired.foreach { case (addr, data, write) =>
        csrs
          .find(c => c.offset == addr && c.access == "rw" && c.name != "START")
          .foreach(c => if (write) regs(c.offset) = data & LaneBits.mask(c.width))
      }
      for (g <- spec.groups.indices) {
        val max = LaneBits.mask(spec.groups(g).csr("BUSY_COUNTER").get.width)
        if (pulse(g)) done(g)              = true else if (starts.contains(g)) done(g) = false
        if (starts.contains(g)) counter(g) = 0 else if (busy(g) && counter(g) < max) counter(g) += 1
      }
      kept = rsp.filter(_ => !rspReady)
      (ready, rsp, starts)
    }
  }

  private def request(r: Rand, spec: AcceleratorSpec): (BigInt, BigInt, Boolean) = {
    val last = spec.csrs.size
    val addr = r.pick(Seq.fill(7)(BigInt(r.between(0, last - 1))) ++ Seq(BigInt(r.between(last, last + 3)), r.word(32)))
    r.pick(Seq("read", "write", "start")) match {
      case "read"  => (addr, BigInt(0), false)
      case "write" => (addr, r.word(32), true)
      case _       =>
        val g = r.between(0, spec.groups.size - 1)
        (BigInt(spec.groups(g).csr("START").get.offset), r.word(32) | (if (r.chance(0.7)) 1 else 0), true)
    }
  }

  behavior of "<tag>_csr under random requests"

  for (seed <- Seeds())
    it should s"give req.ready, the response and each group's start as the protocol's model (seed $seed)" in
      Seeds.run(seed) { r =>
        for (l <- 0 until Layouts) {
          val jw     = r.pick(Seq(1, 5, 8, 16, 32))
          val cw     = r.pick(Seq(1, 2, 3, 8, 32))
          val spec   = layout(Seq.fill(r.between(1, 3))(r.between(0, 2)), jw, cw, r)
          val groups = spec.groups.size
          val model  = new Model(spec)
          val reqs   = IndexedSeq.fill(Cycles)(request(r, spec))
          val gap    = r.probability()
          val core   = new Source(Cycles, BigInt(_), _ => !r.chance(gap))
          val rsp    = Sink.random(r.probability(), r)
          val starts = Seq.fill(groups)(Sink.random(r.probability(), r))
          val pBusy  = r.probability()
          test(new AccCsr(spec)) { dut =>
            for (c <- 0 until Cycles) {
              val clue    = s"layout $l (${spec.csrs.map(_.name).mkString(" ")}, widths $jw, $cw), cycle $c"
              core.cycle(c)
              val req     = if (core.valid) Some(reqs(core.bits.toInt)) else None
              val busy    = Seq.fill(groups)(r.chance(pBusy))
              val pulse   = Seq.fill(groups)(r.chance(0.05))
              dut.io.req.valid.poke(core.valid.B)
              req.foreach { case (addr, data, write) =>
                dut.io.req.bits.addr.poke(addr.U)
                dut.io.req.bits.data.poke(data.U)
                dut.io.req.bits.write.poke(write.B)
              }
              dut.io.rsp.ready.poke(rsp.cycle(c).B)
              val readies = starts.map(_.cycle(c))
              readies.zipWithIndex.foreach { case (v, g) => dut.io.starts(g).ready.poke(v.B) }
              busy.zipWithIndex.foreach { case (v, g) => dut.io.busy(g).poke(v.B) }
              pulse.zipWithIndex.foreach { case (v, g) => dut.io.done(g).poke(v.B) }

              val configs                  = (0 until groups).map(model.config)
              val scalars                  = AccCsr.scalars(spec).map(c => model.regs(c.offset))
              val (ready, answer, started) = model.cycle(req, rsp.isReady, readies, busy, pulse)
              assert(dut.io.req.ready.peek().litToBoolean == ready, s"$clue: req.ready, the model $ready (req $req)")
              assert(dut.io.rsp.valid.peek().litToBoolean == answer.isDefined, s"$clue: rsp.valid, the model $answer")
              answer.foreach(a => assert(dut.io.rsp.bits.data.peek().litValue == a, s"$clue: rsp.bits, the model $a"))
              for (g         <- 0 until groups) {
                val v     = dut.io.starts(g).valid.peek().litToBoolean
                assert(v == started.contains(g), s"$clue: group $g's start ${v}, the model ${started.mkString}")
                assert(
                  dut.io.starts(g).bits.jobs.peek().litValue == configs(g)._1,
                  s"$clue: $g's JOBS"
                )
                val beats = dut.io.starts(g).bits.beats.toSeq
                assert(
                  beats.map(_.peek().litValue) == configs(g)._2,
                  s"$clue: $g's JOB_BEATS"
                )
              }
              for ((want, i) <- scalars.zipWithIndex)
                assert(dut.io.scalars(i).peek().litValue == want, s"$clue: scalar CSR $i, the model $want")
              core.fired(ready)
              dut.clock.step()
            }
          }
        }
      }
}
