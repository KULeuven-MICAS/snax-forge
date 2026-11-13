package snax.forge.ctrl

import chisel3._
import chisel3.util._

/** One unit of a group's description (`accelerator.json`): its name, its timing (L, II, span), its outer inputs (the
  * streamer's ports it reads), its T when fixed at design time (`params.T`; 1 for a unit without a fold), else None: a
  * JOB_BEATS read live from the csr; `linked`, the positions among its inputs (in port order) that no outer port takes,
  * each fed by one link (snax_group.cpp's inputs left ""); and `seeded`, its last outer input a fold's seed, one beat
  * per job.
  */
case class UnitDesc(
  name:    String,
  latency: Int,
  ii:      Int,
  span:    Int,
  inputs:  Int,
  fixedT:  Option[Int],
  linked:  Seq[Int] = Nil,
  seeded:  Boolean  = false
) {
  require(latency           >= 0, s"$name: latency must be 0 or more, got $latency")
  require(ii                >= 1, s"$name: ii must be 1 or more, got $ii")
  require(span              >= 1, s"$name: span must be 1 or more, got $span")
  require(inputs            >= 0, s"$name: inputs must be 0 or more, got $inputs")
  require(!seeded || inputs >= 1, s"$name: a seeded unit's seed is one of its outer inputs")
}

/** A link of a group, by unit index (SnaxGroupRule's Link): unit `from`'s output into input `toInput` of unit `to` (its
  * position among that unit's inputs; the datapaths use it, the rule does not); `depth` 0 a wire, >= 1 a buffer of that
  * many beats (`LinkBuffer`).
  */
case class Link(from: Int, to: Int, toInput: Int, depth: Int)

/** A group's description: its units and the links between them, a graph (SnaxGroupRule's `configure()`). Until tier A
  * one chain: a unit with two out-edges (a fork) or two in-edges (a join), a cycle and units no link joins are refused,
  * with SnaxGroupRule's reasons. From it: each unit's in-edges and out-edges, the units in topological order (Kahn's,
  * the lowest index first among the units ready; the head first) and the unit owning the group's outer output, the one
  * whose output no link takes (the sink).
  */
case class GroupDesc(name: String, units: Seq[UnitDesc], links: Seq[Link]) {
  require(units.nonEmpty, s"group $name: no unit")
  for ((e, i) <- links.zipWithIndex) {
    require(
      Seq(e.from, e.to).forall(units.indices.contains) && e.depth >= 0,
      s"group $name: link $i: no such unit, or a depth below 0"
    )
    require(
      units(e.to).linked.contains(e.toInput) && !links.take(i).exists(x => x.to == e.to && x.toInput == e.toInput),
      s"group $name: link $i: ${units(e.to).name}'s input ${e.toInput} is not an input left to a link, or another " +
        "link feeds it"
    )
  }
  for {
    (u, k)    <- units.zipWithIndex
    j         <- u.linked
  }
    require(links.exists(e => e.to == k && e.toInput == j), s"group $name: ${u.name}'s input $j: no link feeds it")

  /** Per unit its in-edges and its out-edges, by link index. */
  val ins:  Seq[Seq[Int]] = units.indices.map(k => links.indices.filter(links(_).to == k))
  val outs: Seq[Seq[Int]] = units.indices.map(k => links.indices.filter(links(_).from == k))
  for (k <- units.indices) {
    val fork = outs(k).size > 1
    require(
      outs(k).size <= 1 && ins(k).size <= 1,
      s"group $name: unit $k has ${if (fork) s"${outs(k).size} out-edges (a fork)"
        else s"${ins(k).size} in-edges (a join)"}" +
        ": until tier A a group is a chain"
    )
  }

  /** The units in topological order, the head first. */
  val order: Seq[Int] = {
    val need  = Array.tabulate(units.size)(ins(_).size)
    val ready = scala.collection.mutable.ArrayBuffer.from(units.indices.filter(need(_) == 0))
    var r     = 0
    while (r < ready.size) {
      for (i <- outs(ready(r))) {
        need(links(i).to) -= 1
        if (need(links(i).to) == 0) ready += links(i).to
      }
      r += 1
    }
    ready.toSeq
  }
  require(order.size == units.size, s"group $name: the links form a cycle")
  private val sinks = units.indices.filter(outs(_).isEmpty)
  require(sinks.size == 1, s"group $name: ${sinks.size} units have no out-edge: until tier A a group is one chain")

  /** The head, the first unit in topological order; the unit owning the outer output. */
  val head:   Int = order.head
  val output: Int = sinks.head
  require(units(head).inputs >= 1, s"group $name: the head ${units(head).name} has no outer input")

  /** The units whose T is a JOB_BEATS (the csr's, read live), in unit order. */
  def open: Seq[UnitDesc] = units.filter(_.fixedT.isEmpty)

  /** The links that are buffers (depth >= 1), by link index: the ones with enables (a wire has none). */
  def buffers: Seq[Int] = links.indices.filter(links(_).depth > 0)
}

/** A Decoupled port split in two: its data goes to the datapath, its valid and ready to the controller. */
class Handshake extends Bundle {
  val valid = Input(Bool())
  val ready = Output(Bool())
}

/** A unit's enables to its datapath (`ctrl` -> `<tag>_datapath`, the catalogue's pins), and what it gives: `fire` (a
  * beat taken from every input; at span > 1 the last lane's read; a plain Mac's `en` too), `first` (the beat fired or
  * read is its job's first, d = 0), `read` and `lane` (lane `lane` read in place, at every span: at span 1 the read is
  * the firing), `adv` (its pipeline advances: not frozen) and `give` (a result pushed into its link or the outer
  * output).
  */
class UnitEnables(span: Int) extends Bundle {
  val fire  = Output(Bool())
  val first = Output(Bool())
  val read  = Output(Bool())
  val lane  = Output(UInt(log2Ceil(span).W))
  val adv   = Output(Bool())
  val give  = Output(Bool())
}

/** A buffer's enables (`LinkBuffer`'s pins) and its occupancy after the cycle. */
class LinkEnables(depth: Int) extends Bundle {
  val load      = Output(Bool())
  val wslot     = Output(UInt(log2Ceil(depth).W))
  val rslot     = Output(UInt(log2Ceil(depth).W))
  val occupancy = Output(UInt(log2Ceil(depth + 1).W))
}

/** The control model of a group (the Control of CONTRACTS, "Accelerator RTL spec"): the same rule as SnaxGroupRule
  * (snax-gvsoc's `snax/snax_group.hpp`), a register only for what a firing changes (the rules at the top of the class),
  * over datapaths of data and enables only. Not a module: it builds its registers and logic in the module that makes it
  * (`GroupCtrl`, `<tag>_<group>_ctrl`), which drives `io.config` (a start that runs: `GroupCtrl` refuses a start while
  * busy and ends the empty run itself), `io.settings` (the csr's JOBS and JOB_BEATS, wires held constant through the
  * run), the outer valids and `io.out.ready`, and reads the rest. Its per-unit signals are named after the unit
  * (`<unit>_fire`, `<unit>_job_beats_counter`, ...).
  *
  * A start of JOBS and each unit's T: the sink gives R = JOBS results, and a unit k before its successor s gives R_k =
  * R_s · T_s (each firing of s takes one of k's results), one on every T_k-th firing (its job_beats_counter at d_k + 1
  * \= T_k), so it fires n_k = R_k · T_k times. JOBS 0 or any T 0 is the empty run, which `GroupCtrl` ends without
  * starting this. Busy from the cycle after the start to the run's last cycle E, the sink's R-th push, where `done` is
  * high.
  *
  * Each cycle is decided in two passes over the units in topological order, as `SnaxGroupRule::step()`: forward, what
  * each unit offers on its output whatever its room (its head, or at L 0 a firing that gives), so each knows whether
  * its input has a beat (its outer valids and every in-edge's offer: a wire its producer's offer, a buffer occupancy >
  * 0); backward, in reverse order, its room (its outer output's ready if it owns it, and every out-edge's take: a wire
  * its consumer's firing, a buffer occupancy < D or that firing), whether it is frozen (its head, or at L 0 a giving
  * firing, has no room), fires and reads. The outer ports' join is per unit: input j's ready is its unit's firing,
  * which needs every one of that unit's outer valids; a seeded unit's seed (its last outer input) only on its job's
  * first beat (fire = a.valid ∧ (¬first ∨ seed.valid)), and its ready is that firing (seed.ready = fire ∧ first). The
  * outer output's valid is its owner's offer, never its ready's. `cls` is the cycle's class, judged at the owner, as
  * the run line counts it: busy 0, stall_in 1, stall_out 2, flush 3.
  */
class GroupRule(val desc: GroupDesc, jobWidth: Int = 32) {
  // NB (D153): a unit counts its results in nested counters, never against n_k; snax-gvsoc's SnaxGroupRule must
  // count the same.
  // The C++ stays the spec (it counts firings up to n_k); here a register exists only for what a firing changes, and
  // a unit counts its results in nested counters, never against a product. No copy of a setting: JOBS
  // and each open T are the csr's CSRs, read live as `io.settings`, which `<tag>_csr` holds constant through the run
  // (it holds a write to them while the group is busy); a fixed T (`params.T`, `UnitDesc.fixedT`) is a constant.
  //   - A unit with T != 1 keeps its beat counter d (`<unit>_job_beats_counter`, its firings mod T): `first` is d = 0,
  //     and a firing is its job's last, giving a result, at d + 1 = T for a T from the csr (the + 1 shared with the
  //     counter's increment) or d = T − 1 for a fixed T (a constant test).
  //   - The results it has given, g_k, are tested only for g_k != R_k (it wants to fire while it has results left):
  //     - A unit at L 0 whose out-link is a wire keeps no count and borrows its consumer s's test: its giving firing
  //       is s's firing, so g_k = T_s · g_s + d_s, and g_k != R_k exactly when g_s != R_s. A wire chain at L 0 so
  //       ends at the segment's last unit.
  //     - Any other unit (the sink, one feeding a buffer, one at L >= 1) counts its giving firings in mixed radix: a
  //       digit per unit after it with T != 1 (`<unit>_<downstream>_job_beats_counter`, nearest first, each wrapping
  //       at that unit's T as d does), then a jobs digit g (`<unit>_jobs_counter`), left = (g != JOBS). R_k = JOBS ·
  //       ∏ T over those units, so g_k = R_k exactly when its jobs digit is JOBS.
  //   - No count of the sink's pushes: with L its latency, g its jobs digit and v its pipeline's valid bits, done is
  //     its push with
  //       g + 1 = JOBS (L = 0: the push is the giving firing, in the cycle g counts it; the + 1 shared with g's)
  //       g = JOBS ∧ ¬(v(i) for some i < L − 1) (L ≥ 1: g already counts the result given, the pipeline below the head
  //       is empty; the test is ¬left)
  //   - The II gap: a register only at II > 1 (at II 1 it never leaves 0).
  // Kept: those counters, the pipeline, the lane, the buffers' occupancy and slots. Every counter is job_width bits,
  // a digit of a fixed T ⌈log₂ T⌉; LOWER refuses JOBS or a JOB_BEATS above 2^job_width − 1, so d and g stay at
  // most 2^job_width − 2 and d + 1, g + 1 fit.
  private val n     = desc.units.size
  private val descs = desc.units
  private val links = desc.links
  private val owner = desc.output

  val io = Wire(new Bundle {
    val config   = Input(Bool())
    val settings = Input(new GroupSettings(desc.open.size, jobWidth))
    val in       = MixedVec(descs.map(u => Vec(u.inputs, new Handshake)))
    val out      = Flipped(new Handshake)
    val units    = MixedVec(descs.map(u => new UnitEnables(u.span)))
    val links    = MixedVec(desc.buffers.map(i => new LinkEnables(desc.links(i).depth)))
    val busy     = Output(Bool())
    val done     = Output(Bool())
    val cls      = Output(UInt(2.W))
  })

  // The group's state: busy. JOBS and each open unit's T are the csr's, live (SnaxGroupRule's jobs and job_beats)
  private val busy = RegInit(false.B)
  private val jobs = io.settings.jobs
  private val open = descs.indices.filter(descs(_).fixedT.isEmpty)

  /** Unit k's T from the csr, its JOB_BEATS read live; None for a fixed T. */
  private def live(k: Int): Option[UInt] = Some(open.indexOf(k)).filter(_ >= 0).map(io.settings.beats(_))

  /** Whether `c`, a count of unit k's T with `inc` = c + 1, is at its last value: inc = T for a T from the csr, c = T −
    * 1 for a fixed T.
    */
  private def last(k: Int, c: UInt, inc: UInt): Bool =
    live(k).map(inc === _).getOrElse(c === (descs(k).fixedT.get - 1).U)
  private def succ(k: Int): Option[Int] = desc.outs(k).headOption.map(links(_).to)

  /** The units after k, nearest first. */
  private def after(k: Int): Seq[Int] = succ(k).map(s => s +: after(s)).getOrElse(Nil)

  /** Whether k borrows its consumer's test: at L 0 with a wire out. */
  private def borrows(k: Int): Boolean = descs(k).latency == 0 && desc.outs(k).exists(links(_).depth == 0)

  /** A counter of a T: ⌈log₂ T⌉ bits for a fixed T, job_width for one from a JOB_BEATS. */
  private def counter(t: Option[Int], name: String): UInt =
    RegInit(0.U(t.map(log2Ceil(_)).getOrElse(jobWidth).W)).suggestName(name)

  // Per unit (SnaxGroupRule's UnitState): its job_beats_counter d (none at T 1); a counting unit's digits (per unit
  // after it with T != 1, then its jobs digit g); the II gap (has_fired and last_fire; none at II 1), the pipeline
  // (slots, v(L-1) the head), the lane
  private class UnitState(val k: Int) {
    val u      = descs(k)
    val d      = u.fixedT match {
      case Some(1) => 0.U(1.W)
      case t       => counter(t, s"${u.name}_job_beats_counter")
    }
    val digits =
      if (borrows(k)) Nil
      else
        after(k).filter(descs(_).fixedT != Some(1)).map { j =>
          j -> counter(descs(j).fixedT, s"${u.name}_${descs(j).name}_job_beats_counter")
        }
    val dInc   = d + 1.U
    val g      = if (borrows(k)) None else Some(RegInit(0.U(jobWidth.W)).suggestName(s"${u.name}_jobs_counter"))
    val gInc   = g.map(_ + 1.U)
    val gap    = if (u.ii == 1) 0.U(1.W) else RegInit(0.U(log2Ceil(u.ii + 1).W)).suggestName(s"${u.name}_gap")
    val v      = RegInit(VecInit(Seq.fill(u.latency max 1)(false.B))).suggestName(s"${u.name}_v")
    val lane   = RegInit(0.U(log2Ceil(u.span).W)).suggestName(s"${u.name}_lane")
  }
  private val st = descs.indices.map(new UnitState(_))

  /** Whether unit k has results left to give (g_k != R_k): its own jobs digit's, or its consumer's test. */
  private def left(k: Int): Bool = st(k).g.map(_ =/= jobs).getOrElse(left(succ(k).get))

  // Per buffer (SnaxGroupRule's LinkState): occupancy and the ring's write and read pointers, named after the link's
  // ends, `<from>_<to>_...` (a wire's are unused)
  private def link(i: Int, what: String) = s"${descs(links(i).from).name}_${descs(links(i).to).name}_$what"
  private val occupancy = links.zipWithIndex.map { case (e, i) =>
    RegInit(0.U(log2Ceil(e.depth + 1).W)).suggestName(link(i, "occupancy"))
  }
  private val wslot     = links.zipWithIndex.map { case (e, i) =>
    RegInit(0.U(log2Ceil(e.depth max 2).W)).suggestName(link(i, "wslot"))
  }
  private val rslot     = links.zipWithIndex.map { case (e, i) =>
    RegInit(0.U(log2Ceil(e.depth max 2).W)).suggestName(link(i, "rslot"))
  }

  // step()'s forward pass, in topological order: what each unit offers whatever its room
  private val inputs, avail, wants, wouldFire, gives, head, offer = Array.fill(n)(WireDefault(false.B))
  for (k <- desc.order) {
    val u = descs(k)
    val s = st(k)
    inputs(k) := (if (!u.seeded) io.in(k).map(_.valid).foldLeft(true.B)(_ && _)
                  else // the seed only on the job's first beat
                    io.in(k).init.map(_.valid).foldLeft(true.B)(_ && _) && (s.d =/= 0.U || io.in(k).last.valid))
    avail(k) := desc.ins(k).foldLeft(inputs(k)) { (a, i) =>
      a && (if (links(i).depth == 0) offer(links(i).from) else occupancy(i) =/= 0.U)
    }
    wants(k) := busy && left(k) && s.gap === 0.U
    wouldFire(k) := wants(k)     && avail(k) && (if (u.span == 1) true.B else s.lane === (u.span - 1).U)
    gives(k)     := wouldFire(k) && last(k, s.d, s.dInc)
    head(k)      := (if (u.latency > 0) s.v(u.latency - 1) else false.B)
    offer(k)     := (if (u.latency > 0) head(k) else gives(k))
    Seq(avail(k) -> "avail", wants(k) -> "wants", gives(k) -> "gives", offer(k) -> "offer").foreach { case (b, w) =>
      b.suggestName(s"${u.name}_$w")
    }
  }

  // step()'s backward pass, in reverse order: room, frozen, fire, read, give (the C++'s push); an out-edge takes on its consumer's firing
  private val room, frozen, fire, read, give = Array.fill(n)(WireDefault(false.B))
  for (k <- desc.order.reverse) {
    val u    = descs(k)
    val take = desc.outs(k).map { i =>
      val e = links(i)
      if (e.depth == 0) fire(e.to) else occupancy(i) < e.depth.U || fire(e.to)
    }
    room(k) := take.foldLeft(if (k == owner) io.out.ready else true.B)(_ && _)
    frozen(k) := (if (u.latency > 0) head(k) else gives(k)) && !room(k)
    fire(k)   := wouldFire(k)                               && !frozen(k)
    give(k)   := (if (u.latency > 0) head(k) else gives(k)) && !frozen(k)
    read(k)   := wants(k)                                   && avail(k) && !frozen(k)
    Seq(frozen(k) -> "frozen", fire(k) -> "fire", give(k) -> "give", read(k) -> "read").foreach { case (b, w) =>
      b.suggestName(s"${u.name}_$w")
    }
  }

  // The outer ports: input j's ready is its unit's firing (a seed's that firing on its job's first beat); the
  // output's valid its owner's offer
  for {
    k      <- 0 until n
    (p, j) <- io.in(k).zipWithIndex
  } p.ready    := (if (descs(k).seeded && j == descs(k).inputs - 1) fire(k) && st(k).d === 0.U else fire(k))
  io.out.valid := offer(owner)

  // The datapaths' enables and the buffers'
  for (k      <- 0 until n) {
    val e = io.units(k)
    e.fire  := fire(k)
    e.first := st(k).d === 0.U
    e.read  := read(k)
    e.lane  := st(k).lane
    e.adv   := !frozen(k)
    e.give  := give(k)
  }
  for ((i, b) <- desc.buffers.zipWithIndex) {
    val e = io.links(b)
    val l = links(i)
    e.load      := give(l.from)
    e.wslot     := (if (l.depth > 1) wslot(i) else 0.U)
    e.rslot     := (if (l.depth > 1) rslot(i) else 0.U)
    e.occupancy := occupancy(i) + give(l.from) - fire(l.to)
  }

  // The class, judged at the owner (step()'s busy and waits, then its cls)
  private val isBusy = (0 until n).map(k => !frozen(k) && (fire(k) || read(k) || st(k).gap =/= 0.U)).reduce(_ || _)
  private val waits  = (0 until n).map(k => !frozen(k) && wants(k) && !inputs(k)).reduce(_ || _)
  io.cls := Mux(frozen(owner), 2.U, Mux(isBusy, 0.U, Mux(waits, 1.U, 3.U)))

  // The sink's R-th push ends the run (the done formula above): then every unit has fired its n, nothing is in flight
  // and every buffer is empty, SnaxGroupRule's finished()
  private val sinkL = descs(owner).latency
  private val done  = busy && give(owner) &&
    (if (sinkL == 0) st(owner).gInc.get === jobs else !left(owner)) &&
    !(if (sinkL > 1) st(owner).v.take(sinkL - 1).reduce(_ || _) else false.B)

  io.done := done
  io.busy := busy

  // The registers, as step() (the backward pass the pipeline, the II and the lane; then the buffers) and start() (the
  // II gap carries over) change them
  for (k <- 0 until n) {
    val u = descs(k)
    val s = st(k)
    if (u.ii > 1) when(fire(k)) {
      s.gap := (u.ii - 1).U
    }.elsewhen(s.gap =/= 0.U) {
      s.gap := s.gap - 1.U
    }
    if (u.latency > 0) when(!frozen(k)) {
      for (i <- u.latency - 1 until 0 by -1) s.v(i) := s.v(i - 1)
      s.v(0) := fire(k) && gives(k)
    }
    if (u.span > 1) when(fire(k)) {
      s.lane := 0.U
    }.elsewhen(read(k)) {
      s.lane := s.lane + 1.U
    }
  }
  for (i <- desc.buffers) {
    val Link(from, to, _, d) = links(i)
    when(give(from)) { wslot(i) := (if (d > 1) Mux(wslot(i) === (d - 1).U, 0.U, wslot(i) + 1.U) else 0.U) }
    when(fire(to)) { rslot(i) := (if (d > 1) Mux(rslot(i) === (d - 1).U, 0.U, rslot(i) + 1.U) else 0.U) }
    occupancy(i) := occupancy(i) + give(from) - fire(to)
  }

  when(io.config) {
    busy := true.B
    for (s <- st) {
      if (s.u.fixedT != Some(1)) s.d := 0.U
      s.digits.foreach(_._2 := 0.U)
      s.g.foreach(_ := 0.U)
      s.lane                         := 0.U
    }
  }.otherwise {
    for (k <- 0 until n) when(fire(k)) {
      val s = st(k)
      if (descs(k).fixedT != Some(1)) s.d := Mux(gives(k), 0.U, s.dInc)
      // A giving firing counts in mixed radix: each digit wraps at its unit's T and carries into the next, the last
      // into the jobs digit
      when(gives(k)) {
        val carry = s.digits.foldLeft(true.B) { case (in, (j, digit)) =>
          val inc  = digit + 1.U
          val wrap = last(j, digit, inc)
          when(in) { digit := Mux(wrap, 0.U, inc) }
          in && wrap
        }
        for {
          g <- s.g
          inc <- s.gInc
        } when(carry) { g := inc }
      }
    }
    when(done) { busy := false.B }
  }
}
