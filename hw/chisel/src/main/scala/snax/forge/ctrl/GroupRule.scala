package snax.forge.ctrl

import chisel3._
import chisel3.util._

/** One unit of a group's description (C§17, `accelerator.json`): its name, its timing (L, II, span), its outer inputs
  * (the streamer's ports it reads) and its T when fixed at design time (`params.T`; 1 for a unit without a fold), else
  * None: a JOB_BEATS from the start.
  */
case class UnitDesc(name: String, latency: Int, ii: Int, span: Int, inputs: Int, fixedT: Option[Int]) {
  require(latency >= 0, s"$name: latency must be 0 or more, got $latency")
  require(ii      >= 1, s"$name: ii must be 1 or more, got $ii")
  require(span    >= 1, s"$name: span must be 1 or more, got $span")
  require(inputs  >= 0, s"$name: inputs must be 0 or more, got $inputs")
}

/** A group's description: its units, head first, a chain (gen.chain), and each link's depth (one fewer than the units;
  * 0 a wire, >= 1 a buffer of that many beats). A unit after the head takes its link on one input besides its outer
  * ones; the last unit's output is the group's outer output.
  */
case class GroupDesc(name: String, units: Seq[UnitDesc], depths: Seq[Int]) {
  require(units.nonEmpty, s"group $name: no unit")
  require(depths.size == units.size - 1, s"group $name: ${depths.size} links for ${units.size} units, a chain")
  require(depths.forall(_ >= 0), s"group $name: a depth below 0")
  require(units.head.inputs >= 1, s"group $name: the head ${units.head.name} has no outer input")

  /** The units whose T comes from the start (JOB_BEATS), in unit order. */
  def open: Seq[UnitDesc] = units.filter(_.fixedT.isEmpty)

  /** The links that are buffers (depth >= 1), by link index: the ones with enables (a wire has none). */
  def buffers: Seq[Int] = depths.indices.filter(depths(_) > 0)
}

/** A start resolved: JOBS and each unit's T. */
class GroupConfig(n: Int) extends Bundle {
  val jobs = UInt(32.W)
  val ts   = Vec(n, UInt(32.W))
}

/** A unit's enables to its datapath (`ctrl` -> `<tag>_datapath`, the catalogue's pins), and what it gives: `fire` (a
  * beat taken from every input; at span > 1 the last lane's read; a plain Mac's `en` too), `first` (the beat fired or
  * read is its job's first, d = 0), `read` and `lane` (span > 1: lane `lane` read in place), `adv` (its pipeline
  * advances: not frozen) and `give` (a result pushed into its link or the outer output).
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

/** The control model of a group (D147; C§17's Control): snax-gvsoc's `SnaxGroupRule` (`snax/snax_group.hpp`) in Chisel,
  * line by line, over datapaths of data and enables only. Not a module: it builds its registers and logic in the module
  * that makes it (`GroupCtrl`, `<tag>_<group>_ctrl`; `NestCtrl`, its one-unit case, with B1's ports), which drives
  * `io.config`, the outer valids and `io.out.ready`, and reads the rest. Its per-unit signals are named after the unit
  * (`<unit>_fire`, `<unit>_d`, ...).
  *
  * A start of JOBS and each unit's T: unit k fires n_k = JOBS · ∏ T_v over itself and the units after it, so it gives
  * R_k = JOBS · ∏ T_v over the units after it results, one on every T_k-th firing (its beat digit d_k = T_k − 1). JOBS
  * 0 or any T 0 is the empty run: taken, nothing runs (`GroupCtrl` ends it without starting this). Busy from the cycle
  * after the start to the run's last cycle E, the last unit's R-th push, where `done` is high.
  *
  * Each cycle is decided in two passes, as `SnaxGroupRule::step`: forward, what each unit offers on its output whatever
  * its room (its head, or at L 0 a firing that gives), so each knows whether its input has a beat; backward from the
  * last unit, whether each is frozen (its head, or at L 0 a giving firing, has no room), fires and reads, and so
  * whether it takes, which is its producer's room (a wire: the take; a buffer: occupancy < D or the take). The outer
  * ports' join is per unit (C§17): input j's ready is its unit's firing, which needs every one of that unit's outer
  * valids. The outer output's valid is the last unit's offer, never its ready's. `cls` is the cycle's class, as the run
  * line counts it: busy 0, stall_in 1, stall_out 2, flush 3.
  */
class GroupRule(val desc: GroupDesc) {
  private val n     = desc.units.size
  private val descs = desc.units

  val io = Wire(new Bundle {
    val config = Flipped(Decoupled(new GroupConfig(n)))
    val in     = MixedVec(descs.map(u => Vec(u.inputs, new Handshake)))
    val out    = Flipped(new Handshake)
    val units  = MixedVec(descs.map(u => new UnitEnables(u.span)))
    val links  = MixedVec(desc.buffers.map(k => new LinkEnables(desc.depths(k))))
    val busy   = Output(Bool())
    val done   = Output(Bool())
    val cls    = Output(UInt(2.W))
  })

  // The group's state: busy, each unit's T and result count R_k (snax_group.hpp:361-367, start() 189-225); the C++
  // counts firings up to n_k = R_k · T_k, this counts results up to R_k and the beat digit d (firings = g · T + d)
  private val busy = RegInit(false.B)
  private val ts   = descs.map(u => RegInit(1.U(32.W)).suggestName(s"${u.name}_t"))
  private val rs   = descs.map(u => RegInit(0.U(32.W)).suggestName(s"${u.name}_r"))

  // Per unit (snax_group.hpp:370-377, UnitState): its beat digit d (⌈log₂ T⌉ bits for a fixed T, none at T 1), results
  // given g, the II gap (has_fired and last_fire), the pipeline (slots, v(L-1) the head), the lane
  private class Stage(val u: UnitDesc) {
    val d    = u.fixedT match {
      case Some(1) => WireDefault(0.U(1.W)).suggestName(s"${u.name}_d")
      case Some(t) => RegInit(0.U(log2Ceil(t).W)).suggestName(s"${u.name}_d")
      case None    => RegInit(0.U(32.W)).suggestName(s"${u.name}_d")
    }
    val g    = RegInit(0.U(32.W)).suggestName(s"${u.name}_given")
    val gap  = RegInit(0.U(log2Ceil(u.ii + 1).W)).suggestName(s"${u.name}_gap")
    val v    = RegInit(VecInit(Seq.fill(u.latency max 1)(false.B))).suggestName(s"${u.name}_v")
    val lane = RegInit(0.U(log2Ceil(u.span).W)).suggestName(s"${u.name}_lane")
  }
  private val st = descs.map(new Stage(_))

  // Per buffer (snax_group.hpp:379-384, LinkState): occupancy and the ring's write and read pointers
  // named after the link's ends, `<from>_<to>_...` (a wire's are unused)
  private def link(k: Int, what: String) = s"${descs(k).name}_${descs(k + 1).name}_$what"
  private val occ  = desc.depths.zipWithIndex.map { case (d, k) =>
    RegInit(0.U(log2Ceil(d + 1).W)).suggestName(link(k, "occupancy"))
  }
  private val wptr = desc.depths.zipWithIndex.map { case (d, k) =>
    RegInit(0.U(log2Ceil(d max 2).W)).suggestName(link(k, "wslot"))
  }
  private val rptr = desc.depths.zipWithIndex.map { case (d, k) =>
    RegInit(0.U(log2Ceil(d max 2).W)).suggestName(link(k, "rslot"))
  }

  // Forward (snax_group.hpp:238-254): what each unit offers whatever its room
  private val inputs, avail, wants, wouldFire, gives, head, offer = Array.fill(n)(WireDefault(false.B))
  for (k <- 0 until n) {
    val u = descs(k)
    val s = st(k)
    inputs(k) := io.in(k).map(_.valid).foldLeft(true.B)(_ && _)
    val linked = // :242-246
      if (k == 0) true.B
      else if (desc.depths(k - 1) == 0) offer(k - 1)
      else occ(k - 1) =/= 0.U
    avail(k)     := inputs(k) && linked // :247
    wants(k)     := busy      && s.g =/= rs(k) && s.gap === 0.U                                            // :248-249
    wouldFire(k) := wants(k)  && avail(k)      && (if (u.span == 1) true.B else s.lane === (u.span - 1).U) // :250
    gives(k) := wouldFire(k) && s.d === ts(k) - 1.U // :251
    head(k)  := (if (u.latency > 0) s.v(u.latency - 1) else false.B) // :252
    offer(k) := (if (u.latency > 0) head(k) else gives(k)) // :253
    Seq(avail(k) -> "avail", wants(k) -> "wants", gives(k) -> "gives", offer(k) -> "offer").foreach { case (b, w) =>
      b.suggestName(s"${u.name}_$w")
    }
  }

  // Backward (snax_group.hpp:257-303): frozen, fire, read, push; each unit's take is its producer's room
  private val room, frozen, fire, read, push = Array.fill(n)(WireDefault(false.B))
  room(n - 1) := io.out.ready // :257
  for (k <- n - 1 to 0 by -1) {
    val u = descs(k)
    frozen(k) := (if (u.latency > 0) head(k) else gives(k)) && !room(k)   // :264
    fire(k)   := wouldFire(k)                               && !frozen(k) // :266-268
    push(k)   := (if (u.latency > 0) head(k) else gives(k)) && !frozen(k) // :270
    read(k)   := (if (u.span > 1) wants(k) && avail(k) && !frozen(k) else false.B) // :271
    if (k > 0) // :296-302
      room(k - 1) := (if (desc.depths(k - 1) == 0) fire(k)
                      else occ(k - 1) < desc.depths(k - 1).U || fire(k))
    Seq(frozen(k) -> "frozen", fire(k) -> "fire", push(k) -> "give", read(k) -> "read").foreach { case (b, w) =>
      b.suggestName(s"${u.name}_$w")
    }
  }

  // The outer ports: input j's ready is its unit's firing; the output's valid the last unit's offer
  for {
    k <- 0 until n
    p <- io.in(k)
  } p.ready    := fire(k)
  io.out.valid := offer(n - 1)

  // The datapaths' enables and the buffers'
  for (k      <- 0 until n) {
    val e = io.units(k)
    e.fire  := fire(k)
    e.first := st(k).d === 0.U // :265
    e.read  := read(k)
    e.lane  := st(k).lane
    e.adv   := !frozen(k)
    e.give  := push(k)
  }
  for ((k, i) <- desc.buffers.zipWithIndex) {
    val e = io.links(i)
    e.load      := push(k)
    e.wslot     := (if (desc.depths(k) > 1) wptr(k) else 0.U)
    e.rslot     := (if (desc.depths(k) > 1) rptr(k) else 0.U)
    e.occupancy := occ(k) + push(k) - fire(k + 1)
  }

  // The class (snax_group.hpp:281-283, 331)
  private val isBusy = (0 until n).map(k => !frozen(k) && (fire(k) || read(k) || st(k).gap =/= 0.U)).reduce(_ || _)
  private val waits  = (0 until n).map(k => !frozen(k) && wants(k) && !inputs(k)).reduce(_ || _)
  io.cls := Mux(frozen(n - 1), 2.U, Mux(isBusy, 0.U, Mux(waits, 1.U, 3.U)))

  // The last unit's R-th push ends the run: then every unit has fired its n, nothing is in flight and every buffer is
  // empty, finished() (snax_group.hpp:387-410)
  private val pushedLast = RegInit(0.U(32.W)).suggestName(s"${descs.last.name}_pushed")
  private val done       = busy && push(n - 1) && pushedLast === rs(n - 1) - 1.U
  io.done := done
  io.busy := busy

  io.config.ready := !busy

  // The registers, as step() (snax_group.hpp:273-280 the pipeline, 285-295 the II and the lane, 306-329 the buffers)
  // and start() (189-225; the II gap carries over) change them
  for (k      <- 0 until n) {
    val u = descs(k)
    val s = st(k)
    when(fire(k)) {
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
  for ((d, k) <- desc.depths.zipWithIndex if d > 0) {
    when(push(k)) { wptr(k) := (if (d > 1) Mux(wptr(k) === (d - 1).U, 0.U, wptr(k) + 1.U) else 0.U) }
    when(fire(k + 1)) { rptr(k) := (if (d > 1) Mux(rptr(k) === (d - 1).U, 0.U, rptr(k) + 1.U) else 0.U) }
    occ(k) := occ(k) + push(k) - fire(k + 1)
  }

  when(io.config.fire) {
    val c     = io.config.bits
    val empty = c.jobs === 0.U || c.ts.map(_ === 0.U).reduce(_ || _)
    busy     := !empty
    ts.zip(c.ts).foreach { case (t, x) => t := x }
    // R_{n-1} = JOBS, R_k = R_{k+1} · T_{k+1}
    val r = Wire(Vec(n, UInt(32.W)))
    r(n - 1) := c.jobs
    for (k <- n - 2 to 0 by -1) r(k) := (r(k + 1) * c.ts(k + 1))(31, 0)
    rs.zip(r).foreach { case (x, y) => x := y }
    for (s <- st) {
      if (s.u.fixedT != Some(1)) s.d := 0.U
      s.g                            := 0.U
      s.lane                         := 0.U
    }
    pushedLast := 0.U
  }.otherwise {
    for (k <- 0 until n) when(fire(k)) {
      if (descs(k).fixedT != Some(1)) st(k).d := Mux(gives(k), 0.U, st(k).d + 1.U)
      when(gives(k)) { st(k).g := st(k).g + 1.U }
    }
    when(push(n - 1)) { pushedLast := pushedLast + 1.U }
    when(done) { busy := false.B }
  }
}
