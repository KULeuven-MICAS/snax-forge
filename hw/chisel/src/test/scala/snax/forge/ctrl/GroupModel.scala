package snax.forge.ctrl

/** SnaxGroupRule as snax-gvsoc's `snax/snax_group.hpp` states it, in Scala: the golden model of `GroupCtrl`. Written
  * the C++ way, not the Chisel way: each unit counts its firings up to n_k = JOBS · ∏ T_v (v ≥ k), a firing gives a
  * result when its index plus one is a multiple of T_k, results ride an L-slot pipeline that freezes when its head has
  * no room, the II counts from the cycle of the last firing, and the run ends when every unit has fired its n, nothing
  * is in flight and every buffer is empty.
  *
  * Per cycle: `start(...)` first when a start is offered, then `step(inputs, room)`, which steps the rule only in a
  * busy cycle (None otherwise).
  */
class GroupModel(val units: Seq[UnitDesc], val depths: Seq[Int]) {

  case class UnitStep(
    fire:   Boolean,
    result: Boolean,
    push:   Boolean,
    read:   Boolean,
    lane:   Int,
    first:  Boolean,
    frozen: Boolean
  )
  case class LinkStep(load: Boolean, take: Boolean, wslot: Int, rslot: Int, occupancy: Int)
  case class Step(units: Seq[UnitStep], links: Seq[LinkStep], cls: Int, end: Boolean)

  private val n = units.size

  private class UnitState(l: Int) {
    var slots    = Array.fill(l)(false)
    var hasFired = false
    var lastFire = 0L
    var firings  = 0L
    var lane     = 0
  }
  private class LinkState         { var occupancy, wptr, rptr = 0 }

  private val u       = units.map(x => new UnitState(x.latency))
  private val l       = depths.map(_ => new LinkState)
  private var started = false
  private var startAt = 0L
  private var endAt   = -1L
  private var beats   = Seq.fill(n)(1L)
  private var counts  = Seq.fill(n)(0L)
  var cycle           = 0L
  var classes         = Array.fill(4)(0)

  def idle:        Boolean = !started || (endAt != -1 && endAt < cycle)
  def busy:        Boolean = started                  && cycle > startAt && (endAt == -1 || cycle <= endAt)
  def headFirings: Long    = u.head.firings

  def start(jobs: Long, t: Seq[Long]): String =
    if (!idle) "refused"
    else if (jobs == 0 || t.contains(0L)) "ignored"
    else {
      started = true; startAt = cycle; endAt = -1; beats = t
      counts  = t.scanRight(jobs)(_ * _).init
      u.foreach { x => x.firings = 0; x.lane = 0 }
      classes = Array.fill(4)(0)
      "taken"
    }

  private def finished: Boolean =
    u.indices.forall(k => u(k).firings == counts(k) && !u(k).slots.contains(true)) && l.forall(_.occupancy == 0)

  def step(inputs: Seq[Boolean], room: Boolean): Option[Step] = {
    if (!busy) { cycle += 1; return None }
    val avail, wants, wouldFire, gives, head, offer = Array.fill(n)(false)
    for (k <- 0 until n) {
      val c      = units(k); val us = u(k)
      val linked = k == 0 || (if (depths(k - 1) == 0) offer(k - 1) else l(k - 1).occupancy > 0)
      avail(k) = inputs(k) && linked
      val iiOk = !us.hasFired || cycle >= us.lastFire + c.ii
      wants(k)     = us.firings < counts(k) && iiOk
      wouldFire(k) = wants(k)               && avail(k)        && (c.span == 1 || us.lane == c.span - 1)
      gives(k)     = wouldFire(k)           && (us.firings + 1) % beats(k) == 0
      head(k)      = c.latency > 0          && us.slots(c.latency - 1)
      offer(k)     = if (c.latency > 0) head(k) else gives(k)
    }
    val st = Array.fill(n)(UnitStep(false, false, false, false, 0, false, false))
    var unitRoom      = room
    var isBusy, waits = false
    for (k <- n - 1 to 0 by -1) {
      val c      = units(k); val us = u(k)
      val frozen = (if (c.latency > 0) head(k) else gives(k)) && !unitRoom
      val first  = us.firings                                  % beats(k) == 0
      var s      = UnitStep(false, false, false, false, 0, first, frozen)
      if (!frozen) {
        s                              = s.copy(
          fire   = wouldFire(k),
          result = gives(k),
          push   = if (c.latency > 0) head(k) else gives(k),
          read   = c.span > 1 && wants(k) && avail(k),
          lane   = us.lane
        )
        for (i <- c.latency - 1 until 0 by -1) us.slots(i) = us.slots(i - 1)
        if (c.latency > 0) us.slots(0) = s.result
        val inGap = us.hasFired && cycle < us.lastFire + c.ii
        isBusy = isBusy || s.fire || s.read || inGap
        waits  = waits || (wants(k) && !inputs(k))
      }
      if (s.fire) { us.hasFired = true; us.lastFire = cycle; us.firings += 1; us.lane = 0 }
      else if (s.read) us.lane += 1
      if (k > 0) unitRoom = if (depths(k - 1) == 0) s.fire else l(k - 1).occupancy < depths(k - 1) || s.fire
      st(k) = s
    }
    val links = depths.indices.map { k =>
      val d = depths(k); val ls = l(k)
      if (d == 0) LinkStep(false, false, 0, 0, 0)
      else {
        val load = st(k).push; val take = st(k + 1).fire
        val r    = LinkStep(load, take, ls.wptr, ls.rptr, 0)
        if (load) ls.wptr = (ls.wptr + 1) % d
        if (take) ls.rptr = (ls.rptr + 1) % d
        ls.occupancy += (if (load) 1 else 0) - (if (take) 1 else 0)
        r.copy(occupancy  = ls.occupancy)
      }
    }
    val cls = if (st(n - 1).frozen) 2 else if (isBusy) 0 else if (waits) 1 else 3
    classes(cls) += 1
    val end = finished
    if (end) endAt = cycle
    cycle += 1
    Some(Step(st.toSeq, links, cls, end))
  }
}
