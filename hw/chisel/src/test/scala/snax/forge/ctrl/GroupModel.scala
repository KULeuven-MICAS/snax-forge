package snax.forge.ctrl

/** SnaxGroupRule as snax-gvsoc's `snax/snax_group.hpp` states it, in Scala: the golden model of `GroupCtrl`. Written
  * the C++ way, not the Chisel way: the units in Kahn's order over the links (its own, not `GroupDesc`'s), each unit
  * counting its firings up to n_k (JOBS · T for the sink, n_succ · T_k before it), a firing giving a result when its
  * index plus one is a multiple of T_k, results riding an L-slot pipeline that freezes when its head has no room, the
  * II counting from the cycle of the last firing, the class judged at the sink, and the run ending when every unit has
  * fired its n, nothing is in flight and every buffer is empty.
  *
  * Per cycle: `start(...)` first when a start is offered, then `step(inputs, room)`, which steps the rule only in a
  * busy cycle (None otherwise).
  */
class GroupModel(val units: Seq[UnitDesc], val links: Seq[Link]) {

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

  private val n    = units.size
  private val ins  = units.indices.map(k => links.indices.filter(links(_).to == k))
  private val outs = units.indices.map(k => links.indices.filter(links(_).from == k))
  private val order: Seq[Int] = {
    var left = units.indices.toList
    var out  = Vector.empty[Int]
    while (left.nonEmpty) {
      val k = left.find(k => ins(k).forall(i => out.contains(links(i).from))).get
      out :+= k; left = left.filterNot(_ == k)
    }
    out
  }
  private val sink = units.indices.find(outs(_).isEmpty).get

  private class UnitState(l: Int) {
    var slots    = Array.fill(l)(false)
    var hasFired = false
    var lastFire = 0L
    var firings  = 0L
    var lane     = 0
  }
  private class LinkState         { var occupancy, wptr, rptr = 0 }

  private val u       = units.map(x => new UnitState(x.latency))
  private val l       = links.map(_ => new LinkState)
  private var started = false
  private var startAt = 0L
  private var endAt   = -1L
  private var beats   = Seq.fill(n)(1L)
  private var counts  = Seq.fill(n)(0L)
  var cycle           = 0L
  var classes         = Array.fill(4)(0)

  def idle:        Boolean = !started || (endAt != -1 && endAt < cycle)
  def busy:        Boolean = started                  && cycle > startAt && (endAt == -1 || cycle <= endAt)
  def headFirings: Long    = u(order.head).firings

  def start(jobs: Long, t: Seq[Long]): String =
    if (!idle) "refused"
    else if (jobs == 0 || t.contains(0L)) "ignored"
    else {
      started = true; startAt = cycle; endAt = -1; beats = t
      val c = Array.fill(n)(0L)
      for (k <- order.reverse) c(k) = outs(k).headOption.map(i => c(links(i).to)).getOrElse(jobs) * t(k)
      counts  = c.toSeq
      u.foreach { x => x.firings = 0; x.lane = 0 }
      classes = Array.fill(4)(0)
      "taken"
    }

  private def finished: Boolean =
    u.indices.forall(k => u(k).firings == counts(k) && !u(k).slots.contains(true)) && l.forall(_.occupancy == 0)

  def step(inputs: Seq[Boolean], room: Boolean): Option[Step] = {
    if (!busy) { cycle += 1; return None }
    val avail, wants, wouldFire, gives, head, offer = Array.fill(n)(false)
    for (k <- order) {
      val c      = units(k); val us = u(k)
      val linked = ins(k).forall(i => if (links(i).depth == 0) offer(links(i).from) else l(i).occupancy > 0)
      avail(k) = inputs(k) && linked
      val iiOk = !us.hasFired || cycle >= us.lastFire + c.ii
      wants(k)     = us.firings < counts(k) && iiOk
      wouldFire(k) = wants(k)               && avail(k)        && (c.span == 1 || us.lane == c.span - 1)
      gives(k)     = wouldFire(k)           && (us.firings + 1) % beats(k) == 0
      head(k)      = c.latency > 0          && us.slots(c.latency - 1)
      offer(k)     = if (c.latency > 0) head(k) else gives(k)
    }
    val st = Array.fill(n)(UnitStep(false, false, false, false, 0, false, false))
    var isBusy, waits = false
    for (k <- order.reverse) {
      val c        = units(k); val us = u(k)
      val unitRoom = (k != sink || room)                        && outs(k).forall { i =>
        val take = st(links(i).to).fire
        if (links(i).depth == 0) take else l(i).occupancy < links(i).depth || take
      }
      val frozen   = (if (c.latency > 0) head(k) else gives(k)) && !unitRoom
      val first    = us.firings                                  % beats(k) == 0
      var s        = UnitStep(false, false, false, false, 0, first, frozen)
      if (!frozen) {
        s                              = s.copy(
          fire   = wouldFire(k),
          result = gives(k),
          push   = if (c.latency > 0) head(k) else gives(k),
          read   = wants(k) && avail(k),
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
      st(k) = s
    }
    val linkSteps = links.indices.map { i =>
      val d = links(i).depth; val ls = l(i)
      if (d == 0) LinkStep(false, false, 0, 0, 0)
      else {
        val load = st(links(i).from).push; val take = st(links(i).to).fire
        val r    = LinkStep(load, take, ls.wptr, ls.rptr, 0)
        if (load) ls.wptr = (ls.wptr + 1) % d
        if (take) ls.rptr = (ls.rptr + 1) % d
        ls.occupancy += (if (load) 1 else 0) - (if (take) 1 else 0)
        r.copy(occupancy  = ls.occupancy)
      }
    }
    val cls = if (st(sink).frozen) 2 else if (isBusy) 0 else if (waits) 1 else 3
    classes(cls) += 1
    val end = finished
    if (end) endAt = cycle
    cycle += 1
    Some(Step(st.toSeq, linkSteps, cls, end))
  }
}
