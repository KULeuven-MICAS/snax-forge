package snax.forge.ctrl

/** SnaxGroupRule as snax-gvsoc's `snax/snax_group.hpp` states it, in Scala: the golden model of `GroupCtrl`. Written
  * the C++ way, not the Chisel way: the units in Kahn's order over the links (its own, not `GroupDesc`'s), each unit
  * counting its firings up to n_k (its chain's JOBS · T for a sink, n_succ · T_k before it), a firing giving a result
  * when its index plus one is a multiple of T_k, results riding an L-slot pipeline that freezes when its head has no
  * room, the II counting from the cycle of the last firing, the class judged at the sinks, and the run ending when
  * every unit has fired its n, nothing is in flight and every buffer is empty.
  *
  * Several chains (D189): each a path of links from a head to a sink, the sinks in unit order the outputs, each with
  * its JOBS and its room; `forks`, the units each forked input feeds (heads). A forked input's units fire together: the
  * cycle is decided with no unit blocked, then again with every unit that fires blocked while another unit of its fork
  * does not fire, until no fork is split (snax-gvsoc's `snax/snax_group_rule.py`).
  *
  * Per cycle: `start(...)` first when a start is offered, then `step(inputs, rooms, seeds)`, which steps the rule only
  * in a busy cycle (None otherwise). A seeded unit (`UnitDesc.seeded`) takes `inputs(k)` for its other outer inputs and
  * `seeds(k)` for its seed: its job's first beat waits for the seed (fire = a.valid ∧ (¬first ∨ seed.valid)), and that
  * firing takes it (`UnitStep.seed`).
  */
class GroupModel(val units: Seq[UnitDesc], val links: Seq[Link], val forks: Seq[Seq[Int]] = Nil) {

  case class UnitStep(
    fire:   Boolean,
    result: Boolean,
    push:   Boolean,
    read:   Boolean,
    lane:   Int,
    first:  Boolean,
    frozen: Boolean,
    seed:   Boolean = false
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

  /** The sinks in unit order, the outputs; per unit its chain's output. */
  val sinks: Seq[Int] = units.indices.filter(outs(_).isEmpty)
  private val chain = units.indices.map { k =>
    var x = k
    while (outs(x).nonEmpty) x = links(outs(x).head).to
    sinks.indexOf(x)
  }

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

  /** Each unit's firings in a run of `jobs` (one per output), unit k a result every t(k) firings. */
  def firings(jobs: Seq[Long], t: Seq[Long]): Seq[Long] = {
    val c = Array.fill(n)(0L)
    for (k <- order.reverse) c(k) = outs(k).headOption.map(i => c(links(i).to)).getOrElse(jobs(chain(k))) * t(k)
    c.toSeq
  }

  def start(jobs: Long, t: Seq[Long]): String = start(Seq.fill(sinks.size)(jobs), t)

  def start(jobs: Seq[Long], t: Seq[Long]): String =
    if (!idle) "refused"
    else if (jobs.contains(0L) || t.contains(0L)) "ignored"
    else {
      started = true; startAt = cycle; endAt = -1; beats = t
      counts  = firings(jobs, t)
      u.foreach { x => x.firings = 0; x.lane = 0 }
      classes = Array.fill(4)(0)
      "taken"
    }

  private def finished: Boolean =
    u.indices.forall(k => u(k).firings == counts(k) && !u(k).slots.contains(true)) && l.forall(_.occupancy == 0)

  def step(inputs: Seq[Boolean], room: Boolean, seeds: Seq[Boolean]): Option[Step] =
    step(inputs, Seq.fill(sinks.size)(room), seeds)

  def step(inputs: Seq[Boolean], room: Boolean): Option[Step] = step(inputs, Seq.fill(sinks.size)(room), Nil)

  /** One decision of the cycle, changing no state, the units in `blocked` not firing: each unit's step, busy and waits.
    */
  private def decide(
    outer:   Seq[Boolean],
    rooms:   Seq[Boolean],
    blocked: Set[Int]
  ): (Array[UnitStep], Boolean, Boolean) = {
    val avail, wants, wouldFire, gives, head, offer = Array.fill(n)(false)
    for (k <- order) {
      val c      = units(k); val us = u(k)
      val linked = ins(k).forall(i => if (links(i).depth == 0) offer(links(i).from) else l(i).occupancy > 0)
      avail(k) = outer(k) && linked
      val iiOk = !us.hasFired || cycle >= us.lastFire + c.ii
      wants(k)     = us.firings < counts(k) && iiOk            && !blocked(k)
      wouldFire(k) = wants(k)               && avail(k)        && (c.span == 1 || us.lane == c.span - 1)
      gives(k)     = wouldFire(k)           && (us.firings + 1) % beats(k) == 0
      head(k)      = c.latency > 0          && us.slots(c.latency - 1)
      offer(k)     = if (c.latency > 0) head(k) else gives(k)
    }
    val st = Array.fill(n)(UnitStep(false, false, false, false, 0, false, false))
    var isBusy, waits = false
    for (k <- order.reverse) {
      val c        = units(k); val us = u(k)
      val mine     = if (sinks.contains(k)) rooms(sinks.indexOf(k)) else true
      val unitRoom = mine && outs(k).forall { i =>
        val take = st(links(i).to).fire
        if (links(i).depth == 0) take else l(i).occupancy < links(i).depth || take
      }
      val frozen   = (if (c.latency > 0) head(k) else gives(k)) && !unitRoom
      val first    = us.firings % beats(k) == 0
      var s        = UnitStep(false, false, false, false, 0, first, frozen)
      if (!frozen) {
        s      = s.copy(
          fire   = wouldFire(k),
          result = gives(k),
          push   = if (c.latency > 0) head(k) else gives(k),
          read   = wants(k) && avail(k),
          lane   = us.lane
        )
        val inGap = us.hasFired && cycle < us.lastFire + c.ii
        isBusy = isBusy || s.fire || s.read || inGap
        waits  = waits || (wants(k) && !outer(k))
      }
      st(k) = s.copy(seed = s.fire && first && c.seeded)
    }
    (st, isBusy, waits)
  }

  def step(inputs: Seq[Boolean], rooms: Seq[Boolean], seeds: Seq[Boolean]): Option[Step] = {
    if (!busy) { cycle += 1; return None }
    val outer               = units.indices.map { k =>
      inputs(k) && (!units(k).seeded || u(k).firings % beats(k) != 0 || seeds.lift(k).getOrElse(true))
    }
    // NB (D189): a forked input's units fire together; one that would fire waits while another cannot
    var blocked             = Set.empty[Int]
    var d                   = decide(outer, rooms, blocked)
    var held                = forks.flatMap(f => if (f.forall(d._1(_).fire)) Nil else f.filter(d._1(_).fire)).toSet
    while (held.nonEmpty) {
      blocked ++= held
      d    = decide(outer, rooms, blocked)
      held = forks.flatMap(f => if (f.forall(d._1(_).fire)) Nil else f.filter(d._1(_).fire)).toSet
    }
    val (st, isBusy, waits) = d
    for (k <- units.indices) {
      val c = units(k); val us = u(k); val s = st(k)
      if (!s.frozen) {
        for (i <- c.latency - 1 until 0 by -1) us.slots(i) = us.slots(i - 1)
        if (c.latency > 0) us.slots(0) = s.result
      }
      if (s.fire) { us.hasFired = true; us.lastFire = cycle; us.firings += 1; us.lane = 0 }
      else if (s.read) us.lane += 1
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
    val cls = if (sinks.exists(st(_).frozen)) 2 else if (isBusy) 0 else if (waits) 1 else 3
    classes(cls) += 1
    val end = finished
    if (end) endAt = cycle
    cycle += 1
    Some(Step(st.toSeq, linkSteps, cls, end))
  }
}
