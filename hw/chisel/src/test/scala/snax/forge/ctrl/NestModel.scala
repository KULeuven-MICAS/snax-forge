package snax.forge.ctrl

/** SnaxUnit's rule as snax-gvsoc's `snax/snax_unit.hpp` states it (`SnaxUnitRule`), in Scala: the golden model of
  * `NestCtrl`. Written the C++ way, not the Chisel way: a run counts its firings up to J * T, a firing gives a result
  * when its index plus one is a multiple of T, results ride an L-slot pipeline that freezes when its head has no room,
  * the II is counted from the cycle of the last firing, and the run ends when every firing is done and nothing is in
  * flight.
  *
  * Per cycle: `start(...)` first when a start is offered (it answers as the rule does), then `step(valid, ready)`,
  * which steps the rule only in a busy cycle and says what happened.
  */
class NestModel(val latency: Int, val ii: Int) {

  case class Step(fire: Boolean, first: Boolean, outValid: Boolean, push: Boolean, busy: Boolean, done: Boolean)

  private var started  = false
  private var startAt  = 0L
  private var endAt    = -1L
  private var count    = 0L
  private var jobBeats = 1L
  private var firings  = 0L
  private var hasFired = false
  private var lastFire = 0L
  private var slots    = Array.fill(latency)(false)
  private var cycle    = 0L

  def idle: Boolean = !started || (endAt != -1 && endAt < cycle)
  def busy: Boolean = started                  && cycle > startAt && (endAt == -1 || cycle <= endAt)

  /** A start of `jobs` jobs of `t` beats offered this cycle: "taken", "ignored" (an empty run) or "refused" (busy). */
  def start(jobs: Long, t: Long): String =
    if (!idle) "refused"
    else if (jobs == 0 || t == 0) "ignored"
    else {
      started = true; startAt = cycle; endAt = -1; count = jobs * t; jobBeats = t; firings = 0
      "taken"
    }

  /** One cycle: whether every input has a beat and whether the output has room. */
  def step(valid: Boolean, ready: Boolean): Step = {
    val b = busy
    var s = Step(false, false, false, false, b, false)
    if (b) {
      val iiOk    = !hasFired || cycle >= lastFire + ii
      val canFire = firings < count    && iiOk         && valid
      val gives   = canFire            && (firings + 1) % jobBeats == 0
      val head    = latency > 0        && slots(latency - 1)
      val frozen  = if (latency > 0) head && !ready else gives && !ready
      val first   = firings             % jobBeats == 0
      // L = 0: out.valid is a firing that gives, whether or not the output is ready
      if (frozen) s = Step(false, first, if (latency > 0) head else gives, false, b, false)
      else {
        val push = if (latency > 0) head else gives
        for (i <- latency - 1 until 0 by -1) slots(i) = slots(i - 1)
        if (latency > 0) slots(0) = gives
        if (canFire) { hasFired = true; lastFire = cycle; firings += 1 }
        val outValid = if (latency > 0) head else gives
        s                         = Step(canFire, first, outValid, push && ready, b, false)
      }
      if (firings == count && !slots.contains(true)) { endAt = cycle; s = s.copy(done = true) }
    }
    cycle += 1
    s
  }
}
