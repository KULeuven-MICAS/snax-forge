package snax.forge

/** A Decoupled producer of `count` beats (`beat(t)`), in plain Scala: a spec pokes `valid` and `bits` and reports what
  * fired. Once `valid` is raised it stays up with the same bits until it fires (the Decoupled rule); while it is down,
  * `raise(c)` decides in each cycle c whether to raise it, as long as beats are left.
  *
  * Per cycle c: `cycle(c)` first, then poke `valid` and `bits`, then `fired(ready)` with the DUT's ready.
  */
class Source(val count: Int, beat: Int => BigInt, raise: Int => Boolean) {

  /** Beats taken so far. */
  var taken: Int = 0

  private var up = false

  def valid: Boolean = up

  def bits: BigInt = if (up) beat(taken) else BigInt(0)

  def cycle(c: Int): Unit = if (!up && taken < count) up = raise(c)

  /** Whether the beat fired this cycle. */
  def fired(ready: Boolean): Boolean = {
    val f = up && ready
    if (f) { taken += 1; up = false }
    f
  }
}

object Source {

  /** Valid raised with probability `1 - gap` each cycle it is down. */
  def random(beats: Seq[BigInt], gap: Double, rand: Rand): Source = {
    val v = beats.toIndexedSeq
    new Source(v.size, v, _ => !rand.chance(gap))
  }
}

/** A Decoupled consumer, in plain Scala: `ready(c)` decides cycle c's ready, the beats that fire are kept in `got`.
  *
  * Per cycle c: `cycle(c)` first, then poke `ready`, then `fired(valid, bits)` with the DUT's valid and bits.
  */
class Sink(ready: Int => Boolean) {

  var got: Vector[BigInt] = Vector.empty

  private var r = false

  def isReady: Boolean = r

  def cycle(c: Int): Boolean = { r = ready(c); r }

  def fired(valid: Boolean, bits: => BigInt): Boolean = {
    val f = valid && r
    if (f) got :+= bits
    f
  }
}

object Sink {

  /** Ready low with probability `stall` each cycle. */
  def random(stall: Double, rand: Rand): Sink = new Sink(_ => !rand.chance(stall))
}
