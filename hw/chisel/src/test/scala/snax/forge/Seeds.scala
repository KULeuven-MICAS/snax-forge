package snax.forge

import scala.util.Random

/** The seeds of the random specs: a fixed list, or the one `SNAX_FORGE_SEED` names. A random spec runs one test per
  * seed, named after it, and `run` puts the seed in front of every failure, so a failing seed is rerun alone with
  * `SNAX_FORGE_SEED=<seed>`.
  */
object Seeds {

  val Env = "SNAX_FORGE_SEED"

  val all: Seq[Long] = Seq(1L, 2L, 3L, 0x5ecdL, 0xc0ffeeL)

  def apply(): Seq[Long] = sys.env.get(Env).map(s => Seq(s.trim.toLong)).getOrElse(all)

  /** `body` with a fresh `Rand` of `seed`; any failure is rethrown with the seed and the command that reruns it. */
  def run(seed: Long)(body: Rand => Unit): Unit =
    try body(new Rand(seed))
    catch {
      case e: Throwable => throw new AssertionError(s"seed $seed (rerun with $Env=$seed): ${e.getMessage}", e)
    }
}

/** Random values for one seed. */
class Rand(val seed: Long) {

  val rng = new Random(seed)

  /** An int in `lo` to `hi`, both included. */
  def between(lo: Int, hi: Int): Int = lo + rng.nextInt(hi - lo + 1)

  def pick[T](xs: Seq[T]): T = xs(rng.nextInt(xs.size))

  def chance(p: Double): Boolean = rng.nextDouble() < p

  /** A gap or stall probability for one run. */
  def probability(): Double = pick(Seq(0.0, 0.3, 0.7))

  /** A `width`-bit value; one in four is an edge value: 0, 1, all ones or the sign bit alone. */
  def word(width: Int = 64): BigInt =
    if (chance(0.25)) pick(Seq(BigInt(0), BigInt(1), (BigInt(1) << width) - 1, BigInt(1) << (width - 1)))
    else BigInt(width, rng)
}
