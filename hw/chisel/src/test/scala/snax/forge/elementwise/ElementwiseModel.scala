package snax.forge.elementwise

/** Software model of one ALU lane -- the golden reference for the Chisel side.
  *
  * This is the Scala counterpart of what `build.verify` does in Python: compute the answer independently, then require
  * the hardware to match it bit for bit. Random operands are only worth driving if there is something to compare
  * against that was not derived from the design under test, and this is it.
  *
  * Arithmetic is done in `BigInt` and masked at the end, which is what makes wrapping explicit rather than incidental.
  * `BigInt`'s bitwise operators already use two's complement semantics at arbitrary precision, so `(a - b) & mask`
  * yields the wrapped difference for `b > a` without a special case.
  */
object ElementwiseModel {

  /** Fixed seed, so a failure is reproducible. If a random case ever fails, note the operand values from the clue
    * string rather than re-running and hoping for the same draw.
    */
  val Seed: Int = 0x5ecd

  def mask(dataWidth: Int): BigInt = (BigInt(1) << dataWidth) - 1

  /** Reinterpret an unsigned bit pattern as two's complement. */
  def asSigned(x: BigInt, dataWidth: Int): BigInt = {
    val half = BigInt(1) << (dataWidth - 1)
    if (x >= half) x - (BigInt(1) << dataWidth) else x
  }

  /** Operand pairs worth trying regardless of the random draw.
    *
    * Uniform random over `dataWidth` bits essentially never produces 0, 1, or the all-ones pattern, yet those are where
    * wrapping and signed comparison actually change behaviour. Random draws cover the bulk; these cover the corners.
    */
  def edgePairs(dataWidth: Int): Seq[(BigInt, BigInt)] = {
    val m    = mask(dataWidth)
    val sign = BigInt(1) << (dataWidth - 1)
    Seq(
      (BigInt(0), BigInt(0)),
      (BigInt(0), BigInt(1)), // 0 - 1 wraps to all ones
      (m, BigInt(1)),         // all ones + 1 wraps to zero
      (m, m),
      (sign, BigInt(1)),      // smallest signed value against a positive one
      (sign - 1, sign),       // largest positive against smallest negative
      (BigInt(1), BigInt(0))
    )
  }

  /** `NumTests` random pairs, preceded by the edge cases. */
  def operandPairs(dataWidth: Int, numRandom: Int, rng: scala.util.Random): Seq[(BigInt, BigInt)] =
    edgePairs(dataWidth) ++ Seq.fill(numRandom)((BigInt(dataWidth, rng), BigInt(dataWidth, rng)))

  /** Expected result of `op` on two unsigned bit patterns. */
  def apply(op: Int, a: BigInt, b: BigInt, dataWidth: Int, signed: Boolean = false): BigInt = {
    val m = mask(dataWidth)
    op match {
      case ElementwiseOp.Add => (a + b) & m
      case ElementwiseOp.Sub => (a - b) & m
      // Only the low half is kept, which is identical for signed and unsigned
      // operands -- hence no `signed` branch here.
      case ElementwiseOp.Mul => (a * b) & m
      case ElementwiseOp.And => a & b
      case ElementwiseOp.Or  => a | b
      case ElementwiseOp.Xor => a ^ b
      case ElementwiseOp.Min => if (lessThan(a, b, dataWidth, signed)) a else b
      case ElementwiseOp.Max => if (lessThan(a, b, dataWidth, signed)) b else a
      case other             => throw new IllegalArgumentException(s"unknown op encoding: $other")
    }
  }

  private def lessThan(a: BigInt, b: BigInt, dataWidth: Int, signed: Boolean): Boolean =
    if (signed) asSigned(a, dataWidth) < asSigned(b, dataWidth) else a < b

  /** Clue string attached to every expect, so a failure names the case that produced it. */
  def clue(op: Int, a: BigInt, b: BigInt, lane: Int = -1): String = {
    val where = if (lane >= 0) s"lane $lane, " else ""
    f"$where%sop ${ElementwiseOp.name(op)}%s, a=0x$a%x, b=0x$b%x"
  }
}
