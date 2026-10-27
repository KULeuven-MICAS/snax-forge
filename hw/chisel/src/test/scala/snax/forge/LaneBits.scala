package snax.forge

import chisel3._
import chiseltest._

/** A lane's bit pattern, whatever its type: a value from 0 to 2^width - 1, poked into a `UInt` lane as is and into an
  * `SInt` lane as the two's complement number it encodes; peeked back the same way. A beat packs its lanes at each
  * lane's width, lane i at bits `[w(i+1)-1 : wi]`.
  */
object LaneBits {

  def mask(width: Int): BigInt = (BigInt(1) << width) - 1

  /** The two's complement number the `width`-bit pattern `v` encodes. */
  def signed(v: BigInt, width: Int): BigInt = {
    val x = v & mask(width)
    if (x.testBit(width - 1)) x - (BigInt(1) << width) else x
  }

  def poke(d: Data, v: BigInt): Unit =
    d match {
      case s: SInt => s.poke(signed(v, s.getWidth).S(s.getWidth.W))
      case u: UInt => u.poke((v & mask(u.getWidth)).U(u.getWidth.W))
      case other => throw new IllegalArgumentException(s"$other is not a lane")
    }

  def peek(d: Data): BigInt =
    d match {
      case s: SInt => s.peek().litValue & mask(s.getWidth)
      case u: UInt => u.peek().litValue
      case other => throw new IllegalArgumentException(s"$other is not a lane")
    }

  /** The lanes of `d`: a `Vec`'s elements, or `d` itself. */
  def lanes(d: Data): Seq[Data] =
    d match {
      case v: Vec[_] => v.toSeq.map(_.asInstanceOf[Data])
      case other => Seq(other)
    }

  /** `values` packed at `width` bits each, the first at the low end. */
  def pack(values: Seq[BigInt], width: Int): BigInt =
    values.zipWithIndex.map { case (v, l) => (v & mask(width)) << (width * l) }.sum

  /** A packed beat poked into `d`'s lanes. */
  def pokeBeat(d: Data, beat: BigInt): Unit = {
    val w = lanes(d).head.getWidth
    lanes(d).zipWithIndex.foreach { case (l, i) => poke(l, beat >> (w * i)) }
  }

  /** `d`'s lanes peeked and packed. */
  def peekBeat(d: Data): BigInt = {
    val w = lanes(d).head.getWidth
    pack(lanes(d).map(peek), w)
  }
}
