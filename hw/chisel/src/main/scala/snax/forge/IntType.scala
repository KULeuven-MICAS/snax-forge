package snax.forge

import chisel3._

/** An integer element type as NumPy names it, "int8" ... "uint64": a width and a signedness (D142, D143).
  *
  * A block's types are design-time parameters: Python chooses them (`snax_forge/brm/dtypes.py`, an op's type included)
  * and `rtl.json` names them; a block only constructs from them (C§17). A signed type's lanes are `SInt`, an unsigned
  * one's `UInt`, so the shell wrapper's `asTypeOf` gives typed lanes.
  */
case class IntType(width: Int, signed: Boolean) {

  /** NumPy's name: int32, uint8. */
  def name: String = (if (signed) "int" else "uint") + width

  /** The short form in module names: i32, u8. */
  def tag: String = (if (signed) "i" else "u") + width

  /** A lane of this type: `SInt` when signed, else `UInt`. */
  def lane: Bits = if (signed) SInt(width.W) else UInt(width.W)
}

object IntType {

  val widths: Seq[Int] = Seq(8, 16, 32, 64)

  /** The eight names a type param takes. */
  val names: Seq[String] = for {
    s <- Seq("int", "uint")
    w <- widths
  } yield s"$s$w"

  private val Name = "(u?)int([0-9]+)".r

  def apply(name: String): IntType =
    name match {
      case Name(u, w) if widths.contains(w.toInt) => IntType(w.toInt, u.isEmpty)
      case _ => throw new IllegalArgumentException(s"type $name is not one of ${names.mkString(", ")}")
    }

  /** `x`, a lane of type `from` (or its bits), as a lane of `to`: extended by `from`'s signedness when `to` is wider,
    * its low bits when `to` is narrower or as wide (NumPy's `astype`).
    */
  def cast(x: Bits, from: IntType, to: IntType): Bits = {
    val u    = x.asUInt
    val bits =
      if (to.width <= from.width) u(to.width - 1, 0)
      else if (from.signed) x.asSInt.pad(to.width).asUInt
      else u.pad(to.width)
    if (to.signed) bits.asSInt else bits
  }
}
