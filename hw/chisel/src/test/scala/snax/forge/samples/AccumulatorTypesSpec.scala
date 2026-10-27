package snax.forge.samples

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits}

/** The Accumulator's types on fixed vectors (D143): each lane cast to the output type (sign-extended from a signed
  * input) before the tree, the tree and the register wrapping at the output type. Each sum is NumPy's
  * `x.astype(out).sum(dtype=out)`, taken once through the tree (one beat of all the values) and once through the
  * register (one value per beat).
  */
class AccumulatorTypesSpec extends AnyFlatSpec with ForgeTester {

  /** The sum of `values` (numbers of `inType`) as a number of `outType`, both ways. */
  private def sum(inType: String, outType: String, values: Seq[BigInt], want: BigInt): Unit = {
    val out = IntType(outType)
    for (lanes <- Seq(values.size, 1)) {
      val beats = values.grouped(lanes).toSeq
      test(new Accumulator(inType, outType, lanes)) { dut =>
        dut.io.loopCount.poke(beats.size.U)
        dut.io.len.poke(beats.size.U)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        dut.io.out.ready.poke(true.B)
        for (beat <- beats) {
          dut.io.in.valid.poke(true.B)
          for ((v, l) <- beat.zipWithIndex) LaneBits.poke(dut.io.in.bits(l), v)
          dut.clock.step()
        }
        dut.io.in.valid.poke(false.B)
        dut.io.out.valid.expect(true.B)
        val bits = LaneBits.peek(dut.io.out.bits)
        val got = if (out.signed) LaneBits.signed(bits, out.width) else bits
        assert(got == want, s"$inType -> $outType, $lanes lanes: $got, NumPy $want")
      }
    }
  }

  private val Max32 = Int.MaxValue // 2^31 - 1

  behavior of "the Accumulator's types"

  it should "sign-extend int32 lanes into an int64 sum" in {
    sum("int32", "int64", Seq(Max32, 1, 5, -2).map(BigInt(_)), BigInt(2147483651L))
  }

  it should "wrap an int32 sum at int32" in {
    sum("int32", "int32", Seq(Max32, 1, 5, -2).map(BigInt(_)), BigInt(-2147483645))
  }

  it should "extend int8 by its sign and uint8 by zeros into int32" in {
    // 0xfd is -3 as int8 and 253 as uint8.
    sum("int8", "int32", Seq(BigInt(-3), BigInt(5)), BigInt(2))
    sum("uint8", "int32", Seq(BigInt(253), BigInt(5)), BigInt(258))
  }

  it should "sum int8 into int64 without wrapping at 8 bits" in {
    sum("int8", "int64", Seq.fill(4)(BigInt(100)), BigInt(400))
  }

  it should "refuse an outType narrower than inType" in {
    val e = intercept[Exception](circt.stage.ChiselStage.emitCHIRRTL(new Accumulator("int32", "int16", 1)))
    assert(e.getMessage.contains("outType int16 must be at least as wide as inType int32"), e.getMessage)
  }
}
