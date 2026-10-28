package snax.forge.datapath

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds}
import snax.forge.elementwise.ElementwiseModel

/** `TreeAccumulate`: its types on fixed vectors (D143: each lane cast to the output type, sign-extended from a signed
  * input, before the tree, the tree and the register wrapping at the output type; each sum NumPy's
  * `x.astype(out).sum(dtype=out)`, taken once through the tree, one beat of all the values, and once through the
  * register, one value per beat), then random beats with `fire` and `first` driven at random against the model: `acc`
  * takes the beat's sum on a fire with `first`, adds it on a fire without, and holds otherwise.
  */
class TreeAccumulateSpec extends AnyFlatSpec with ForgeTester {

  /** The sum of `values` (numbers of `inType`) as a number of `outType`, both ways. */
  private def sum(inType: String, outType: String, values: Seq[BigInt], want: BigInt): Unit = {
    val out = IntType(outType)
    for (lanes <- Seq(values.size, 1)) {
      val beats = values.grouped(lanes).toSeq
      test(new TreeAccumulate(inType, outType, lanes)) { dut =>
        for ((beat, k) <- beats.zipWithIndex) {
          dut.io.fire.poke(true.B)
          dut.io.first.poke((k == 0).B)
          for ((v, l) <- beat.zipWithIndex) LaneBits.poke(dut.io.in(l), v)
          dut.clock.step()
        }
        dut.io.fire.poke(false.B)
        val bits = LaneBits.peek(dut.io.out)
        val got = if (out.signed) LaneBits.signed(bits, out.width) else bits
        assert(got == want, s"$inType -> $outType, $lanes lanes: $got, NumPy $want")
      }
    }
  }

  private val Max32 = Int.MaxValue // 2^31 - 1

  behavior of "TreeAccumulate's types"

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
    val e = intercept[Exception](circt.stage.ChiselStage.emitCHIRRTL(new TreeAccumulate("int32", "int16", 1)))
    assert(e.getMessage.contains("outType int16 must be at least as wide as inType int32"), e.getMessage)
  }

  behavior of "TreeAccumulate under random fire and first"

  for (seed <- Seeds())
    it should s"hold, start and add the model's sums (seed $seed)" in Seeds.run(seed) { r =>
      for (_ <- 0 until 4) {
        val ti    = IntType(r.pick(IntType.names))
        val to    = IntType(r.pick(IntType.names.filter(n => IntType(n).width >= ti.width)))
        val lanes = r.pick(Seq(1, 2, 4, 8))
        val clue  = s"${ti.name} -> ${to.name}, $lanes lanes"
        test(new TreeAccumulate(ti.name, to.name, lanes)) { dut =>
          var acc = BigInt(0)
          for (c <- 0 until 32) {
            val beat  = Seq.fill(lanes)(r.word(ti.width))
            val fire  = r.chance(0.7)
            val first = c == 0 || r.chance(0.3)
            dut.io.fire.poke(fire.B)
            dut.io.first.poke(first.B)
            beat.zipWithIndex.foreach { case (v, l) => LaneBits.poke(dut.io.in(l), v) }
            if (c > 0) assert(LaneBits.peek(dut.io.out) == acc, s"$clue, cycle $c: out")
            val s     = beat.map(ElementwiseModel.cast(_, ti, to)).sum & LaneBits.mask(to.width)
            if (fire) acc = if (first) s else (acc + s) & LaneBits.mask(to.width)
            dut.clock.step()
          }
        }
      }
    }
}
