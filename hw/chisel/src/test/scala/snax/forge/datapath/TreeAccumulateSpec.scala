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
  * takes the beat's sum on a fire with `first`, adds it on a fire without, and holds otherwise. Seeded (D149, B2 R5),
  * at W = 1, 2 and 4: a random seed each cycle, and a fire with `first` takes seed + the beat's sum; matvec_acc's
  * worked example at W = 2, rows (1, 2, 3, 4) · p = (1, 2, 0, 1) from q = 1: 6, then 10.
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

  behavior of "TreeAccumulate with a seed"

  it should "give matvec_acc's worked example: q = 1, then 6, then 10" in {
    test(new TreeAccumulate("int64", "int64", 2, seed = true)) { dut =>
      for ((beat, q, first) <- Seq((Seq(1, 4), 1, true), (Seq(0, 4), 7, false))) {
        dut.io.fire.poke(true.B); dut.io.first.poke(first.B)
        beat.zipWithIndex.foreach { case (v, l) => LaneBits.poke(dut.io.in(l), v) }
        LaneBits.poke(dut.io.seed.get, q) // read on the first beat only
        dut.clock.step()
      }
      assert(LaneBits.signed(LaneBits.peek(dut.io.out), 64) == 10)
    }
  }

  for (seed <- Seeds())
    it should s"start each job from its seed at W = 1, 2 and 4 (seed $seed)" in Seeds.run(seed) { r =>
      for (lanes <- Seq(1, 2, 4)) {
        val ti   = IntType(r.pick(Seq("int64", "int32", "int8", "uint32")))
        val to   = IntType(if (ti.width == 64) ti.name else r.pick(Seq("int64", ti.name)))
        val clue = s"seeded, ${ti.name} -> ${to.name}, $lanes lanes"
        test(new TreeAccumulate(ti.name, to.name, lanes, seed = true)) { dut =>
          var acc = BigInt(0)
          for (c <- 0 until 32) {
            val beat  = Seq.fill(lanes)(r.word(ti.width))
            val q     = r.word(to.width)
            val fire  = r.chance(0.7)
            val first = c == 0 || r.chance(0.3)
            dut.io.fire.poke(fire.B)
            dut.io.first.poke(first.B)
            beat.zipWithIndex.foreach { case (v, l) => LaneBits.poke(dut.io.in(l), v) }
            LaneBits.poke(dut.io.seed.get, q)
            if (c > 0) assert(LaneBits.peek(dut.io.out) == acc, s"$clue, cycle $c: out")
            val s     = beat.map(ElementwiseModel.cast(_, ti, to)).sum
            if (fire) acc = ((if (first) q else acc) + s) & LaneBits.mask(to.width)
            dut.clock.step()
          }
        }
      }
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
