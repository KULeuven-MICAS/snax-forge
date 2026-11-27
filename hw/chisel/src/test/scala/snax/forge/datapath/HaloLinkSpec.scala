package snax.forge.datapath

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Rand, Seeds}

/** `HaloLink` against `HaloModel`, written from CONTRACTS, "Accelerator RTL spec" (D206): H = ⌈(T − 1) / W⌉ beats held,
  * the elements of the held beats oldest first then the incoming beat's, tap t lane l the element l + t, every tap read
  * in the cycle its input comes; on load the held beats shift by one. Its name, flops and refusals; the worked examples
  * (W 4, 1 and 2 at T 3, 3 and 5; W 8 at T 3 as bit slices of {incoming, held}; W 4's taps into axpbypcz, k2 2 and d 4,
  * a blur's first pixels); then per seed T in {2, 3, 5} at W in {1, 2, 4, 8, 32}, uint8, and int32 at W 4, random beats
  * and loads, every tap every cycle.
  */
class HaloLinkSpec extends AnyFlatSpec with ForgeTester {

  /** The block as "The block" states it: the held beats (bit patterns, oldest first), their taps and their shift. */
  private class HaloModel(taps: Int, lanes: Int) {
    val held: Int = math.ceil((taps - 1).toDouble / lanes).toInt
    private var beats = Seq.fill(held)(Seq.fill(lanes)(BigInt(0)))

    /** Tap t's lanes with `in` the incoming beat: e(l + t) of the held beats' elements then `in`'s. */
    def tapsOf(in: Seq[BigInt]): Seq[Seq[BigInt]] = {
      val e = beats.flatten ++ in
      Seq.tabulate(taps, lanes)((t, l) => e(l + t))
    }

    def load(in: Seq[BigInt]): Unit = beats = beats.tail :+ in
  }

  private def peekTaps(dut: HaloLink, taps: Int, lanes: Int): Seq[Seq[BigInt]] =
    Seq.tabulate(taps, lanes)((t, l) => LaneBits.peek(dut.io.taps(t)(l)))

  private def poke(dut: HaloLink, beat: Seq[BigInt]): Unit =
    beat.zipWithIndex.foreach { case (v, l) => LaneBits.poke(dut.io.in(l), v) }

  behavior of "HaloLink"

  it should "be named for its type, lanes and taps, hold ⌈(T − 1) / W⌉ beats and refuse T < 2 and W < 1" in {
    val sv = circt.stage.ChiselStage.emitCHIRRTL(new HaloLink(3, 8, "uint8"))
    assert(sv.contains("module HaloLink_u8_n8_t3"), sv.take(200))
    for (((t, w), h) <- Seq((3, 8) -> 1, (3, 1) -> 2, (5, 2) -> 2, (2, 1) -> 1, (5, 4) -> 1, (9, 4) -> 2)) {
      var got = 0
      circt.stage.ChiselStage.emitCHIRRTL { val m = new HaloLink(t, w, "uint8"); got = m.held; m }
      assert(got == h, s"T $t, W $w: H $got, not $h")
    }
    def refused(body: => HaloLink, text: String): Unit = {
      val e = intercept[Exception](circt.stage.ChiselStage.emitCHIRRTL(body))
      assert(e.getMessage.contains(text), e.getMessage)
    }
    refused(new HaloLink(1, 4, "uint8"), "taps must be at least 2 (a window and one shifted by an element), got 1")
    refused(new HaloLink(3, 0, "uint8"), "lanes must be at least 1, got 0")
  }

  /** Beats in order (values of `dtype`), a load each; from beat H + 1 on (1-indexed) the taps must be `want`'s next. */
  private def worked(taps: Int, lanes: Int, beats: Seq[Seq[Int]], want: Seq[Seq[Seq[Int]]]): Unit =
    test(new HaloLink(taps, lanes, "uint8")) { dut =>
      val h       = (taps - 1 + lanes - 1) / lanes
      var outputs = 0
      for ((b, i) <- beats.zipWithIndex) {
        poke(dut, b.map(BigInt(_)))
        dut.io.load.poke(true.B)
        if (i >= h) {
          val got = peekTaps(dut, taps, lanes)
          assert(got == want(outputs).map(_.map(BigInt(_))), s"W $lanes, T $taps, beat ${i + 1}: $got")
          outputs += 1
        }
        dut.clock.step()
      }
      assert(outputs == beats.size - h && outputs == want.size, s"W $lanes, T $taps: $outputs outputs")
    }

  it should "give the worked examples: W 4 and W 1 at T 3, W 2 at T 5, beats in − H outputs" in {
    worked(
      3,
      4,
      Seq(Seq(1, 2, 3, 4), Seq(5, 6, 7, 8), Seq(9, 10, 11, 12)),
      Seq(
        Seq(Seq(1, 2, 3, 4), Seq(2, 3, 4, 5), Seq(3, 4, 5, 6)),
        Seq(Seq(5, 6, 7, 8), Seq(6, 7, 8, 9), Seq(7, 8, 9, 10))
      )
    )
    worked(3, 1, Seq(1, 2, 3, 4).map(Seq(_)), Seq(Seq(1, 2, 3), Seq(2, 3, 4)).map(_.map(Seq(_))))
    worked(
      5,
      2,
      Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6), Seq(7, 8)),
      Seq(
        Seq(Seq(1, 2), Seq(2, 3), Seq(3, 4), Seq(4, 5), Seq(5, 6)),
        Seq(Seq(3, 4), Seq(4, 5), Seq(5, 6), Seq(6, 7), Seq(7, 8))
      )
    )
  }

  it should "give, at W 8 and T 3 on uint8, taps that are bits [0, 64), [8, 72) and [16, 80) of {incoming, held}" in
    Seeds.run(1) { r =>
      test(new HaloLink(3, 8, "uint8")) { dut =>
        var held = BigInt(0)
        for (beat <- 0 until 12) {
          val in   = Seq.fill(8)(r.word(8))
          poke(dut, in)
          dut.io.load.poke(true.B)
          val word = LaneBits.pack(in, 8) << 64 | held // 128 bits, the held beat in the low half
          for (t <- 0 until 3) {
            val got = LaneBits.pack(Seq.tabulate(8)(l => LaneBits.peek(dut.io.taps(t)(l))), 8)
            assert(got == ((word >> (8 * t)) & LaneBits.mask(64)), s"beat $beat, tap $t")
          }
          held = LaneBits.pack(in, 8)
          dut.clock.step()
        }
      }
    }

  it should "give a blur's first pixels through axpbypcz (k2 2, d 4): 2, 3, 4, 5 from W 4's first row" in {
    class Blur extends Module {
      val link = Module(new HaloLink(3, 4, "uint8"))
      val simd = Module(
        new SimdLanes(Seq.fill(3)("uint8"), "uint16", "uint8", 4, SimdLanes.Axpbypcz, k2 = Some(2), d = Some(4))
      )
      val io   = IO(new Bundle {
        val in   = Input(Vec(4, UInt(8.W)))
        val load = Input(Bool())
        val out  = Output(Vec(4, UInt(8.W)))
      })
      link.io.in := io.in
      link.io.load  := io.load
      simd.io.a     := link.io.taps(0)
      simd.io.b.get := link.io.taps(1)
      simd.io.c.get := link.io.taps(2)
      io.out        := simd.io.out
    }
    test(new Blur) { dut =>
      for ((b, i) <- Seq(Seq(1, 2, 3, 4), Seq(5, 6, 7, 8)).zipWithIndex) {
        b.zipWithIndex.foreach { case (v, l) => dut.io.in(l).poke(v.U) }
        dut.io.load.poke(true.B)
        if (i == 1) assert(dut.io.out.map(_.peek().litValue) == Seq(2, 3, 4, 5).map(BigInt(_)))
        dut.clock.step()
      }
    }
  }

  // The cross-check with SNAX-GVSoC: its 200 beats at W 8, T 3, uint8 (`halo_uint8_w8_t3_200` of snax-gvsoc's
  // vectors, which its `SnaxHaloLink` gives in its tb), every step's taps from the model alone
  snax.forge.SnaxDatapathVectors.all.flatMap(_.find(_.name == "halo_uint8_w8_t3_200")) match {
    case None    =>
      it should "give SNAX-GVSoC's taps on its 200 beats at W 8, T 3" in
        cancel(s"${snax.forge.SnaxDatapathVectors.Env} is not set, or has no halo_uint8_w8_t3_200")
    case Some(v) =>
      it should "give SNAX-GVSoC's taps on its 200 beats at W 8, T 3, the model alone" in {
        val model = new HaloModel(3, 8)
        assert(v.steps.size == 200)
        for ((s, k) <- v.steps.zipWithIndex) {
          val got = model.tapsOf(s.ins.head).flatten
          for ((w, i) <- s.out.zipWithIndex)
            assert(got(i) == (w & 0xff), s"beat $k, tap ${i / 8}, lane ${i % 8}: the model ${got(i)}, SNAX-GVSoC $w")
          if (s.load) model.load(s.ins.head)
        }
      }
  }

  /** The configurations: T in {2, 3, 5} at W in {1, 2, 4, 8, 32} on uint8, and at W 4 on int32. */
  private val Configs =
    (for {
      t <- Seq(2, 3, 5)
      w <- Seq(1, 2, 4, 8, 32)
    } yield (t, w, "uint8")) ++ Seq(2, 3, 5).map(t => (t, 4, "int32"))

  /** `cycles` cycles of random beats and loads, every tap against the model every cycle. */
  private def random(r: Rand, taps: Int, lanes: Int, dtype: String, cycles: Int): Unit = {
    val ty = IntType(dtype)
    test(new HaloLink(taps, lanes, dtype)) { dut =>
      val model = new HaloModel(taps, lanes)
      val p     = r.probability().max(0.3)
      for (c <- 0 until cycles) {
        val in   = Seq.fill(lanes)(r.word(ty.width))
        val load = r.chance(p)
        poke(dut, in)
        dut.io.load.poke(load.B)
        val want = model.tapsOf(in)
        for {
          t <- 0 until taps
          l <- 0 until lanes
        } {
          val got = LaneBits.peek(dut.io.taps(t)(l))
          assert(
            got == want(t)(l),
            s"$dtype, W $lanes, T $taps, cycle $c, tap $t, lane $l: $got, the model ${want(t)(l)}"
          )
        }
        if (load) model.load(in)
        dut.clock.step()
      }
    }
  }

  for (seed <- Seeds())
    it should s"give the model's taps every cycle at random beats and loads (seed $seed)" in Seeds.run(seed) { r =>
      for ((t, w, d) <- Configs) random(r, t, w, d, 40)
    }
}
