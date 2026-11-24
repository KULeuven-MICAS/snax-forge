package snax.forge.shell

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Rand, Seeds, Sink, Source}
import snax.forge.elementwise.ElementwiseModel

/** A broadcast outer input in the wrapper on a seeded Mac, mvt's column form in small: one group of one accumulating
  * `Mac` of W lanes, `a` W lanes a beat, `b` broadcast (one lane a beat, from its reader or through its slicer at d_in
  * \= 64 / width), the seed W lanes of the output's type once a job. The model, from CONTRACTS, "Accelerator RTL spec":
  * `b`'s stream is one lane, lane 0 of its word (the bits above it ignored), given to every lane; job j's lane l is
  * seed(j, l) + Σ_t cast_out(cast_op(a(t, l)) · cast_op(b(t))), wrapping at the output's type. Per seed of `Seeds`:
  * random types, W, JOBS and JOB_BEATS, data with junk above `b`'s lane, the streamer's gaps and stalls, with and
  * without the slicer.
  */
class ShellBroadcastSpec extends AnyFlatSpec with ForgeTester {

  /** The accelerator: inputs a, b (broadcast, `dIn` > 1 sliced) and seed, output out, the CSRs from offset 0. */
  private def spec(t: String, op: String, out: String, lanes: Int, dIn: Int): AcceleratorSpec = {
    def csr(n: String, access: String, offset: Int, width: Int)          =
      ujson.Obj("name" -> n, "access" -> access, "offset" -> offset, "width" -> width)
    def port(name: String, dtype: String, extra: (String, ujson.Value)*) =
      name -> ujson.Obj.from(
        Seq[(String, ujson.Value)](
          "port"  -> name,
          "dir"   -> (if (name == "out") "out" else "in"),
          "dtype" -> dtype,
          "lanes" -> lanes
        ) ++ extra
      )
    val b = Seq[(String, ujson.Value)]("broadcast" -> true) ++ (if (dIn > 1) Seq("d_in" -> ujson.Num(dIn)) else Nil)
    val unit  = ujson.Obj(
      "name"     -> "mac",
      "brm"      -> "mac",
      "datapath" -> "Mac",
      "ports"    -> ujson.Obj(port("a", t), port("b", t, b: _*), port("seed", out), port("out", out)),
      "opType"   -> op,
      "params"   -> ujson.Obj("accumulate" -> true, "regOut" -> false),
      "timing"   -> ujson.Obj("latency" -> 1, "ii" -> 1, "span" -> 1),
      "csrs"     -> ujson.Arr(csr("JOB_BEATS", "rw", 2, 32))
    )
    val group = ujson.Obj(
      "name"  -> "mac",
      "csrs"  -> ujson.Arr(
        csr("START", "rw", 0, 1),
        csr("JOBS", "rw", 1, 32),
        csr("STATUS", "ro", 3, 2),
        csr("BUSY_COUNTER", "ro", 4, 32)
      ),
      "units" -> ujson.Arr(unit),
      "links" -> ujson.Arr()
    )
    AcceleratorSpec.parse(
      ujson.write(
        ujson.Obj(
          "design_point" -> "mac_bcast",
          "tag"          -> "mac_bcast",
          "accelerators" -> ujson.Arr(
            ujson.Obj(
              "name"    -> "mac",
              "inputs"  -> ujson.Arr("mac.a", "mac.b", "mac.seed"),
              "outputs" -> ujson.Arr("mac.out"),
              "groups"  -> ujson.Arr(group)
            )
          )
        )
      )
    )
  }

  private def run(r: Rand, sliced: Boolean): Unit = {
    val names        = IntType.names
    val t            = IntType(r.pick(names))
    val op           = IntType(r.pick(names.filter(IntType(_).width >= t.width)))
    val out          = IntType(r.pick(names))
    val lanes        = r.pick(Seq(1, 2, 4, 8))
    val dIn          = if (sliced) 64 / t.width else 1
    val jobs         = r.between(1, 3)
    val beats        = r.between(1, 5)
    val n            = jobs * beats
    val clue         = s"${t.name} at ${op.name} -> ${out.name}, $lanes lanes, d_in $dIn, $jobs jobs of $beats"
    val a            = Seq.fill(n, lanes)(r.word(t.width))
    val b            = Seq.fill(n)(r.word(t.width))
    val q            = Seq.fill(jobs, lanes)(r.word(out.width))
    // b's words: one lane each with junk above it, or dIn lanes a word, the last short
    val bWords       =
      if (!sliced) b.map(x => (BigInt(64 - t.width, r.rng) << t.width) | x)
      else b.grouped(dIn).map(LaneBits.pack(_, t.width)).toSeq
    val (gap, stall) = (r.probability(), r.probability())
    test(new ShellHarness(spec(t.name, op.name, out.name, lanes, dIn))) { dut =>
      val w = math.max(64, lanes * out.width)
      assert(
        dut.io.readers == Seq(math.max(64, lanes * t.width), 64, w) && dut.io.writers == Seq(w),
        s"$clue: readers ${dut.io.readers}, writers ${dut.io.writers}"
      )
      val d    = new ShellDriver(
        dut,
        Seq(
          Source.random(a.map(LaneBits.pack(_, t.width)), gap, r),
          Source.random(bWords, gap, r),
          Source.random(q.map(LaneBits.pack(_, out.width)), gap, r)
        ),
        Seq(Sink.random(stall, r))
      )
      d.write(dut.csr("JOBS"), jobs)
      d.write(dut.csr("JOB_BEATS"), beats)
      d.write(dut.csr("START"), 1)
      while ((d.read(dut.csr("STATUS")) & 1) != 0) {}
      val want = (0 until jobs).map { j =>
        LaneBits.pack(
          (0 until lanes).map { l =>
            val ps = (0 until beats).map { k =>
              val i = j * beats + k
              ElementwiseModel.cast(
                (ElementwiseModel.cast(a(i)(l), t, op) * ElementwiseModel.cast(b(i), t, op)) & LaneBits.mask(op.width),
                op,
                out
              )
            }
            (q(j)(l) + ps.sum) & LaneBits.mask(out.width)
          },
          out.width
        )
      }
      assert(d.taken == Seq(n, bWords.size, jobs), s"$clue: beats taken ${d.taken}")
      assert(d.outputs(0) == want, s"$clue: sums ${d.outputs(0).map(_.toString(16))}, want ${want.map(_.toString(16))}")
    }
  }

  behavior of "a broadcast outer input on a seeded Mac"

  for (seed <- Seeds()) {
    it should s"give every lane b's one lane, from its reader (seed $seed)" in Seeds.run(seed)(run(_, sliced = false))
    it should s"give every lane b's one lane, through its slicer (seed $seed)" in Seeds.run(seed)(run(_, sliced = true))
  }
}
