package snax.forge.shell

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Rand, Seeds, Sink, Source}
import snax.forge.shell.AcceleratorSpec.{CsrSpec, GroupSpec, LinkSpec, PortSpec, UnitSpec}

/** `<tag>_csr` (`AccCsr`) under random requests, against a cycle-level model of the protocol in its docstring
  * (`Model`).
  *
  * Per seed, `Layouts` CSR layouts of 1..3 groups, each a fold (with JOB_BEATS) or not, the offsets in CSR order, each
  * driven for `Cycles` cycles: a core that offers a request after a random gap and holds it until taken (reads, writes
  * and starts of a random group, in range, just past the last CSR and anywhere in 32 bits), `rsp.ready` and each
  * group's `starts` ready stalled at random, the groups' STATUS and COUNTER random every cycle. Checked every cycle:
  * `req.ready`, `rsp.valid`, `rsp.bits` while valid, each group's `starts.valid` and its JOBS and each JOB_BEATS in
  * unit order (a group of 0, 1 or 2 folds: JOBS and 1, or a JOB_BEATS per fold).
  */
class AccCsrRandomSpec extends AnyFlatSpec with ForgeTester {

  private val Layouts = 3
  private val Cycles  = 300
  private val Mask32  = (BigInt(1) << 32) - 1

  /** A layout of `folds.size` groups, group g of `folds(g)` folds (0: one unit without a JOB_BEATS; 1: one fold; 2: two
    * folds linked by a wire, a JOB_BEATS each).
    */
  private def layout(folds: Seq[Int]): AcceleratorSpec = {
    var at     = 0
    def next() = { at += 1; at - 1 }
    val groups = folds.zipWithIndex.map { case (n, g) =>
      val start  = CsrSpec("START", "rw", next(), Some(32))
      val jobs   = CsrSpec("JOBS", "rw", next(), Some(32))
      val ports  = Seq("a" -> PortSpec("in", "in", "int64", 1), "out" -> PortSpec("out", "out", "int64", 1))
      val units  = (0 until (n max 1)).map { k =>
        val beats = if (n > 0) Seq(CsrSpec("JOB_BEATS", "rw", next(), Some(32))) else Seq.empty
        UnitSpec(s"u${g}_$k", "accumulate", "TreeAccumulate", ports, None, Map.empty, 1, 1, 1, beats)
      }
      val status = CsrSpec("STATUS", "ro", next(), None)
      val count  = CsrSpec("COUNTER", "ro", next(), None)
      val links  = units
        .sliding(2)
        .collect { case Seq(a, b) =>
          LinkSpec(s"l$g", a.name -> "out", b.name -> "a", 0)
        }
        .toSeq
      GroupSpec(s"g$g", Seq(start, jobs, status, count), units, links)
    }
    AcceleratorSpec(
      "t",
      "p",
      "t",
      groups.map(g => g.units.head.name -> "a"),
      groups.map(g => g.units.last.name -> "out"),
      groups
    )
  }

  /** The protocol as `AccCsr`'s docstring states it. */
  private class Model(spec: AcceleratorSpec) {
    val csrs = spec.csrs
    val regs = Array.fill(csrs.size)(BigInt(0))
    var kept = Option.empty[BigInt] // a read's answer not yet taken
    def at(g: Int, n: String): Option[Int] = spec.groups(g).csr(n).map(_.offset)
    def config(g: Int): (BigInt, Seq[BigInt]) = {
      val beats = spec.groups(g).jobBeats.map(c => regs(c.offset))
      (regs(at(g, "JOBS").get), if (beats.isEmpty) Seq(BigInt(1)) else beats)
    }

    /** One cycle: req.ready, the response, which group starts; then the registers and the kept answer. */
    def cycle(
      req:        Option[(BigInt, BigInt, Boolean)],
      rspReady:   Boolean,
      startReady: Seq[Boolean],
      status:     Seq[BigInt],
      count:      Seq[BigInt]
    ) = {
      val start  = req.flatMap { case (addr, data, write) =>
        spec.groups.indices.find(g => write && addr == at(g, "START").get && data.testBit(0))
      }
      val ready  = kept.isEmpty && start.forall(startReady)
      val fired  = req.filter(_ => ready)
      val answer = fired.collect { case (addr, _, false) =>
        csrs
          .find(_.offset == addr)
          .map { c =>
            val g = spec.groups.indexWhere(_.window.contains(c))
            if (c.access == "ro") (if (c.name == "STATUS") status(g) else count(g)) else regs(c.offset)
          }
          .getOrElse(BigInt(0))
      }
      val rsp    = answer.orElse(kept)
      val starts = fired.flatMap(_ => start)
      fired.foreach { case (addr, data, write) =>
        csrs.find(c => c.offset == addr && c.access == "rw").foreach(c => if (write) regs(c.offset) = data)
      }
      kept = rsp.filter(_ => !rspReady)
      (ready, rsp, starts)
    }
  }

  private def request(r: Rand, spec: AcceleratorSpec): (BigInt, BigInt, Boolean) = {
    val last = spec.csrs.size
    val addr = r.pick(Seq.fill(7)(BigInt(r.between(0, last - 1))) ++ Seq(BigInt(r.between(last, last + 3)), r.word(32)))
    r.pick(Seq("read", "write", "start")) match {
      case "read"  => (addr, BigInt(0), false)
      case "write" => (addr, r.word(32), true)
      case _       =>
        val g = r.between(0, spec.groups.size - 1)
        (BigInt(spec.groups(g).csr("START").get.offset), r.word(32) | (if (r.chance(0.7)) 1 else 0), true)
    }
  }

  behavior of "<tag>_csr under random requests"

  for (seed <- Seeds())
    it should s"give req.ready, the response and each group's start as the protocol's model (seed $seed)" in
      Seeds.run(seed) { r =>
        for (l <- 0 until Layouts) {
          val spec   = layout(Seq.fill(r.between(1, 3))(r.between(0, 2)))
          val groups = spec.groups.size
          val model  = new Model(spec)
          val reqs   = IndexedSeq.fill(Cycles)(request(r, spec))
          val gap    = r.probability()
          val core   = new Source(Cycles, BigInt(_), () => !r.chance(gap))
          val rsp    = Sink.random(r.probability(), r)
          val starts = Seq.fill(groups)(Sink.random(r.probability(), r))
          test(new AccCsr(spec)) { dut =>
            for (c <- 0 until Cycles) {
              val clue    = s"layout $l (${spec.csrs.map(_.name).mkString(" ")}), cycle $c"
              core.cycle()
              val req     = if (core.valid) Some(reqs(core.bits.toInt)) else None
              val status  = Seq.fill(groups)(r.word(32))
              val count   = Seq.fill(groups)(r.word(32))
              dut.io.req.valid.poke(core.valid.B)
              req.foreach { case (addr, data, write) =>
                dut.io.req.bits.addr.poke(addr.U)
                dut.io.req.bits.data.poke(data.U)
                dut.io.req.bits.write.poke(write.B)
              }
              dut.io.rsp.ready.poke(rsp.cycle().B)
              val readies = starts.map(_.cycle())
              readies.zipWithIndex.foreach { case (v, g) => dut.io.starts(g).ready.poke(v.B) }
              status.zipWithIndex.foreach { case (v, g) => dut.io.status(g).poke(v.U) }
              count.zipWithIndex.foreach { case (v, g) => dut.io.count(g).poke(v.U) }

              val configs                  = (0 until groups).map(model.config)
              val (ready, answer, started) = model.cycle(req, rsp.isReady, readies, status, count)
              assert(dut.io.req.ready.peek().litToBoolean == ready, s"$clue: req.ready, the model $ready (req $req)")
              assert(dut.io.rsp.valid.peek().litToBoolean == answer.isDefined, s"$clue: rsp.valid, the model $answer")
              answer.foreach(a => assert(dut.io.rsp.bits.data.peek().litValue == (a & Mask32), s"$clue: rsp.bits"))
              for (g <- 0 until groups) {
                val v     = dut.io.starts(g).valid.peek().litToBoolean
                assert(v == started.contains(g), s"$clue: group $g's start ${v}, the model ${started.mkString}")
                assert(dut.io.starts(g).bits.jobs.peek().litValue == (configs(g)._1 & Mask32), s"$clue: $g's JOBS")
                val beats = dut.io.starts(g).bits.jobBeats +: dut.io.starts(g).bits.beats.map(_.toSeq).getOrElse(Nil)
                assert(beats.map(_.peek().litValue) == configs(g)._2.map(_ & Mask32), s"$clue: $g's JOB_BEATS")
              }
              core.fired(ready)
              dut.clock.step()
            }
          }
        }
      }
}
