package snax.forge.shell

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Rand, Seeds, Sink, Source}

/** `CsrManager` under random requests, against a cycle-level model of the protocol in its docstring (`Model`).
  *
  * Per seed, `Managers` managers with 1..4 fields of 1..32 bits and `numRo` 2, each driven for `Cycles` cycles: a core
  * that offers a request after a random gap and holds it until taken (reads, writes and starts, in range, just past the
  * last CSR and anywhere in 32 bits), `rsp.ready` and `ctrl.ready` stalled at random, the `ro` inputs random every
  * cycle. Checked every cycle: `req.ready`, `rsp.valid`, `rsp.bits` while valid, `ctrl.valid` and `ctrl.bits`.
  */
class CsrManagerRandomSpec extends AnyFlatSpec with ForgeTester {

  private val Managers = 3
  private val Cycles   = 300
  private val NumRo    = 2
  private val Mask32   = (BigInt(1) << 32) - 1

  /** The protocol as `CsrManager`'s docstring states it: the fields then start read-write, `numRo` read-only after. */
  private class Model(widths: Seq[Int], numRo: Int) {
    val numRw    = widths.size + 1
    val regs     = Array.fill(numRw)(BigInt(0))
    var kept     = Option.empty[BigInt] // a read's answer not yet taken
    def ctrlBits = widths.zipWithIndex.map { case (w, i) => regs(i) & ((BigInt(1) << w) - 1) }

    /** One cycle: req.ready, the response, ctrl.valid; then the registers and the kept answer for the next cycle. */
    def cycle(req: Option[(BigInt, BigInt, Boolean)], rspReady: Boolean, ctrlReady: Boolean, ro: Seq[BigInt]) = {
      val start  = req.exists { case (addr, data, write) => write && addr == numRw - 1 && data.testBit(0) }
      val ready  = kept.isEmpty    && (!start || ctrlReady)
      val fired  = req.filter(_ => ready)
      val answer = fired.collect { case (addr, _, false) =>
        if (addr < numRw) regs(addr.toInt) else if (addr < numRw + numRo) ro((addr - numRw).toInt) else BigInt(0)
      }
      val rsp    = answer.orElse(kept)
      val ctrl   = fired.isDefined && start
      fired.foreach { case (addr, data, write) => if (write && addr < numRw) regs(addr.toInt) = data }
      kept = rsp.filter(_ => !rspReady)
      (ready, rsp, ctrl)
    }
  }

  private def request(r: Rand, numRw: Int): (BigInt, BigInt, Boolean) = {
    val last = numRw + NumRo
    val addr = r.pick(Seq.fill(7)(BigInt(r.between(0, last - 1))) ++ Seq(BigInt(r.between(last, last + 3)), r.word(32)))
    r.pick(Seq("read", "write", "start")) match {
      case "read"  => (addr, BigInt(0), false)
      case "write" => (addr, r.word(32), true)
      case _       => (BigInt(numRw - 1), r.word(32) | (if (r.chance(0.7)) 1 else 0), true)
    }
  }

  behavior of "CsrManager under random requests"

  for (seed <- Seeds())
    it should s"give req.ready, the response and ctrl as the protocol's model, every cycle (seed $seed)" in
      Seeds.run(seed) { r =>
        for (m <- 0 until Managers) {
          val widths = Seq.fill(r.between(1, 4))(r.between(1, 32))
          val fields = widths.zipWithIndex.map { case (w, i) => s"f$i" -> w }
          val model  = new Model(widths, NumRo)
          val reqs   = IndexedSeq.fill(Cycles)(request(r, model.numRw))
          val gap    = r.probability()
          val core   = new Source(Cycles, BigInt(_), () => !r.chance(gap))
          val rsp    = Sink.random(r.probability(), r)
          val ctrl   = Sink.random(r.probability(), r)
          test(new CsrManager(fields, NumRo, "t")) { dut =>
            for (c <- 0 until Cycles) {
              val clue = s"manager $m (fields $widths), cycle $c"
              core.cycle()
              val req  = if (core.valid) Some(reqs(core.bits.toInt)) else None
              val ro   = Seq.fill(NumRo)(r.word(32))
              dut.io.req.valid.poke(core.valid.B)
              req.foreach { case (addr, data, write) =>
                dut.io.req.bits.addr.poke(addr.U)
                dut.io.req.bits.data.poke(data.U)
                dut.io.req.bits.write.poke(write.B)
              }
              dut.io.rsp.ready.poke(rsp.cycle().B)
              dut.io.ctrl.ready.poke(ctrl.cycle().B)
              ro.zipWithIndex.foreach { case (v, i) => dut.io.ro(i).poke(v.U) }

              val bits                       = model.ctrlBits
              val (ready, answer, ctrlValid) = model.cycle(req, rsp.isReady, ctrl.isReady, ro)
              assert(dut.io.req.ready.peek().litToBoolean == ready, s"$clue: req.ready, the model $ready (req $req)")
              assert(dut.io.rsp.valid.peek().litToBoolean == answer.isDefined, s"$clue: rsp.valid, the model $answer")
              answer.foreach(a => assert(dut.io.rsp.bits.data.peek().litValue == (a & Mask32), s"$clue: rsp.bits"))
              assert(dut.io.ctrl.valid.peek().litToBoolean == ctrlValid, s"$clue: ctrl.valid, the model $ctrlValid")
              fields.zip(bits).foreach { case ((n, _), v) =>
                assert(dut.io.ctrl.bits(n).peek().litValue == v, s"$clue: ctrl.bits $n, the model $v")
              }
              core.fired(ready)
              dut.clock.step()
            }
          }
        }
      }
}
