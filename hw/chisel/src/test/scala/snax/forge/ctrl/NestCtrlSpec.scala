package snax.forge.ctrl

import chisel3._
import chisel3.util.{Decoupled, RegEnable}
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Seeds}

/** `NestCtrl` with a test datapath of L registers on `adv` (L = 0: none), which carries each firing's index from `tag`
  * to `bits`, so a pushed result names the firing that gave it.
  */
class NestUnderTest(latency: Int, ii: Int, inputs: Int) extends Module {
  val ctrl = Module(new NestCtrl(latency, ii, inputs))
  val io   = IO(new Bundle {
    val config = Flipped(Decoupled(new NestConfig))
    val in     = Vec(inputs, new Handshake)
    val out    = Flipped(new Handshake)
    val tag    = Input(UInt(16.W))
    val bits   = Output(UInt(16.W))
    val fire   = Output(Bool())
    val first  = Output(Bool())
    val busy   = Output(Bool())
    val done   = Output(Bool())
  })
  ctrl.io.config <> io.config
  ctrl.io.in <> io.in
  ctrl.io.out <> io.out
  io.fire := ctrl.io.fire
  io.first := ctrl.io.first
  io.busy  := ctrl.io.busy
  io.done  := ctrl.io.done
  io.bits  := (0 until latency).foldLeft(io.tag)((d, _) => RegEnable(d, 0.U, ctrl.io.adv))
}

/** `NestCtrl` against `NestModel` (SnaxUnit's rule, snax-gvsoc `snax_unit.hpp`) under random input valids and output
  * stalls, for L in {0, 1, 2}, II in {1, 2}, T in {1, 3, 4} and J in {0, 1, 5} (J = 0: the empty run, ignored). Per
  * seed and configuration, `Runs` runs back to back, each started once the last has ended (a start while busy is
  * offered once at random too: refused). Checked every cycle: fire, first (on a fire), out.valid, busy, done, each
  * input's ready (the others' valids, never its own), and every pushed result is the firing that gave it: a job's last
  * beat, in order.
  */
class NestCtrlSpec extends AnyFlatSpec with ForgeTester {

  private val Runs = 4

  behavior of "NestCtrl against SnaxUnit's rule"

  for {
    latency <- Seq(0, 1, 2)
    ii      <- Seq(1, 2)
    seed    <- Seeds()
  }
    it should s"fire, push, be busy and end as the rule says (L $latency, II $ii, seed $seed)" in Seeds.run(seed) { r =>
      val inputs = r.pick(Seq(1, 2))
      test(new NestUnderTest(latency, ii, inputs)) { dut =>
        val model  = new NestModel(latency, ii)
        var cycle  = 0
        var firing = 0 // the run's firings so far
        var gives  = Vector.empty[Int]
        var pushed = Vector.empty[Int]
        for (run <- 0 until Runs) {
          val jobs    = r.pick(Seq(0, 1, 5))
          val t       = r.pick(Seq(1, 3, 4))
          val gap     = r.probability()
          val stall   = r.probability()
          val clue    = s"run $run: J $jobs, T $t, inputs $inputs, gap $gap, stall $stall"
          var started = false
          var ended   = false
          var limit   = 0
          firing = 0
          gives  = Vector.empty
          pushed = Vector.empty
          while (!ended) {
            assert(limit < 2000, s"$clue: no end")
            limit += 1
            // A start when idle; while busy, now and then one that must be refused
            val offer  = if (!started) true else model.busy && r.chance(0.05)
            dut.io.config.valid.poke(offer.B)
            dut.io.config.bits.jobs.poke(jobs.U)
            dut.io.config.bits.jobBeats.poke(t.U)
            val ready  = dut.io.config.ready.peek().litToBoolean
            val answer = if (offer) model.start(jobs, t) else ""
            if (offer) {
              assert(ready == (answer != "refused"), s"$clue, cycle $cycle: config.ready $ready, the rule $answer")
              if (!started) {
                started                        = true
                if (answer == "ignored") ended = true
              }
            }
            val valids = Seq.fill(inputs)(!r.chance(gap))
            val outRdy = !r.chance(stall)
            valids.zipWithIndex.foreach { case (v, k) => dut.io.in(k).valid.poke(v.B) }
            dut.io.out.ready.poke(outRdy.B)
            dut.io.tag.poke(firing.U)
            val s      = model.step(valids.forall(identity), outRdy)
            def high(b: Bool) = b.peek().litToBoolean
            val c = s"$clue, cycle $cycle"
            assert(high(dut.io.busy) == s.busy, s"$c: busy")
            assert(high(dut.io.fire) == s.fire, s"$c: fire ${high(dut.io.fire)}, the rule ${s.fire}")
            if (s.fire) assert(high(dut.io.first) == s.first, s"$c: first")
            assert(high(dut.io.out.valid) == s.outValid, s"$c: out.valid")
            assert(high(dut.io.done) == s.done, s"$c: done")
            for (k <- 0 until inputs) {
              val others = valids.zipWithIndex.filter(_._2 != k).forall(_._1)
              if (!s.busy) assert(!high(dut.io.in(k).ready), s"$c: input $k ready while idle")
              if (high(dut.io.in(k).ready)) assert(others, s"$c: input $k ready without the others' valids")
              if (s.fire) assert(high(dut.io.in(k).ready), s"$c: input $k not ready on a firing")
            }
            if (s.fire) {
              if ((firing + 1) % t == 0) gives :+= firing
              firing += 1
            }
            if (s.push) pushed :+= dut.io.bits.peek().litValue.toInt
            if (s.done) ended = true
            dut.clock.step()
            cycle += 1
          }
          dut.io.config.valid.poke(false.B)
          if (jobs > 0) {
            assert(firing == jobs * t, s"$clue: $firing firings")
            assert(pushed == gives, s"$clue: pushed $pushed, the firings that gave them $gives")
          }
          // Idle cycles between runs, the model's too: the II counts every cycle
          for (_ <- 0 until r.between(0, 2)) {
            (0 until inputs).foreach(k => dut.io.in(k).valid.poke(false.B))
            model.step(false, true)
            dut.clock.step()
            cycle += 1
          }
        }
      }
    }
}
