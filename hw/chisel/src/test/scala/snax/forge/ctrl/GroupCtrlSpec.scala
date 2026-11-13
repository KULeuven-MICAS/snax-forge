package snax.forge.ctrl

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Seeds}

/** `GroupCtrl` against `GroupModel` (SnaxGroupRule, written from snax-gvsoc's `snax/snax_group.hpp`) on random groups:
  * chains of 1, 2 and 3 units, a third each (one unit: a vecadd or vsum accelerator's whole group), each of L 0-2, II
  * 1-2, span 1, 2 or 4, T 1-4 (fixed by `params.T` or a JOB_BEATS on its wire) and 0-2 outer inputs (the head at least
  * 1); links of depth 0, 1 or 2, half the chains in a shuffled unit order (the head, the sink and the topological order
  * other than the units' indices), the link on the input after a unit's outer ones; starts of J 0, 1 or 3 (and T 0 now
  * and then) offered at random times, while busy too, at job_width 3 or 32, JOBS and the T's on their wires drawn again
  * only while idle, as the csr holds their writes through a run; random input valid and output ready gaps. Every cycle:
  * the start's answer; busy and done; every input's ready (its unit's firing, the per-unit join of CONTRACTS,
  * "Accelerator RTL spec"); out.valid, poked with out.ready low and high (it never depends on it); per unit fire,
  * first, read, lane, adv, give; per buffer load, its slots and its occupancy; the class; the done pulse (high in a
  * run's last cycle E, and in the cycle an empty run's start is accepted). At each run's end the head's firings.
  * BUSY_COUNTER, which `<tag>_csr` keeps from busy, is AccCsrRandomSpec's and the shell specs'. The whole runs again
  * with seeded units: half the units with an outer input to spare (the head two, another one) seeded, their last outer
  * input the seed, its valid at random; a seed's ready is its unit's firing on its job's first beat (the model's
  * `seed`).
  */
class GroupCtrlSpec extends AnyFlatSpec with ForgeTester {

  private val Groups = 27
  private val Cycles = 400
  private var ended  = 0 // runs that reached their last cycle, every seed

  behavior of "GroupCtrl against SnaxGroupRule's model"

  for {
    seeded <- Seq(false, true)
    seed   <- Seeds()
  }
    it should s"give the model's cycles on random${if (seeded) " seeded" else ""} groups (seed $seed)" in Seeds.run(
      seed
    ) { r =>
      for (g <- 0 until Groups) {
        val n     = 1 + g % 3
        val chain = { // the units in chain order, head first
          val a = (0 until n).toArray
          if (r.chance(0.5)) for (i <- n - 1 to 1 by -1) {
            val j = r.between(0, i); val t = a(i); a(i) = a(j); a(j) = t
          }
          a.toSeq
        }
        val sink  = chain.last
        val units = (0 until n).map { k =>
          val ins = if (k == chain.head) r.between(1, 2) else r.between(0, 2)
          UnitDesc(
            s"u$k",
            r.between(0, 2),
            r.between(1, 2),
            r.pick(Seq(1, 2, 4)),
            ins,
            if (r.chance(0.5)) Some(r.between(1, 4)) else None,
            if (k == chain.head) Nil else Seq(ins)
          )
        }.map { u => // seeded: drawn only in the seeded run, so the unseeded one's groups stay as they were
          val spare = u.inputs >= (if (u.linked.isEmpty) 2 else 1)
          if (seeded && spare && r.chance(0.5)) u.copy(seeded = true) else u
        }
        val links = chain.zip(chain.tail).map { case (a, b) => Link(a, b, units(b).inputs, r.between(0, 2)) }
        val desc  = GroupDesc("rand", units, links)
        val clue0 = s"seed $seed, group $g: ${units.mkString(", ")}, links ${links.mkString(",")}"
        val model = new GroupModel(units, links)
        val pv    = r.probability(); val pr = r.probability()
        val jw    = r.pick(Seq(3, 32)) // job_width: J <= 3 and T <= 4 fit 3 bits
        test(new GroupCtrl(desc, "t", jw)) { dut =>
          val open  = desc.open
          var jobs  = 0
          var opent = open.map(_ => 1)
          for (c <- 0 until Cycles) {
            val clue   = s"$clue0, job_width $jw, cycle $c"
            val offer  = r.chance(0.1)
            // JOBS and each JOB_BEATS, the csr's wires: drawn again only while idle, held through a run
            if (model.idle) {
              jobs  = r.pick(Seq(0, 1, 3, 3, 3))
              opent = open.map(_ => if (r.chance(0.05)) 0 else r.between(1, 4))
            }
            dut.io.start.valid.poke(offer.B)
            dut.io.settings.jobs.poke(jobs.U)
            dut.io.settings.beats.zip(opent).foreach { case (p, t) => p.poke(t.U) }
            assert(high(dut.io.start.ready) == model.idle, s"$clue: start.ready")
            val answer = if (offer) {
              var next = 0
              val ts   = units.map(_.fixedT.map(_.toLong).getOrElse { next += 1; opent(next - 1).toLong })
              model.start(jobs, ts)
            } else "none"
            val valids = units.map(u => Seq.fill(u.inputs)(!r.chance(pv)))
            valids.flatten.zipWithIndex.foreach { case (v, k) => dut.io.in(k).valid.poke(v.B) }
            dut.io.out.ready.poke(false.B)
            val vLow   = high(dut.io.out.valid)
            dut.io.out.ready.poke(true.B)
            assert(high(dut.io.out.valid) == vLow, s"$clue: out.valid depends on out.ready")
            val ready  = !r.chance(pr)
            dut.io.out.ready.poke(ready.B)
            val busy   = model.busy
            assert(high(dut.io.busy) == busy, s"$clue: busy")
            val heads  = model.headFirings
            val step   = model.step(
              valids.zip(units).map { case (v, u) => (if (u.seeded) v.init else v).forall(identity) },
              ready,
              valids.zip(units).map { case (v, u) => !u.seeded || v.last }
            )
            val pulse  = answer == "ignored" || step.exists(_.end)
            assert(high(dut.io.pulse) == pulse, s"$clue: the done pulse, the model $pulse")
            step match {
              case None    =>
                for (k <- 0 until valids.flatten.size) assert(!high(dut.io.in(k).ready), s"$clue: input $k ready, idle")
                assert(!high(dut.io.out.valid), s"$clue: out.valid while idle")
              case Some(s) =>
                var at = 0
                for (k      <- 0 until n) {
                  val e = dut.io.units(k); val m = s.units(k); val uc = s"$clue, ${units(k).name}"
                  for (j <- 0 until units(k).inputs) {
                    val want = if (units(k).seeded && j == units(k).inputs - 1) m.seed else m.fire
                    assert(high(dut.io.in(at).ready) == want, s"$uc: ready"); at += 1
                  }
                  assert(high(e.fire) == m.fire, s"$uc: fire")
                  assert(high(e.give) == m.push, s"$uc: give")
                  assert(high(e.read) == m.read, s"$uc: read")
                  assert(high(e.adv) == !m.frozen, s"$uc: adv")
                  if (m.fire || m.read) assert(high(e.first) == m.first, s"$uc: first")
                  if (m.read) assert(int(e.lane) == m.lane, s"$uc: lane")
                }
                assert(high(dut.io.out.valid) == (s.units(sink).push || s.units(sink).frozen), s"$clue: out.valid")
                for ((k, i) <- desc.buffers.zipWithIndex) {
                  val d = links(k).depth; val e = dut.io.links(i); val m = s.links(k)
                  assert(high(e.load) == m.load, s"$clue, link $k: load")
                  if (m.load && d > 1) assert(int(e.wslot) == m.wslot, s"$clue, link $k: wslot")
                  if (m.take && d > 1) assert(int(e.rslot) == m.rslot, s"$clue, link $k: rslot")
                  assert(int(e.occupancy) == m.occupancy, s"$clue, link $k: occupancy")
                }
                assert(int(dut.io.cls) == s.cls, s"$clue: class")
                if (s.end) {
                  ended += 1
                  assert(heads + (if (s.units(chain.head).fire) 1 else 0) == model.headFirings, s"$clue: firings")
                }
            }
            dut.clock.step()
          }
        }
      }
    }

  it should "have ended enough random runs" in {
    info(s"$ended runs ended")
    assert(ended >= 2 * 60 * Seeds().size, s"$ended runs ended")
  }
}
