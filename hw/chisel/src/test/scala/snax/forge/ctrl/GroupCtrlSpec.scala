package snax.forge.ctrl

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Seeds}

/** `GroupCtrl` against `GroupModel` (SnaxGroupRule, written from snax-gvsoc's `snax/snax_group.hpp`) on random groups:
  * chains of 2 and 3 units, each of L 0-2, II 1-2, span 1, 2 or 4, T 1-4 (fixed by `params.T` or carried by the start)
  * and 0-2 outer inputs (the head at least 1); links of depth 0, 1 or 2, half the chains in a shuffled unit order (the
  * head, the sink and the topological order other than the units' indices), the link on the input after a unit's outer
  * ones; starts of J 0, 1 or 3 (and T 0 now and then) offered at random times, while busy too; random input valid and
  * output ready gaps. Every cycle: the start's answer; busy and done; every input's ready (its unit's firing, C§17's
  * per-unit join); out.valid, poked with out.ready low and high (it never depends on it); per unit fire, first, read,
  * lane, adv, give; per buffer load, its slots and its occupancy; the class. At each run's end the run line's totals.
  */
class GroupCtrlSpec extends AnyFlatSpec with ForgeTester {

  private val Groups = 20
  private val Cycles = 400
  private var ended  = 0 // runs that reached their last cycle, every seed

  behavior of "GroupCtrl against SnaxGroupRule's model"

  for (seed <- Seeds())
    it should s"give the model's cycles on random groups (seed $seed)" in Seeds.run(seed) { r =>
      for (g <- 0 until Groups) {
        val n     = if (g % 2 == 0) 2 else 3
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
        }
        val links = chain.zip(chain.tail).map { case (a, b) => Link(a, b, units(b).inputs, r.between(0, 2)) }
        val desc  = GroupDesc("rand", units, links)
        val clue0 = s"seed $seed, group $g: ${units.mkString(", ")}, links ${links.mkString(",")}"
        val model = new GroupModel(units, links)
        val pv    = r.probability(); val pr = r.probability()
        test(new GroupCtrl(desc, "t")) { dut =>
          def high(b: Bool) = b.peek().litToBoolean
          def int(u:  UInt) = u.peek().litValue.toInt
          val open = desc.open
          for (c <- 0 until Cycles) {
            val clue   = s"$clue0, cycle $c"
            val offer  = r.chance(0.1)
            val jobs   = r.pick(Seq(0, 1, 3, 3, 3))
            val opent  = open.map(_ => if (r.chance(0.05)) 0 else r.between(1, 4))
            dut.io.start.valid.poke(offer.B)
            dut.io.start.bits.jobs.poke(jobs.U)
            dut.io.start.bits.jobBeats.poke((opent.headOption.getOrElse(1)).U)
            dut.io.start.bits.beats.foreach(v => v.zip(opent.drop(1)).foreach { case (p, t) => p.poke(t.U) })
            assert(high(dut.io.start.ready) == model.idle, s"$clue: start.ready")
            if (offer) {
              var next = 0
              val ts   = units.map(_.fixedT.map(_.toLong).getOrElse { next += 1; opent(next - 1).toLong })
              model.start(jobs, ts)
            }
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
            model.step(valids.map(_.forall(identity)), ready) match {
              case None    =>
                for (k <- 0 until valids.flatten.size) assert(!high(dut.io.in(k).ready), s"$clue: input $k ready, idle")
                assert(!high(dut.io.out.valid), s"$clue: out.valid while idle")
              case Some(s) =>
                var at = 0
                for (k      <- 0 until n) {
                  val e = dut.io.units(k); val m = s.units(k); val uc = s"$clue, ${units(k).name}"
                  for (_ <- 0 until units(k).inputs) {
                    assert(high(dut.io.in(at).ready) == m.fire, s"$uc: ready"); at += 1
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
                assert(high(dut.io.fire) == s.units(chain.head).fire, s"$clue: the head's fire")
                if (s.end) {
                  ended += 1
                  assert(model.classes.sum == int(dut.io.counter) + 1, s"$clue: COUNTER")
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
    assert(ended >= 60 * Seeds().size, s"$ended runs ended")
  }
}
