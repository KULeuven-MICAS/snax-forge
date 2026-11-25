package snax.forge.ctrl

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Seeds}

/** `GroupCtrl` against `GroupModel` on random groups of atax's shape: a head (L 0-2, 1-2 outer inputs, T 1 or 2) into a
  * middle unit (L 0-2, T 1-3) over a wire or a buffer of 1-2, into a bank of R = 2 or 4 rows (L 1-2, as a bank's
  * accumulating Mac; its ROWS 1 to R from its wire, T = M · ROWS for M 1-3) over a held link of depth 1-2, the bank's
  * own outer input (A read again) or, half the time, the head's first input forked to it through a fork link of depth
  * T_h · T_m to + 3 (T_h · T_m = ROWS, so the head and the bank fire alike; below that the run never ends, which LOWER
  * refuses); the bank seeded now and then. Each T fixed or a JOB_BEATS on its wire, JOBS 1-2, now and then a JOBS or a
  * T 0, drawn again only while idle, at job_width 6 or 32; starts at random times, while busy too; random reader
  * valids, output readies and seeds. Every cycle: the start's answer, busy and the done pulse; every input's ready; per
  * unit fire, first, read, adv, give, the bank's take; per buffer and fork link load, slots and occupancy; the class;
  * at each run's end the head's firings.
  */
class GroupHoldForkSpec extends AnyFlatSpec with ForgeTester {

  private val Groups = 24
  private val Cycles = 300
  private var ended  = 0

  behavior of "GroupCtrl against SnaxGroupRule's model on a held link and a fork link"

  for (seed <- Seeds())
    it should s"give the model's cycles on random held and forked groups (seed $seed)" in Seeds.run(seed) { r =>
      for (g <- 0 until Groups) {
        val forked  = r.chance(0.5)
        val rows    = r.pick(Seq(2, 4))
        val seeded  = r.chance(0.3)
        val bankIns = (if (forked) 0 else 1) + (if (seeded) 1 else 0)
        val head    = UnitDesc("h", r.between(0, 2), 1, 1, r.between(1, 2), if (r.chance(0.5)) Some(1) else None)
        val mid     = UnitDesc("m", r.between(0, 2), 1, 1, 0, if (r.chance(0.5)) Some(1) else None, Seq(0))
        val fed     = bankIns until bankIns + (if (forked) 2 else 1)
        val bank    =
          UnitDesc("b", r.between(1, 2), 1, 1, bankIns, None, fed, seeded = seeded, rows = rows)
        val units   = Seq(head, mid, bank)
        val links   = Seq(Link(0, 1, 0, r.between(0, 2)), Link(1, 2, fed.last, r.between(1, 2), hold = true))
        val forks   = if (forked) Seq(Seq(0)) else Nil
        val depthOf = r.between(0, 3) // the fork link's depth: T_h · T_m and this more
        val jw      = r.pick(Seq(6, 32))
        val pv      = r.probability(); val pr = r.probability()
        // Each run's Ts (head, middle, bank) and ROWS: with a fork T_h · T_m = ROWS, its first draw's for the group (the
        // fork link's depth is set from it); the bank's T a multiple of ROWS
        var rowsOnce: Option[Int]     = None
        def draw():   (Seq[Int], Int) = {
          val rw = rowsOnce.getOrElse(r.pick((1 to rows).filter(rows % _ == 0)))
          if (forked) rowsOnce = Some(rw)
          val (th, tm) =
            if (forked) { val h = if (rw % 2 == 0 && r.chance(0.5)) 2 else 1; (h, rw / h) }
            else (r.between(1, 2), r.between(1, 3))
          (Seq(th, tm, r.between(1, 3) * rw), rw)
        }
        // Fixed Ts are the group's first draw; a fork link's depth is set from it
        val (first, firstRows) = draw()
        val fixed      = units.indices.map(k => r.chance(0.5) && (k != 2 || first(2) > 1)) // a bank's T > 1
        val fixedUnits = units.zipWithIndex.map { case (u, k) =>
          u.copy(fixedT = if (fixed(k)) Some(first(k)) else None)
        }
        val depth      = first(0) * first(1) + depthOf
        val forkLinks  = if (forked) Seq(ForkLink(0, 2, fed.head, depth)) else Nil
        val desc       = GroupDesc("atax", fixedUnits, links, forks, forkLinks)
        val model      = new GroupModel(fixedUnits, links, forks, forkLinks)
        val clue0      = s"seed $seed, group $g: ${fixedUnits.mkString(", ")}, links ${links.mkString(",")}, " +
          s"forks $forks $forkLinks"
        test(new GroupCtrl(desc, "t", jw)) { dut =>
          val open  = desc.open
          var jobs  = 1
          var ts    = first
          var rw    = firstRows
          var opent = open.map(u => ts(fixedUnits.indexOf(u)))
          for (c <- 0 until Cycles) {
            val clue   = s"$clue0, job_width $jw, cycle $c"
            val offer  = r.chance(0.1)
            // JOBS, each JOB_BEATS and ROWS, the csr's wires: drawn again only while idle, a fixed T kept (its ROWS
            // then one that divides it, and with a fork the open Ts matched to it)
            if (model.idle) {
              val (t, x) = draw()
              val fits   = units.indices.filter(fixed).forall(k => fixedUnits(k).fixedT.contains(t(k)))
              if (fits) { ts = t; rw = x }
              jobs = r.between(1, 2)
              opent                    = open.map(u => ts(fixedUnits.indexOf(u)))
              if (r.chance(0.05)) jobs = 0
              if (open.nonEmpty && r.chance(0.05)) opent = opent.updated(r.between(0, open.size - 1), 0)
            }
            dut.io.start.valid.poke(offer.B)
            dut.io.settings.jobs.poke(jobs.U)
            dut.io.settings.beats.zip(opent).foreach { case (p, t) => p.poke(t.U) }
            dut.io.settings.rows.get(0).poke(rw.U)
            assert(high(dut.io.start.ready) == model.idle, s"$clue: start.ready")
            val answer = if (offer) {
              var next = 0
              val t    = fixedUnits.map(_.fixedT.map(_.toLong).getOrElse { next += 1; opent(next - 1).toLong })
              model.start(Seq(jobs.toLong), t, Seq(1L, 1L, rw.toLong))
            } else "none"
            val valids = fixedUnits.map(u => Seq.fill(u.inputs)(!r.chance(pv)))
            var at     = 0
            for (v <- valids.flatten) { dut.io.in(at).valid.poke(v.B); at += 1 }
            val room   = !r.chance(pr)
            dut.io.out.ready.poke(room.B)
            assert(high(dut.io.busy) == model.busy, s"$clue: busy")
            val heads0 = model.headFirings
            val step   = model.step(
              valids.zip(fixedUnits).map { case (v, u) => (if (u.seeded) v.init else v).forall(identity) },
              room,
              valids.zip(fixedUnits).map { case (v, u) => !u.seeded || v.last }
            )
            val pulse  = answer == "ignored" || step.exists(_.end)
            assert(high(dut.io.pulse) == pulse, s"$clue: the done pulse, the model $pulse")
            step match {
              case None    =>
                for (k <- 0 until at) assert(!high(dut.io.in(k).ready), s"$clue: input $k ready, idle")
              case Some(s) =>
                var in = 0
                for (k      <- units.indices) {
                  val e = dut.io.units(k); val m = s.units(k); val uc = s"$clue, ${units(k).name}"
                  for (j <- 0 until fixedUnits(k).inputs) {
                    val want = if (fixedUnits(k).seeded && j == fixedUnits(k).inputs - 1) m.seed else m.fire
                    assert(high(dut.io.in(in).ready) == want, s"$uc: input $j's ready"); in += 1
                  }
                  assert(high(e.fire) == m.fire, s"$uc: fire")
                  assert(high(e.give) == m.push, s"$uc: give")
                  assert(high(e.read) == m.read, s"$uc: read")
                  assert(high(e.adv) == !m.frozen, s"$uc: adv")
                  if (m.fire || m.read) assert(high(e.first) == m.first, s"$uc: first")
                  for (t <- e.take) assert(high(t) == m.take, s"$uc: take")
                }
                assert(high(dut.io.out.valid) == (s.units(2).push || s.units(2).frozen), s"$clue: the output's valid")
                for ((k, i) <- desc.buffers.zipWithIndex) {
                  val d = links(k).depth; val e = dut.io.links(i); val m = s.links(k)
                  assert(high(e.load) == m.load, s"$clue, link $k: load")
                  if (m.load && d > 1) assert(int(e.wslot) == m.wslot, s"$clue, link $k: wslot")
                  if (m.take && d > 1) assert(int(e.rslot) == m.rslot, s"$clue, link $k: rslot")
                  assert(int(e.occupancy) == m.occupancy, s"$clue, link $k: occupancy")
                }
                for {
                  ports     <- dut.io.forkLinks
                  (m, i)    <- s.forks.zipWithIndex
                } {
                  val e = ports(i)
                  assert(high(e.load) == m.load, s"$clue, fork link: load")
                  if (m.load && depth > 1) assert(int(e.wslot) == m.wslot, s"$clue, fork link: wslot")
                  if (m.take && depth > 1) assert(int(e.rslot) == m.rslot, s"$clue, fork link: rslot")
                  assert(int(e.occupancy) == m.occupancy, s"$clue, fork link: occupancy")
                }
                assert(int(dut.io.cls) == s.cls, s"$clue: class")
                if (s.end) {
                  ended += 1
                  assert(heads0 + (if (s.units(0).fire) 1 else 0) == model.headFirings, s"$clue: firings")
                }
            }
            dut.clock.step()
          }
        }
      }
    }

  it should "have ended enough random runs" in {
    info(s"$ended runs ended")
    assert(ended >= 40 * Seeds().size, s"$ended runs ended")
  }
}
