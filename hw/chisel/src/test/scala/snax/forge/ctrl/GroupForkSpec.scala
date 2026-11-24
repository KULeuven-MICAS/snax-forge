package snax.forge.ctrl

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, Seeds}

/** `GroupCtrl` against `GroupModel` on random groups of several chains on forked inputs: 2 or 3 chains of 1 or 2 units,
  * each head of span 1 with 1-2 outer inputs, its first forked to every other head's, a second fork of two heads'
  * second inputs now and then; a chain's second unit of span 1, 2 or 4 on the link from its head (depth 0, 1 or 2) with
  * 0-1 outer inputs of its own; every unit of L 0-2, II 1-2, T 1, 2 or 4 (fixed or a JOB_BEATS on its wire). Each
  * chain's JOBS makes every head fire alike (N, its chain's T product times its JOBS; LOWER refuses heads that fire
  * unlike, a run that would never end), now and then a JOBS or a T 0, at job_width 6 or 32, drawn again only while
  * idle; starts at random times, while busy too; random reader valids (a forked input's on each of its units'
  * handshakes), output readies and, in the seeded runs, seeds. Every cycle: the start's answer, busy and the done
  * pulse; every input's ready (a forked one's equal on each of its units); each output's valid; per unit fire, first,
  * read, lane, adv, give; per buffer load, slots and occupancy; the class; at each run's end the head's firings.
  */
class GroupForkSpec extends AnyFlatSpec with ForgeTester {

  private val Groups = 24
  private val Cycles = 300
  private var ended  = 0

  behavior of "GroupCtrl against SnaxGroupRule's model on forked inputs"

  for {
    seeded <- Seq(false, true)
    seed   <- Seeds()
  }
    it should s"give the model's cycles on random${if (seeded) " seeded" else ""} forked groups (seed $seed)" in Seeds
      .run(seed) { r =>
        for (g <- 0 until Groups) {
          // The chains, each its units' indices from the head; units numbered chain after chain
          val nChains = r.between(2, 3)
          val lengths = Seq.fill(nChains)(r.between(1, 2))
          val chains  = lengths.scanLeft(0)(_ + _).zip(lengths).map { case (at, n) => at until at + n }
          val heads   = chains.map(_.head)
          val units   = chains.flatMap { c =>
            c.map { k =>
              val head  = k == c.head
              val ins   = if (head) r.between(1, 2) else r.between(0, 1)
              val fold  = if (r.chance(0.5)) Some(r.pick(Seq(1, 2, 4))) else None
              val u     = UnitDesc(
                s"u$k",
                r.between(0, 2),
                r.between(1, 2),
                if (head) 1 else r.pick(Seq(1, 2, 4)),
                ins,
                fold,
                if (head) Nil else Seq(ins)
              )
              // seeded: an outer input to spare, the last one the seed (never a forked one, the first of a head)
              val spare = if (head) u.inputs >= 2 else u.inputs >= 1
              if (seeded && spare && r.chance(0.5)) u.copy(seeded = true) else u
            }
          }
          val links   = chains.filter(_.size == 2).map(c => Link(c(0), c(1), units(c(1)).inputs, r.between(0, 2)))
          // A head's forkable inputs: its outer ones but its seed
          def plain(k: Int) = units(k).inputs - (if (units(k).seeded) 1 else 0)
          val second    = heads.filter(plain(_) >= 2)
          val forks     = Seq(heads) ++ (if (second.size >= 2 && r.chance(0.5)) Seq(second.take(2)) else Nil)
          val forkInput = Seq(heads.map(_ -> 0)) ++ forks.drop(1).map(_.map(_ -> 1))
          val desc      = GroupDesc("fork", units, links, forks)
          val model     = new GroupModel(units, links, forks)
          // Per unit and outer input, its reader: a forked input at its first unit's
          val reader    = {
            var next = 0
            val at   = scala.collection.mutable.Map.empty[(Int, Int), Int]
            units.indices.map { k =>
              (0 until units(k).inputs).map { j =>
                val first = forkInput.find(_.contains(k -> j)).map(_.head).filter(_ != (k -> j))
                first.map(at).getOrElse { at(k -> j) = next; next += 1; next - 1 }
              }
            }
          }
          val readers   = reader.flatten.distinct.size
          val clue0     = s"seed $seed, group $g: ${units.mkString(", ")}, links ${links.mkString(",")}, forks $forks"
          val pv        = r.probability(); val pr = r.probability()
          val jw        = r.pick(Seq(6, 32))
          test(new GroupCtrl(desc, "t", jw)) { dut =>
            val open  = desc.open
            var jobs  = Seq.fill(nChains)(1)
            var opent = open.map(_ => 1)
            for (c <- 0 until Cycles) {
              val clue  = s"$clue0, job_width $jw, cycle $c"
              val offer = r.chance(0.1)
              // Each chain's JOBS and each JOB_BEATS, the csr's wires: drawn again only while idle; the heads fire
              // alike, n each, unless a value is 0 (the empty run)
              if (model.idle) {
                opent = open.map(_ => r.pick(Seq(1, 2, 4)))
                var next = 0
                val ts   = units.map(_.fixedT.getOrElse { next += 1; opent(next - 1) })
                val prod = chains.map(_.map(ts(_)).product)
                val n    = prod.max * r.pick(Seq(1, 2))
                jobs                     = desc.outputs.indices.map(o => if (n % prod(o) == 0) n / prod(o) else 0)
                if (r.chance(0.05)) jobs = jobs.updated(r.between(0, nChains - 1), 0)
                if (open.nonEmpty && r.chance(0.05)) opent = opent.updated(r.between(0, open.size - 1), 0)
              }
              dut.io.start.valid.poke(offer.B)
              dut.io.settings.jobs.poke(jobs.head.U)
              for {
                more <- dut.io.settings.more
                (p, j) <- more.zip(jobs.tail)
              } p.poke(j.U)
              dut.io.settings.beats.zip(opent).foreach { case (p, t) => p.poke(t.U) }
              assert(high(dut.io.start.ready) == model.idle, s"$clue: start.ready")
              val answer = if (offer) {
                var next = 0
                val ts   = units.map(_.fixedT.map(_.toLong).getOrElse { next += 1; opent(next - 1).toLong })
                model.start(jobs.map(_.toLong), ts)
              } else "none"
              val valid = Seq.fill(readers)(!r.chance(pv))
              val valids = reader.map(_.map(valid))
              var at     = 0
              for (v <- valids.flatten) { dut.io.in(at).valid.poke(v.B); at += 1 }
              val rooms  = desc.outputs.map(_ => !r.chance(pr))
              rooms.zipWithIndex.foreach { case (x, o) => dut.io.output(o).ready.poke(x.B) }
              assert(high(dut.io.busy) == model.busy, s"$clue: busy")
              val heads0 = model.headFirings
              val step   = model.step(
                valids.zip(units).map { case (v, u) => (if (u.seeded) v.init else v).forall(identity) },
                rooms,
                valids.zip(units).map { case (v, u) => !u.seeded || v.last }
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
                    for (j <- 0 until units(k).inputs) {
                      val want = if (units(k).seeded && j == units(k).inputs - 1) m.seed else m.fire
                      assert(high(dut.io.in(in).ready) == want, s"$uc: input $j's ready"); in += 1
                    }
                    assert(high(e.fire) == m.fire, s"$uc: fire")
                    assert(high(e.give) == m.push, s"$uc: give")
                    assert(high(e.read) == m.read, s"$uc: read")
                    assert(high(e.adv) == !m.frozen, s"$uc: adv")
                    if (m.fire || m.read) assert(high(e.first) == m.first, s"$uc: first")
                    if (m.read) assert(int(e.lane) == m.lane, s"$uc: lane")
                  }
                  for (f      <- forks) assert(f.map(s.units(_).fire).distinct.size == 1, s"$clue: fork $f fires apart")
                  for ((k, o) <- desc.outputs.zipWithIndex)
                    assert(
                      high(dut.io.output(o).valid) == (s.units(k).push || s.units(k).frozen),
                      s"$clue: output $o's valid"
                    )
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
                    assert(heads0 + (if (s.units(heads.head).fire) 1 else 0) == model.headFirings, s"$clue: firings")
                  }
              }
              dut.clock.step()
            }
          }
        }
      }

  it should "have ended enough random runs" in {
    info(s"$ended runs ended")
    assert(ended >= 2 * 40 * Seeds().size, s"$ended runs ended")
  }
}
