package snax.forge.datapath

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, IntType, LaneBits, Seeds}

/** `LinkBuffer` (D149) against a Scala model written from C§17: on load slot wslot takes the beat, `out` is slot rslot,
  * so a beat written in one cycle reads from the next. Random beats and loads driven as a ring of the occupancy the
  * control unit keeps, at depth 1 (slots 0 bits wide), 2 and 4, int64 and int32, signed and unsigned.
  */
class LinkBufferSpec extends AnyFlatSpec with ForgeTester {

  behavior of "LinkBuffer"

  it should "have 0-bit slots at depth 1" in {
    test(new LinkBuffer(1, 4, "int64")) { dut =>
      assert(dut.io.wslot.getWidth == 0 && dut.io.rslot.getWidth == 0)
    }
  }

  for (seed <- Seeds())
    it should s"give the model's beats as a ring (seed $seed)" in Seeds.run(seed) { r =>
      for (depth <- Seq(1, 2, 4)) {
        val t     = IntType(r.pick(Seq("int64", "uint64", "int32", "uint32")))
        val lanes = r.pick(Seq(2, 4))
        test(new LinkBuffer(depth, lanes, t.name)) { dut =>
          val slots      = Array.fill(depth)(Seq.fill(lanes)(BigInt(0)))
          var w, rd, occ = 0
          for (c <- 0 until 40) {
            val take = occ > 0               && r.chance(0.6)
            val load = (occ < depth || take) && r.chance(0.7)
            val beat = Seq.fill(lanes)(r.word(t.width))
            beat.zipWithIndex.foreach { case (v, l) => LaneBits.poke(dut.io.in(l), v) }
            dut.io.load.poke(load.B)
            if (depth > 1) { dut.io.wslot.poke(w.U); dut.io.rslot.poke(rd.U) }
            for (l <- 0 until lanes) assert(LaneBits.peek(dut.io.out(l)) == slots(rd)(l), s"depth $depth, cycle $c")
            if (load) { slots(w) = beat; w = (w + 1) % depth }
            if (take) rd = (rd + 1) % depth
            occ += (if (load) 1 else 0) - (if (take) 1 else 0)
            dut.clock.step()
          }
        }
      }
    }
}
