package snax.forge.datapath

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{Datapaths, ForgeTester, LaneBits, SnaxDatapathVectors}
import snax.forge.shell.Block

/** Every catalogue datapath on snax-gvsoc's NumPy vectors (`SnaxDatapathVectors`): each test's datapath built from the
  * registry (`Datapaths`) by its params, then per step its enables and inputs poked, its output peeked before the clock
  * edge and compared lane by lane, as snax-gvsoc's `tests/snax/datapath/tb.cpp` checks its classes on the same file.
  * Without `SNAX_GVSOC_DATAPATH_VECTORS` the spec cancels.
  */
class DatapathVectorsSpec extends AnyFlatSpec with ForgeTester {

  private def run(t: SnaxDatapathVectors.Test): Unit =
    test(new Block(s"vectors_${t.datapath}", Datapaths(t.datapath), t.params)) { dut =>
      val io = dut.io.elements
      def bool(n: String, v: Boolean) = io.get(n).foreach(_.asInstanceOf[Bool].poke(v.B))
      def slot(n: String, v: Int)     = io.get(n).foreach(x => x.asInstanceOf[UInt].poke(v.U))
      val inputs = t.datapath match {
        case "ElementwiseLanes" | "Mac" => Seq("a", "b", "c")
        case _                          => Seq("in")
      }
      for ((s, k) <- t.steps.zipWithIndex) {
        bool("fire", s.fire); bool("first", s.first); bool("read", s.read); bool("en", s.en)
        bool("load", s.load); slot("lane", s.lane); slot("wslot", s.wslot); slot("rslot", s.rslot)
        for {
          (lanes, i) <- s.ins.zipWithIndex
          d = io(inputs(i))
          (v, l) <- lanes.zipWithIndex
        }
          LaneBits.poke(LaneBits.lanes(d)(l), v)
        val out = LaneBits.lanes(io("out"))
        for ((want, l) <- s.out.zipWithIndex) {
          val w = out(l).getWidth
          assert(LaneBits.peek(out(l)) == (want & LaneBits.mask(w)), s"${t.name}: step $k, lane $l")
        }
        dut.clock.step()
      }
    }

  behavior of "the datapaths on snax-gvsoc's vectors"

  SnaxDatapathVectors.all match {
    case None      =>
      it should "give every vector's output" in
        cancel(s"${SnaxDatapathVectors.Env} is not set (snax-gvsoc's tests/snax/datapath)")
    case Some(all) =>
      it should "find the vectors of every catalogue datapath" in
        assert(all.map(_.datapath).toSet == Datapaths.all.map(_.name).toSet)
      for (t <- all) it should s"give ${t.name}'s outputs (${t.datapath})" in run(t)
  }
}
