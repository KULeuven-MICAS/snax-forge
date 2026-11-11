package snax.forge.shell

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.{ForgeTester, LaneBits, Rand, Seeds}
import snax.forge.shell.AcceleratorSpec.{CsrSpec, MoverSpec, MuxUser, PortMuxSpec, SetSpec}

/** `PortMux`, a `RawModule`, inside a `Module` for chiseltest: its ports as they are. */
class PortMuxHarness(tag: String, set: SetSpec) extends Module {
  private val mux = Module(new PortMux(tag, set))
  val io          = IO(chiselTypeOf(mux.io))
  io <> mux.io
}

/** `<tag>_<set>_port_mux` (`PortMux`) against a model of the data-side mux as D169 and C§17 state it (`Model`), never
  * read off the Chisel.
  *
  * Per seed, `Shapes` random shared sets: 2 or 3 groups (PORT_MUX_SEL 1 or 2 bits), 1 to 3 reader and 1 to 3 writer
  * movers of 1 to 4 channels, each group a user on a mover with probability 0.6 (so some movers serve one group or
  * none), a user of 1 to its mover's channels; and the sets of the shared fixtures, snax-gvsoc's `forge/dot_1_shared/`
  * and `forge/mvt_1_dot2_shared/`. Each driven `Cycles` cycles with every input random each cycle: the select over all
  * its values (2 or 3 groups, and 3 at 2 bits, which selects none), the movers' and users' data, valids and readies.
  * Every output checked each cycle; then every ready flipped and every valid checked unchanged: `valid` never depends
  * on `ready`.
  */
class PortMuxSpec extends AnyFlatSpec with ForgeTester {

  private val Shapes = 20
  private val Cycles = 40

  /** What the mux gives, from its inputs: `sel`, per mover its data, valid (reader) or ready (writer), per user its
    * ready (reader) or data and valid (writer).
    */
  private class Model(set: SetSpec) {
    private val groups = set.portMux.get.groups
    private def on(sel: BigInt, u: MuxUser): Boolean = sel == groups.indexOf(u.group)

    /** Every output by its port name. */
    def outputs(sel: BigInt, in: Map[String, BigInt]): Map[String, BigInt] =
      set.movers.flatMap { m =>
        def b(x: Boolean) = if (x) BigInt(1) else BigInt(0)
        if (m.reader) {
          val users = m.users.flatMap { u =>
            val n = PortMux.user(u)
            Seq(
              s"${n}_data_o"  -> (in(s"${m.name}_data_i") & LaneBits.mask(u.numChannel * 64)),
              s"${n}_valid_o" -> b(in(s"${m.name}_valid_i") == 1 && on(sel, u))
            )
          }
          val ready = m.users.exists(u => on(sel, u) && in(s"${PortMux.user(u)}_ready_i") == 1)
          users :+ (s"${m.name}_ready_o" -> b(ready))
        } else {
          val picked = m.users.find(on(sel, _)).map(PortMux.user)
          Seq(
            s"${m.name}_data_o"  -> picked.map(n => in(s"${n}_data_i")).getOrElse(BigInt(0)),
            s"${m.name}_valid_o" -> b(picked.exists(n => in(s"${n}_valid_i") == 1))
          ) ++ m.users.map(u => s"${PortMux.user(u)}_ready_o" -> b(in(s"${m.name}_ready_i") == 1 && on(sel, u)))
        }
      }.toMap
  }

  /** A random shared set of `groups` groups (the doc). */
  private def shape(r: Rand): SetSpec = {
    val groups                               = Seq.tabulate(r.between(2, 3))(g => s"g$g")
    def movers(role: String, prefix: String) =
      Seq.tabulate(r.between(1, 3)) { k =>
        val nc    = r.between(1, 4)
        val users = groups.filter(_ => r.chance(0.6)).map(g => MuxUser(g, s"u${g}_$prefix$k" -> "p", r.between(1, nc)))
        MoverSpec(s"$prefix$k", role, nc, users.exists(_.numChannel < nc), users)
      }
    val bits                                 = if (groups.size <= 2) 1 else 2
    SetSpec(
      "s",
      Seq.empty,
      Seq.empty,
      groups,
      movers("read", "reader") ++ movers("write", "writer"),
      Some(PortMuxSpec(groups, CsrSpec("PORT_MUX_SEL", "rw", 0, bits)))
    )
  }

  /** `set`'s mux for `Cycles` random cycles against the model. */
  private def drive(set: SetSpec, r: Rand, clue: String): Unit = {
    val model = new Model(set)
    val width = set.portMux.get.csr.width
    test(new PortMuxHarness("t", set)) { dut =>
      val io      = dut.io
      val inputs  = io.elements.toSeq.filter { case (_, d) =>
        chisel3.reflect.DataMirror.directionOf(d) == ActualDirection.Input
      }
      val outputs = io.elements.keys.filterNot(inputs.map(_._1).toSet).toSeq
      for (c <- 0 until Cycles) {
        val sel  = BigInt(r.between(0, (1 << width) - 1))
        val in   = inputs.collect {
          case (n, d) if n != "sel_i" =>
            n -> (if (d.getWidth == 1) BigInt(if (r.chance(0.5)) 1 else 0) else r.word(d.getWidth))
        }.toMap
        io.uint("sel_i").poke(sel.U)
        for ((n, v) <- in) io.uint(n).poke(v.U)
        val want = model.outputs(sel, in)
        assert(want.keySet == outputs.toSet, s"$clue: the model's outputs ${want.keySet}, the mux's $outputs")
        for ((n, v) <- want) assert(io.uint(n).peek().litValue == v, s"$clue, cycle $c, sel $sel: $n, the model $v")
        // valid never depends on ready: flip every ready, the valids stay
        val valids = outputs.filter(_.endsWith("_valid_o"))
        for ((n, v) <- in if n.endsWith("_ready_i")) io.uint(n).poke((1 - v).U)
        for (n <- valids) assert(io.uint(n).peek().litValue == want(n), s"$clue, cycle $c: $n changed with the readies")
        dut.clock.step()
      }
    }
  }

  behavior of "<tag>_<set>_port_mux"

  for (seed <- Seeds())
    it should s"give every output as the mux's model on random shared sets (seed $seed)" in Seeds.run(seed) { r =>
      for (s <- 0 until Shapes) {
        val set = shape(r)
        drive(
          set,
          r,
          s"shape $s (${set.movers
              .map(m => s"${m.name}:${m.numChannel}:${m.users.map(u => s"${u.group}/${u.numChannel}").mkString(",")}")
              .mkString(" ")})"
        )
      }
    }

  private lazy val shared = RtlFixtures.all.filter(_.name.endsWith("_shared"))

  it should "have the shared fixtures dot_1_shared and mvt_1_dot2_shared" in
    assert(shared.map(_.name) == Seq("dot_1_shared", "mvt_1_dot2_shared"))

  for (name <- Seq("dot_1_shared", "mvt_1_dot2_shared"))
    it should s"give every output as the mux's model on $name's set" in Seeds.run(Seeds().head) { r =>
      val f    = RtlFixtures.all.find(_.name == name).get
      val sets = f.spec.sets.filter(_.shared)
      assert(sets.size == 1, s"$name: one shared set")
      drive(sets.head, r, name)
    }
}
