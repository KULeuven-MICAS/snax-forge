package snax.forge.shell

import scala.collection.immutable.SeqMap

import chisel3._

import snax.forge.shell.AcceleratorSpec.{MoverSpec, MuxUser, SetSpec}

/** `PortMux`'s ports: `sel_i`, PORT_MUX_SEL's value; per mover in the set's order, a reader's
  * `<mover>_{data_i,valid_i,ready_o}` and a writer's `<mover>_{data_o,valid_o,ready_i}`, `num_channel * 64` bits; per
  * user, its group's side, a reader's `<unit>_<port>_{data_o,valid_o,ready_i}` and a writer's
  * `<unit>_<port>_{data_i,valid_i,ready_o}`, the user's `num_channel * 64` bits.
  */
class PortMuxPorts(val set: SetSpec, val selWidth: Int) extends Record {
  private def mover(m: MoverSpec): Seq[(String, Data)] =
    if (m.reader)
      Seq(
        s"${m.name}_data_i"  -> Input(UInt(m.width.W)),
        s"${m.name}_valid_i" -> Input(Bool()),
        s"${m.name}_ready_o" -> Output(Bool())
      )
    else
      Seq(
        s"${m.name}_data_o"  -> Output(UInt(m.width.W)),
        s"${m.name}_valid_o" -> Output(Bool()),
        s"${m.name}_ready_i" -> Input(Bool())
      )

  private def user(m: MoverSpec, u: MuxUser): Seq[(String, Data)] = {
    val n = PortMux.user(u)
    if (m.reader)
      Seq(
        s"${n}_data_o"  -> Output(UInt((u.numChannel * 64).W)),
        s"${n}_valid_o" -> Output(Bool()),
        s"${n}_ready_i" -> Input(Bool())
      )
    else
      Seq(
        s"${n}_data_i"  -> Input(UInt((u.numChannel * 64).W)),
        s"${n}_valid_i" -> Input(Bool()),
        s"${n}_ready_o" -> Output(Bool())
      )
  }

  val elements: SeqMap[String, Data] =
    SeqMap[String, Data]("sel_i" -> Input(UInt(selWidth.W))) ++ set.movers.flatMap(mover) ++
      set.movers.flatMap(m => m.users.flatMap(user(m, _)))

  def apply(name: String): Data = elements(name)
  def bool(name:  String): Bool = elements(name).asInstanceOf[Bool]
  def uint(name:  String): UInt = elements(name).asInstanceOf[UInt]
}

/** `<tag>_<set>_port_mux`: the data-side mux of a streamer set whose groups share its movers (D169, C§17), between the
  * movers' ports at the wrapper and the groups' ports; pure wires, no register. snax-gvsoc's `SnaxPortMuxRule`
  * (`snax/snax_port_mux.hpp`), combinational and with no fatal case: what the model refuses (a beat of a group not
  * selected) the hardware never shows its group.
  *
  * `sel_i`, PORT_MUX_SEL, is the selected group's index in `port_mux.groups`; a user is selected when its group is. Per
  * reader mover: each user's data is the mover's low `num_channel * 64` bits, whatever the select; its valid is the
  * mover's valid when selected, else 0; the mover's ready is its selected user's ready, 0 when the selected group has
  * no user on it. Per writer mover: its data is the selected user's zero-padded to the mover's width, its valid that
  * user's valid, both 0 with no selected user; each user's ready is the mover's ready when selected, else 0. A valid
  * depends on valids and the select only, never on a ready.
  */
class PortMux(tag: String, set: SetSpec) extends RawModule {
  private val mux = set.portMux.getOrElse(throw new IllegalArgumentException(s"set ${set.name} shares no movers"))
  override def desiredName: String = s"${tag}_${set.name}_port_mux"

  val io = FlatIO(new PortMuxPorts(set, mux.csr.width))

  private val sel = io.uint("sel_i")

  for (m <- set.movers) {
    val picked = m.users.map { u =>
      val g = mux.groups.indexOf(u.group)
      require(g >= 0, s"${m.name}: user ${u.port} of ${u.group}, a group not in port_mux.groups ${mux.groups}")
      require(1 <= u.numChannel && u.numChannel <= m.numChannel, s"${m.name}: user of ${u.numChannel} channels")
      (sel === g.U, PortMux.user(u), u)
    }
    if (m.reader) {
      for ((on, n, u) <- picked) {
        io(s"${n}_data_o")  := io.uint(s"${m.name}_data_i")(u.numChannel * 64 - 1, 0)
        io(s"${n}_valid_o") := io.bool(s"${m.name}_valid_i") && on
      }
      io(s"${m.name}_ready_o") := picked.map { case (on, n, _) => on && io.bool(s"${n}_ready_i") }
        .foldLeft(false.B)(_ || _)
    } else {
      io(s"${m.name}_data_o")  := picked.map { case (on, n, _) =>
        Mux(on, io.uint(s"${n}_data_i").pad(m.width), 0.U(m.width.W))
      }.foldLeft(0.U(m.width.W))(_ | _)
      io(s"${m.name}_valid_o") := picked.map { case (on, n, _) => on && io.bool(s"${n}_valid_i") }
        .foldLeft(false.B)(_ || _)
      for ((on, n, _) <- picked) io(s"${n}_ready_o") := io.bool(s"${m.name}_ready_i") && on
    }
  }
}

object PortMux {

  /** A user's port name on the mux: `<unit>_<port>`. */
  def user(u: MuxUser): String = s"${u.port._1}_${u.port._2}"
}
