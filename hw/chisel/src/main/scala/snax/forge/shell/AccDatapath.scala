package snax.forge.shell

import chisel3._

import snax.forge.Datapaths

/** `<tag>_datapath`: every unit's datapath (D147), a catalogue datapath (`Datapaths`) built from the unit's ports by
  * its fill rule. Data and enables only: per unit its input ports' bits, its output's bits, and the enables of its
  * controller its catalogue entry names as pins (`fire`, `first`).
  */
class AccDatapath(spec: AcceleratorSpec) extends Module {
  override def desiredName: String = s"${spec.tag}_datapath"

  private val built = spec.units.map { u =>
    val d    = Datapaths(u.datapath)
    val ins  = spec.ins(u.name).map(_._2)
    val outs = spec.outs(u.name).map(_._2)
    (u, d, d.build(d.fill(u, ins, outs)))
  }

  /** One unit's side: its BRM ports' data, by BRM port, and its pins. */
  class UnitIo(u: AcceleratorSpec.UnitSpec, b: Datapaths.Built, d: Datapaths.Datapath) extends Record {
    private val dirs = d.ports.map(x => x.name -> x.dir).toMap
    val elements     = scala.collection.immutable.SeqMap.from(
      u.ports.map { case (brm, p) =>
        val t = chiselTypeOf(b.io.elements(p.port))
        brm -> (if (dirs(p.port) == "in") Input(t) else Output(t))
      } ++ d.pins.map(n => n.name -> Input(Bool()))
    )
    def apply(name: String): Data = elements(name)
  }

  val io = IO(new Bundle {
    val units = chisel3.util.MixedVec(built.map { case (u, d, b) => new UnitIo(u, b, d) })
  })

  for (((u, d, b), k) <- built.zipWithIndex) {
    val dirs = d.ports.map(x => x.name -> x.dir).toMap
    for ((brm, p) <- u.ports) {
      if (dirs(p.port) == "in") b.io.elements(p.port) := io.units(k)(brm)
      else io.units(k)(brm)                           := b.io.elements(p.port)
    }
    d.pins.foreach(n => b.io.elements(n.name) := io.units(k)(n.name))
  }
}
