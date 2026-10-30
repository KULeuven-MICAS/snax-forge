package snax.forge.shell

import scala.collection.immutable.SeqMap

import chisel3._
import chisel3.util.MixedVec

import snax.forge.Datapaths

/** One catalogue datapath (`Datapaths`) as its own module, named after its unit or link: `<tag>_<unit>`, `<tag>_<link>`
  * (C§18's module tree). Its IO is the datapath's: the catalogue's live ports and pins.
  */
class Block(name: String, d: Datapaths.Datapath, p: Map[String, ujson.Value]) extends Module {
  override def desiredName: String = name

  private val built = chisel3.experimental.noPrefix(d.build(p)) // the datapath's instance stays `m`
  val io            = IO(chiselTypeOf(built))
  io <> built
}

/** `<tag>_datapath`: every unit's datapath (D147), a catalogue datapath built from the unit's ports by its fill rule,
  * in a module `<tag>_<unit>`, and every buffer's `LinkBuffer` (a link of depth >= 1), in a module `<tag>_<link>`. Data
  * and enables only. A link is plain wiring: a wire from its producer's output to its consumer's input; a buffer
  * between them, loaded from the producer and read by the consumer. Outside, per unit its outer ports' data, by BRM
  * port, and the pins its catalogue entry has live (`fire`, `first`, `read`, `lane`, `en`); per buffer its pins
  * (`load`, `wslot`, `rslot`), group by group.
  */
class AccDatapath(spec: AcceleratorSpec) extends Module {
  override def desiredName: String = s"${spec.tag}_datapath"

  private val blocks = spec.units.map { u =>
    val d = Datapaths(u.datapath)
    val p = d.fill(u)
    (u, d, p, Module(new Block(s"${spec.tag}_${u.name}", d, p)))
  }

  private val bufs = for {
    g <- spec.groups
    l <- g.links if l.depth > 0
  } yield {
    val d    = Datapaths("LinkBuffer")
    val from = spec.unit(l.from._1).port(l.from._2)
    val p    = d.fillLink(l.depth, from.lanes, from.dtype)
    (l, d, p, Module(new Block(s"${spec.tag}_${l.name}", d, p)))
  }

  /** One unit's side: its outer ports' data, by BRM port, and its live pins. */
  class UnitIo(u: AcceleratorSpec.UnitSpec, d: Datapaths.Datapath, p: Map[String, ujson.Value], b: Block)
      extends Record {
    private val dirs  = d.ports.map(x => x.name -> x.dir).toMap
    private val outer = spec.ins(u.name).map(_._2) ++ spec.outs(u.name).map(_._2)
    val elements      = SeqMap.from(
      u.ports.filter { case (brm, _) => outer.contains(brm) }.map { case (brm, x) =>
        val t = chiselTypeOf(b.io.elements(x.port))
        brm -> (if (dirs(x.port) == "in") Input(t) else Output(t))
      } ++ d.livePins(p).map(n => n.name -> Input(chiselTypeOf(b.io.elements(n.name))))
    )
    def apply(name: String): Data = elements(name)
  }

  /** One buffer's side: its pins. */
  class BufferIo(d: Datapaths.Datapath, p: Map[String, ujson.Value], b: Block) extends Record {
    val elements = SeqMap.from(d.livePins(p).map(n => n.name -> Input(chiselTypeOf(b.io.elements(n.name)))))
    def apply(name: String): Data = elements(name)
  }

  val io = IO(new Bundle {
    val units   = MixedVec(blocks.map { case (u, d, p, b) => new UnitIo(u, d, p, b) })
    val buffers = MixedVec(bufs.map { case (_, d, p, b) => new BufferIo(d, p, b) })
  })

  private def port(unit: String, brm: String): Data = {
    val (u, _, _, b) = blocks.find(_._1.name == unit).get
    b.io.elements(u.port(brm).port)
  }

  for (((u, d, p, b), k) <- blocks.zipWithIndex) {
    val dirs = d.ports.map(x => x.name -> x.dir).toMap
    for ((brm, x) <- u.ports if io.units(k).elements.contains(brm)) {
      if (dirs(x.port) == "in") b.io.elements(x.port) := io.units(k)(brm)
      else io.units(k)(brm)                           := b.io.elements(x.port)
    }
    d.livePins(p).foreach(n => b.io.elements(n.name) := io.units(k)(n.name))
  }

  /** Each buffer's index in `io.buffers`, by its link's name (a wire has none). */
  val buffered: Map[String, Int] = bufs.map(_._1.name).zip(bufs.indices).toMap

  // The links: a wire, or a buffer between the ends
  for {
    g <- spec.groups
    l <- g.links
  } {
    val from = port(l.from._1, l.from._2)
    val to   = port(l.to._1, l.to._2)
    buffered.get(l.name) match {
      case None    => to := from.asTypeOf(to)
      case Some(i) =>
        val (_, d, p, b) = bufs(i)
        b.io.elements("in") := from.asTypeOf(b.io.elements("in"))
        to                  := b.io.elements("out").asTypeOf(to)
        d.livePins(p).foreach(n => b.io.elements(n.name) := io.buffers(i)(n.name))
    }
  }
}
