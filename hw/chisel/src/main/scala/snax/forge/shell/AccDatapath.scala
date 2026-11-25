package snax.forge.shell

import scala.collection.immutable.SeqMap

import chisel3._
import chisel3.util.MixedVec

import snax.forge.Datapaths

/** One catalogue datapath (`Datapaths`) as its own module, named after its unit or link: `<tag>_<unit>`, `<tag>_<link>`
  * (the module tree of CONTRACTS, "Bundle"). Its IO is the datapath's: the catalogue's live ports and pins.
  */
class Block(name: String, d: Datapaths.Datapath, p: Map[String, ujson.Value]) extends Module {
  override def desiredName: String = name

  private val built = chisel3.experimental.noPrefix(d.build(p)) // the datapath's instance stays `m`
  val io            = IO(chiselTypeOf(built))
  io <> built
}

/** `<tag>_datapath`: every unit's datapath, a catalogue datapath built from the unit's ports by its fill rule, in a
  * module `<tag>_<unit>`, and every buffer's `LinkBuffer` (a link of depth >= 1, then a fork link), in a module
  * `<tag>_<link>`, group by group (its units, then its buffers). Data and enables only. A link is plain wiring: a wire
  * from its producer's output to its consumer's input; a buffer between them, loaded from the producer and read by the
  * consumer. A held link's buffer is read by the wrapper instead (`held`, its one-lane beat, kept a row in the
  * wrapper's `<unit>_<port>_hold` and broadcast), which gives its consumer's port as an outer one; a fork link's buffer
  * is loaded with the fork's reader's beat (its first unit's port) and read by its consumer. Outside, per unit its
  * outer ports' data (and a held link's port), by BRM port, and the pins its catalogue entry has live (`fire`, `first`,
  * `read`, `lane`, `en`); per buffer its pins (`load`, `wslot`, `rslot`), group by group.
  */
class AccDatapath(spec: AcceleratorSpec) extends Module {
  override def desiredName: String = s"${spec.tag}_datapath"

  // Each block made group by group, its units then its buffers (the module tree of CONTRACTS, "Bundle"), the
  // instances named as when the units and the buffers were made in two passes (`blocks_<k>_4`, `bufs_<i>_4`), so the
  // SV of a design whose two orders agree stays as it was; `blocks` in unit order, `bufs` in link order
  private val made: Map[String, (Datapaths.Datapath, Map[String, ujson.Value], Block)] = {
    var (k, i) = (0, 0)
    spec.groups.flatMap { g =>
      g.units.map { u =>
        val d = Datapaths(u.datapath)
        val p = d.fill(u)
        val b = Module(new Block(s"${spec.tag}_${u.name}", d, p)).suggestName(s"blocks_${k}_4")
        k += 1
        u.name -> (d, p, b)
      } ++ g.links.filter(_.depth > 0).map { l =>
        val d    = Datapaths("LinkBuffer")
        val from = spec.unit(l.from._1).port(l.from._2)
        val p    = d.fillLink(l.depth, from.lanes, from.dtype)
        val b    = Module(new Block(s"${spec.tag}_${l.name}", d, p)).suggestName(s"bufs_${i}_4")
        i += 1
        l.name -> (d, p, b)
      } ++ g.forkLinks.map { f =>
        val d  = Datapaths("LinkBuffer")
        val to = spec.unit(f.to._1).port(f.to._2)
        val p  = d.fillLink(f.depth, to.streamLanes, to.dtype)
        val b  = Module(new Block(s"${spec.tag}_${f.name}", d, p)).suggestName(s"bufs_${i}_4")
        i += 1
        f.name -> (d, p, b)
      }
    }.toMap
  }

  private val blocks = spec.units.map { u =>
    val (d, p, b) = made(u.name)
    (u, d, p, b)
  }

  private val bufs =
    spec.groups.flatMap(g => g.links.filter(_.depth > 0).map(_.name) ++ g.forkLinks.map(_.name)).map { name =>
      val (d, p, b) = made(name)
      (name, d, p, b)
    }

  /** The held links, each into a held port: its buffer's beat goes to the wrapper (`io.held`). */
  private val heldLinks = for {
    g <- spec.groups
    l <- g.links if spec.unit(l.to._1).port(l.to._2).hold
  } yield l

  /** One unit's side: its outer ports' data, by BRM port, and its live pins. */
  class UnitIo(u: AcceleratorSpec.UnitSpec, d: Datapaths.Datapath, p: Map[String, ujson.Value], b: Block)
      extends Record {
    private val dirs  = d.ports.map(x => x.name -> x.dir).toMap
    private val outer = spec.ins(u.name).map(_._2) ++ spec.outs(u.name).map(_._2) ++
      heldLinks.filter(_.to._1 == u.name).map(_.to._2)
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
    val held    = Option.when(heldLinks.nonEmpty)(MixedVec(heldLinks.map { l =>
      Output(chiselTypeOf(made(l.name)._3.io.elements("out")))
    }))
  })

  /** Each held link's place in `io.held`, by its link's name. */
  val heldOf: Map[String, Int] = heldLinks.map(_.name).zipWithIndex.toMap

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

  /** Each buffer's index in `io.buffers`, by its link's (or fork link's) name (a wire has none). */
  val buffered: Map[String, Int] = bufs.map(_._1).zip(bufs.indices).toMap

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
        heldOf.get(l.name) match { // NB (D197): a held link's beat to the wrapper's hold register
          case Some(h) => io.held.get(h) := b.io.elements("out")
          case None    => to             := b.io.elements("out").asTypeOf(to)
        }
        d.livePins(p).foreach(n => b.io.elements(n.name) := io.buffers(i)(n.name))
    }
  }

  // The fork links: the fork's reader's beat (its first port's, as the wrapper gives it) into the buffer, its
  // consumer's port from the buffer
  for {
    g <- spec.groups
    f <- g.forkLinks
  } {
    val reader       = spec.forks.find(_.contains(f.to)).get.head
    val i            = buffered(f.name)
    val (_, d, p, b) = bufs(i)
    val k            = spec.units.indexWhere(_.name == reader._1)
    val to           = port(f.to._1, f.to._2)
    b.io.elements("in") := io.units(k)(reader._2).asTypeOf(b.io.elements("in"))
    to                  := b.io.elements("out").asTypeOf(to)
    d.livePins(p).foreach(n => b.io.elements(n.name) := io.buffers(i)(n.name))
  }
}
