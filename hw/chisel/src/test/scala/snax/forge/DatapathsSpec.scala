package snax.forge

import java.nio.file.Files

import chisel3._
import _root_.circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.Datapaths.{DtypeParam, DtypesParam, IntParam, OpParam}

/** The datapath catalogue: `Emit --describe` (the registry, `Datapaths`) equals the checked-in `hw/chisel/blocks.json`,
  * the file the Python side checks `accelerator.json` against (D130). Compared as JSON, key and list order included, so
  * the file may be laid out by hand. Each datapath's builder also gives an IO with exactly the catalogue's ports and
  * pins, and no valid, ready or counter (D147).
  */
class DatapathsSpec extends AnyFlatSpec {

  behavior of "the datapath catalogue"

  it should "equal blocks.json" in {
    val file = RepoPaths.root.resolve("hw/chisel/blocks.json")
    val want = ujson.read(Files.readString(file))
    assert(ujson.write(ujson.read(Emit.describe)) == ujson.write(want))
  }

  /** The least params each type takes: an int's min, the first op, the first type (two of it for a list). */
  private def least(d: Datapaths.Datapath): Map[String, ujson.Value] =
    d.params.map {
      case (n, IntParam(min), _) => n -> ujson.Num(min)
      case (n, OpParam(vs), _)   => n -> ujson.Str(vs.head)
      case (n, DtypeParam, _)    => n -> ujson.Str(IntType.names.head)
      case (n, DtypesParam, _)   => n -> ujson.Arr(IntType.names.head, IntType.names.head)
    }.toMap

  it should "name every port and pin of each datapath's IO, and nothing else" in {
    for (d <- Datapaths.all) {
      var names = Set.empty[String]
      ChiselStage.emitCHIRRTL(new Module {
        val built = d.build(least(d))
        names = built.io.elements.keySet.toSet
        built.io := DontCare
      })
      assert((d.ports.map(_._1) ++ d.pins).toSet == names, s"${d.name}: catalogue, IO ${names.toSeq.sorted}")
    }
  }
}
