package snax.forge

import java.nio.file.Files

import chisel3._
import _root_.circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.Blocks.{BoolParam, DtypeParam, DtypesParam, IntParam, OpsParam, Params}

/** The block catalogue: `Emit --describe` (the registry, `Blocks`) equals the checked-in `hw/chisel/blocks.json`, the
  * file the Python side checks `rtl.json` against (D130). Compared as JSON, key and list order included, so the file
  * may be laid out by hand. Each block's builder also gives an IO with exactly the catalogue's pins and ports.
  */
class BlocksSpec extends AnyFlatSpec {

  behavior of "the block catalogue"

  it should "equal blocks.json" in {
    val file = RepoPaths.root.resolve("hw/chisel/blocks.json")
    val want = ujson.read(Files.readString(file))
    assert(ujson.write(ujson.read(Emit.describe)) == ujson.write(want))
  }

  /** The least params each type takes: an int's min, false, the first op, the first type (two of it for a list: the
    * elementwise block's a and b).
    */
  private def least(b: Blocks.Block): Params =
    Params(b.params.map {
      case (n, IntParam(min, _)) => n -> ujson.Num(min)
      case (n, BoolParam)        => n -> ujson.Bool(false)
      case (n, OpsParam(vals))   => n -> ujson.Arr(ujson.Str(vals.head._1))
      case (n, DtypeParam)       => n -> ujson.Str(IntType.names.head)
      case (n, DtypesParam)      => n -> ujson.Arr(IntType.names.head, IntType.names.head)
    }.toMap)

  it should "name every pin and port of each block's IO" in {
    for (b <- Blocks.all) {
      var names = Set.empty[String]
      ChiselStage.emitCHIRRTL(new Module {
        val built = b.build(least(b))
        names = built.io.elements.keySet.toSet
        built.io := DontCare
      })
      val pins  = b.ports.map(_._1) ++ b.rw.map(_.pin) ++ Seq(b.start, b.busy, b.done) ++ b.ties.map(_.pin)
      assert(pins.toSet == names, s"${b.name}: catalogue ${pins.sorted}, IO ${names.toSeq.sorted}")
    }
  }
}
