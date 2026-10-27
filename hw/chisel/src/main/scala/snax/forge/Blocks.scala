package snax.forge

import chisel3._

import snax.forge.elementwise.{ElementwiseOp, ElementwiseTiledSpatial}
import snax.forge.samples.Accumulator

/** The blocks a generated accelerator can be built from: name -> signature and builder (D130).
  *
  * The signature is what `hw/chisel/blocks.json` holds and `Emit --describe` prints: the params with their types and
  * limits, the data ports with their directions, and the `csr` section (the read-write pins in CSR order with their
  * widths, the start pin, the status pins busy and done, and the pins tied to a constant). It holds no values: the
  * Python side (`snax_forge/gen/rtl.py`) resolves a design point against it into `rtl.json`, and `Emit --spec` builds
  * from that with `build`. A spec checks `describe` equal to `blocks.json`.
  *
  * Param types: `int` (`min`, and `max` when there is one), `bool`, `ops`, a list of op names from `values` (each
  * name's `ElementwiseOp` encoding), `dtype`, an element type's NumPy name from `values` (`IntType`'s eight), and
  * `dtypes`, a list of them. A read-write pin's `width` is an int or the name of an int param. A tie's `op` names an
  * `ops` param holding one op; the pin is tied to that op's encoding.
  */
object Blocks {

  sealed trait ParamType
  case class IntParam(min: Int, max: Option[Int] = None) extends ParamType
  case object BoolParam                                  extends ParamType
  case class OpsParam(values: Seq[(String, Int)])        extends ParamType
  case object DtypeParam                                 extends ParamType
  case object DtypesParam                                extends ParamType

  /** A read-write CSR pin: its width, bits or an int param's name. */
  case class RwPin(pin: String, width: Either[Int, String])

  /** A pin tied to the encoding of the one op of the `ops` param `op`. */
  case class Tie(pin: String, width: Int, op: String)

  /** A built block: the module and its IO bundle, whose elements are the catalogue's pins and ports. */
  case class Built(module: Module, io: Bundle)

  /** A block's params as `rtl.json` gives them, read by type. */
  case class Params(values: Map[String, ujson.Value]) {
    def int(name:  String): Int     = values(name).num.toInt
    def bool(name: String): Boolean = values(name).bool
    def ops(name: String): Seq[Int] =
      values(name).arr.toSeq.map(v => ElementwiseOp.names.collectFirst { case (k, n) if n == v.str => k }.get)

    /** A `dtype` param: one of `IntType.names`. */
    def dtype(name: String): String = {
      val t = values(name).str
      require(IntType.names.contains(t), s"$name: $t is not one of ${IntType.names.mkString(", ")}")
      t
    }

    /** A `dtypes` param: a list of `IntType.names`. */
    def dtypes(name: String): Seq[String] = values(name).arr.toSeq.map(v => Params(Map(name -> v)).dtype(name))
  }

  case class Block(
    name:   String,
    params: Seq[(String, ParamType)],
    ports:  Seq[(String, String)],
    rw:     Seq[RwPin],
    start:  String,
    busy:   String,
    done:   String,
    ties:   Seq[Tie],
    build:  Params => Built
  )

  private val ops = OpsParam(ElementwiseOp.all.map(op => ElementwiseOp.name(op) -> op))

  val all: Seq[Block] = Seq(
    Block(
      "ElementwiseTiledSpatial",
      Seq(
        "inTypes"        -> DtypesParam,
        "opType"         -> DtypeParam,
        "outType"        -> DtypeParam,
        "lanes"          -> IntParam(1),
        "loopCountWidth" -> IntParam(1, Some(32)),
        "supportedOps"   -> ops
      ),
      Seq("a"            -> "in", "b" -> "in", "out" -> "out"),
      Seq(RwPin("loopCount", Right("loopCountWidth"))),
      "start",
      "busy",
      "done",
      Seq(Tie("opSel", ElementwiseOp.width, "supportedOps")),
      p => {
        val m = Module(
          new ElementwiseTiledSpatial(
            inTypes        = p.dtypes("inTypes"),
            opType         = p.dtype("opType"),
            outType        = p.dtype("outType"),
            lanes          = p.int("lanes"),
            loopCountWidth = p.int("loopCountWidth"),
            supportedOps   = p.ops("supportedOps")
          )
        )
        Built(m, m.io)
      }
    ),
    Block(
      "Accumulator",
      Seq("inType" -> DtypeParam, "outType" -> DtypeParam, "lanes" -> IntParam(1)),
      Seq("in"     -> "in", "out"           -> "out"),
      Seq(RwPin("loopCount", Left(32)), RwPin("len", Left(32))),
      "start",
      "busy",
      "done",
      Seq.empty,
      p => {
        val m = Module(new Accumulator(p.dtype("inType"), p.dtype("outType"), p.int("lanes")))
        Built(m, m.io)
      }
    )
  )

  def apply(name: String): Block =
    all
      .find(_.name == name)
      .getOrElse(
        throw new IllegalArgumentException(s"no block $name (have ${all.map(_.name).mkString(", ")})")
      )

  /** The catalogue as `blocks.json` holds it. */
  def describe: ujson.Obj =
    ujson.Obj.from(all.map { b =>
      b.name -> ujson.Obj(
        "params" -> ujson.Obj.from(b.params.map { case (n, t) => n -> paramJson(t) }),
        "ports"  -> ujson.Obj.from(b.ports.map { case (n, d) => n -> ujson.Str(d) }),
        "csr"    -> ujson.Obj(
          "rw"     -> ujson.Arr.from(b.rw.map { r =>
            ujson.Obj("pin" -> r.pin, "width" -> r.width.fold(ujson.Num(_), ujson.Str(_)))
          }),
          "start"  -> b.start,
          "status" -> ujson.Obj("busy" -> b.busy, "done" -> b.done),
          "tie"    -> ujson.Arr.from(b.ties.map(t => ujson.Obj("pin" -> t.pin, "width" -> t.width, "op" -> t.op)))
        )
      )
    })

  private def paramJson(t: ParamType): ujson.Value =
    t match {
      case IntParam(min, max) =>
        ujson.Obj.from(Seq("type" -> ujson.Str("int"), "min" -> ujson.Num(min)) ++ max.map(m => "max" -> ujson.Num(m)))
      case BoolParam          => ujson.Obj("type" -> "bool")
      case OpsParam(vals)     =>
        ujson.Obj("type" -> "ops", "values" -> ujson.Obj.from(vals.map { case (n, e) => n -> ujson.Num(e) }))
      case DtypeParam         => ujson.Obj("type" -> "dtype", "values" -> ujson.Arr.from(IntType.names))
      case DtypesParam        => ujson.Obj("type" -> "dtypes", "values" -> ujson.Arr.from(IntType.names))
    }
}
