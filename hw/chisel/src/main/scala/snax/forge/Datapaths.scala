package snax.forge

import chisel3._

import snax.forge.datapath.{ElementwiseLanes, TreeAccumulate}
import snax.forge.elementwise.ElementwiseOp
import snax.forge.shell.AcceleratorSpec

/** The datapath catalogue: name -> signature and builder (D130, D147).
  *
  * The signature is what `hw/chisel/blocks.json` holds and `Emit --describe` prints: the params with their types and
  * limits, the data ports with their directions, the latency L its nest controller runs it at, its pins (the enables it
  * takes from the controller: `fire`, `first`, `adv`) and `fill`, how each param comes from the unit of
  * `accelerator.json`:
  *
  *   - `inputs.dtype`: the unit's input ports' dtypes, in order (a `dtypes` param), or the one input's (a `dtype`);
  *   - `outputs.dtype`: the output port's dtype;
  *   - `inputs.lanes`: the input ports' lanes (one value: they are equal);
  *   - `unit.opType`, `unit.params.op`: the unit's `opType` and its `params.op`.
  *
  * The Python side (`snax_forge/gen/rtl.py`) checks each unit's `timing.latency` and ports against it; `Emit --spec`
  * builds from the unit with `build`. A spec checks `describe` equal to `blocks.json`.
  *
  * Param types: `int` (`min`), `op` (one op name of `values`), `dtype` (an element type's NumPy name of `values`,
  * `IntType`'s eight) and `dtypes` (a list of them).
  */
object Datapaths {

  sealed trait ParamType
  case class IntParam(min: Int) extends ParamType
  case object OpParam           extends ParamType
  case object DtypeParam        extends ParamType
  case object DtypesParam       extends ParamType

  /** A built datapath: the module and its IO, whose elements are the catalogue's ports and pins. */
  case class Built(module: Module, io: Bundle)

  case class Datapath(
    name:    String,
    params:  Seq[(String, ParamType, String)], // name, type, fill
    ports:   Seq[(String, String)],
    latency: Int,
    pins:    Seq[String],
    build:   Map[String, ujson.Value] => Built
  ) {

    /** The params of `unit` by the fill rule; `ins` and `outs` its input and output BRM ports in order. */
    def fill(unit: AcceleratorSpec.UnitSpec, ins: Seq[String], outs: Seq[String]): Map[String, ujson.Value] =
      params.map { case (n, t, rule) =>
        val inPorts = ins.map(unit.port)
        val value: ujson.Value = rule match {
          case "inputs.dtype" if t == DtypesParam => ujson.Arr.from(inPorts.map(_.dtype))
          case "inputs.dtype"                     => ujson.Str(inPorts.head.dtype)
          case "outputs.dtype"                    => ujson.Str(unit.port(outs.head).dtype)
          case "inputs.lanes"                     => ujson.Num(inPorts.head.lanes)
          case "unit.opType"                      => ujson.Str(unit.opType.get)
          case "unit.params.op"                   => unit.params("op")
          case other                              => throw new IllegalArgumentException(s"$name.$n: fill rule $other")
        }
        n -> value
      }.toMap
  }

  private def str(p:   Map[String, ujson.Value], n: String): String = p(n).str
  private def op(name: String): Int = ElementwiseOp.names.collectFirst { case (k, v) if v == name => k }.get

  val all: Seq[Datapath] = Seq(
    Datapath(
      "ElementwiseLanes",
      Seq(
        ("inTypes", DtypesParam, "inputs.dtype"),
        ("opType", DtypeParam, "unit.opType"),
        ("outType", DtypeParam, "outputs.dtype"),
        ("lanes", IntParam(1), "inputs.lanes"),
        ("op", OpParam, "unit.params.op")
      ),
      Seq("a" -> "in", "b" -> "in", "out" -> "out"),
      0,
      Seq.empty,
      p => {
        val m = Module(
          new ElementwiseLanes(
            p("inTypes").arr.toSeq.map(_.str),
            str(p, "opType"),
            str(p, "outType"),
            p("lanes").num.toInt,
            op(str(p, "op"))
          )
        )
        Built(m, m.io)
      }
    ),
    Datapath(
      "TreeAccumulate",
      Seq(
        ("inType", DtypeParam, "inputs.dtype"),
        ("outType", DtypeParam, "outputs.dtype"),
        ("lanes", IntParam(1), "inputs.lanes")
      ),
      Seq("in" -> "in", "out" -> "out"),
      1,
      Seq("fire", "first"),
      p => {
        val m = Module(new TreeAccumulate(str(p, "inType"), str(p, "outType"), p("lanes").num.toInt))
        Built(m, m.io)
      }
    )
  )

  def apply(name: String): Datapath =
    all
      .find(_.name == name)
      .getOrElse(throw new IllegalArgumentException(s"no datapath $name (have ${all.map(_.name).mkString(", ")})"))

  /** The catalogue as `blocks.json` holds it. */
  def describe: ujson.Obj =
    ujson.Obj.from(all.map { d =>
      d.name -> ujson.Obj(
        "params"  -> ujson.Obj.from(d.params.map { case (n, t, _) => n -> paramJson(t) }),
        "ports"   -> ujson.Obj.from(d.ports.map { case (n, dir) => n -> ujson.Str(dir) }),
        "latency" -> ujson.Num(d.latency),
        "pins"    -> ujson.Arr.from(d.pins),
        "fill"    -> ujson.Obj.from(d.params.map { case (n, _, rule) => n -> ujson.Str(rule) })
      )
    })

  private def paramJson(t: ParamType): ujson.Value =
    t match {
      case IntParam(min) => ujson.Obj("type" -> "int", "min" -> ujson.Num(min))
      case OpParam => ujson.Obj("type" -> "op", "values" -> ujson.Arr.from(ElementwiseOp.all.map(ElementwiseOp.name)))
      case DtypeParam  => ujson.Obj("type" -> "dtype", "values" -> ujson.Arr.from(IntType.names))
      case DtypesParam => ujson.Obj("type" -> "dtypes", "values" -> ujson.Arr.from(IntType.names))
    }
}
