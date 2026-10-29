package snax.forge

import chisel3._

import snax.forge.datapath.{ElementwiseLanes, TreeAccumulate}
import snax.forge.elementwise.ElementwiseOp
import snax.forge.shell.AcceleratorSpec

/** The datapath catalogue: name -> signature and builder (D129, D130, D147, D149; C§17's datapath catalogue).
  *
  * The signature is what `hw/chisel/blocks.json` holds and `Emit --describe` prints: the params with their types and
  * limits, the data ports with their directions, the latency L and the span its control unit runs it at, its pins (the
  * enables it takes from the control unit: `fire`, `first`, `read`, `lane`, `en`, `load`, the slots) and `fill`, how
  * each param comes from the unit (or, for a `LinkBuffer`, the link) of `accelerator.json`:
  *
  *   - `inputs.dtype`: the unit's input ports' dtypes, in order (a `dtypes` param), or the first input's (a `dtype`);
  *   - `outputs.dtype`: the output port's dtype;
  *   - `inputs.lanes`: the first input port's lanes;
  *   - `unit.opType`, `unit.params.<name>`: the unit's `opType` and its `params.<name>`;
  *   - `link.depth`, `link.lanes`, `link.dtype`: a buffer's link's depth and its ends' lanes and dtype.
  *
  * A port is `"in"` or `"out"`, or `{"dir": ..., "when": {param: value, ...}}`; a pin is a name, or `{"name": ...,
  * "when": {...}}`. `when` is a match: the port or pin exists only when every listed param has that value, decided at
  * elaboration (`live`); one without `when` always does. A latency or a span is an int or an expression over the params
  * in `snax_forge/expr.py`'s grammar (ints, names, `+ - * //`, `max` and `min` of two), a bool as 0 or 1 (`value`).
  *
  * The Python side (`snax_forge/gen/rtl.py`) checks each unit's timing and ports against it; `Emit --spec` builds from
  * the unit with `build`. A spec checks `describe` equal to `blocks.json`.
  *
  * Param types: `int` (`min`), `bool`, `op` (one op name of `values`: ElementwiseLanes takes every ElementwiseOp,
  * TreeAccumulate `add` only), `dtype` (an element type's NumPy name of `values`, `IntType`'s eight) and `dtypes` (a
  * list of them).
  */
object Datapaths {

  sealed trait ParamType
  case class IntParam(min: Int)           extends ParamType
  case class OpParam(values: Seq[String]) extends ParamType
  case object BoolParam                   extends ParamType
  case object DtypeParam                  extends ParamType
  case object DtypesParam                 extends ParamType

  /** A data port: its name, its direction and the params it exists for (empty: always). */
  case class Port(name: String, dir: String, when: Seq[(String, ujson.Value)] = Seq.empty)

  /** A pin: its name and the params it exists for (empty: always). */
  case class Pin(name: String, when: Seq[(String, ujson.Value)] = Seq.empty)

  /** A built datapath: the module and its IO, whose elements are the catalogue's live ports and pins. */
  case class Built(module: Module, io: Bundle)

  case class Datapath(
    name:    String,
    params:  Seq[(String, ParamType, String)], // name, type, fill
    ports:   Seq[Port],
    latency: ujson.Value,
    span:    ujson.Value,
    pins:    Seq[Pin],
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
          case r if r.startsWith("unit.params.")  => unit.params(r.stripPrefix("unit.params."))
          case other                              => throw new IllegalArgumentException(s"$name.$n: fill rule $other")
        }
        n -> value
      }.toMap

    /** A buffer's params by the fill rule: its link's depth, and its ends' lanes and dtype. */
    def fillLink(depth: Int, lanes: Int, dtype: String): Map[String, ujson.Value] =
      params.map { case (n, _, rule) =>
        n -> (rule match {
          case "link.depth" => ujson.Num(depth)
          case "link.lanes" => ujson.Num(lanes)
          case "link.dtype" => ujson.Str(dtype)
          case other        => throw new IllegalArgumentException(s"$name.$n: fill rule $other")
        })
      }.toMap

    /** The ports and pins whose `when` holds on `p`. */
    def livePorts(p: Map[String, ujson.Value]): Seq[Port] = ports.filter(x => holds(x.when, p))
    def livePins(p:  Map[String, ujson.Value]): Seq[Pin]  = pins.filter(x => holds(x.when, p))

    /** Its latency and span on `p`. */
    def latencyOf(p: Map[String, ujson.Value]): Int = value(latency, p)
    def spanOf(p:    Map[String, ujson.Value]): Int = value(span, p)
  }

  private def holds(when: Seq[(String, ujson.Value)], p: Map[String, ujson.Value]): Boolean =
    when.forall { case (k, v) => p.get(k).contains(v) }

  /** A latency or span: an int, or an expression over the params' ints and bools (module doc). */
  def value(v: ujson.Value, p: Map[String, ujson.Value]): Int =
    v match {
      case ujson.Num(n) => n.toInt
      case ujson.Str(s) =>
        val env = p.collect {
          case (k, ujson.Num(n))  => k -> n.toInt
          case (k, ujson.Bool(b)) => k -> (if (b) 1 else 0)
        }
        Expr.evaluate(s, env)
      case other        => throw new IllegalArgumentException(s"a latency or span is an int or an expression: $other")
    }

  /** `snax_forge/expr.py`'s grammar, enough for a latency or span: ints, names, `+ - * //`, parentheses and `max`,
    * `min` of two.
    */
  object Expr {
    def evaluate(text: String, env: Map[String, Int]): Int = {
      val tokens = "\\d+|[A-Za-z_][A-Za-z0-9_]*|//|[-+*(),]".r.findAllIn(text).toVector
      var at     = 0
      def peek   = if (at < tokens.size) tokens(at) else ""
      def take() = { at += 1; tokens(at - 1) }
      def expect(t: String): Unit = require(take() == t, s"expected $t in $text")
      def atom():    Int = {
        val t = take()
        if (t == "(") { val v = sum(); expect(")"); v }
        else if (t == "-") -atom()
        else if (t.head.isDigit) t.toInt
        else if (t == "max" || t == "min") {
          expect("("); val a = sum(); expect(","); val b = sum(); expect(")")
          if (t == "max") a.max(b) else a.min(b)
        } else env.getOrElse(t, throw new IllegalArgumentException(s"$t is not a param in $text"))
      }
      def product(): Int = {
        var v = atom()
        while (peek == "*" || peek == "//") { v = if (take() == "*") v * atom() else Math.floorDiv(v, atom()) }
        v
      }
      def sum():     Int = {
        var v = product()
        while (peek == "+" || peek == "-") { v = if (take() == "+") v + product() else v - product() }
        v
      }
      val v = sum()
      require(at == tokens.size, s"trailing tokens in $text")
      v
    }
  }

  private def str(p:   Map[String, ujson.Value], n: String): String = p(n).str
  private def op(name: String): Int = ElementwiseOp.names.collectFirst { case (k, v) if v == name => k }.get

  /** Every dtype a type param takes. */
  private val dtypes = IntType.names

  val all: Seq[Datapath] = Seq(
    Datapath(
      "ElementwiseLanes",
      Seq(
        ("inTypes", DtypesParam, "inputs.dtype"),
        ("opType", DtypeParam, "unit.opType"),
        ("outType", DtypeParam, "outputs.dtype"),
        ("lanes", IntParam(1), "inputs.lanes"),
        ("op", OpParam(ElementwiseOp.all.map(ElementwiseOp.name)), "unit.params.op")
      ),
      Seq(Port("a", "in"), Port("b", "in"), Port("out", "out")),
      ujson.Num(0),
      ujson.Num(1),
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
        ("lanes", IntParam(1), "inputs.lanes"),
        ("op", OpParam(Seq("add")), "unit.params.op")
      ),
      Seq(Port("in", "in"), Port("out", "out")),
      ujson.Num(1),
      ujson.Num(1),
      Seq(Pin("fire"), Pin("first")),
      p => {
        require(str(p, "op") == "add", s"TreeAccumulate adds, op ${str(p, "op")}")
        val m = Module(new TreeAccumulate(str(p, "inType"), str(p, "outType"), p("lanes").num.toInt))
        Built(m, m.io)
      }
    ),
    Datapath(
      "Mac",
      Seq(
        ("inTypes", DtypesParam, "inputs.dtype"),
        ("opType", DtypeParam, "unit.opType"),
        ("outType", DtypeParam, "outputs.dtype"),
        ("lanes", IntParam(1), "inputs.lanes"),
        ("accumulate", BoolParam, "unit.params.accumulate"),
        ("regOut", BoolParam, "unit.params.regOut")
      ),
      Seq(
        Port("a", "in"),
        Port("b", "in"),
        Port("c", "in", Seq("accumulate" -> ujson.False)),
        Port("out", "out")
      ),
      ujson.Str("max(accumulate, regOut)"),
      ujson.Num(1),
      Seq(
        Pin("fire", Seq("accumulate" -> ujson.True)),
        Pin("first", Seq("accumulate" -> ujson.True)),
        Pin("en", Seq("accumulate" -> ujson.False, "regOut" -> ujson.True))
      ),
      _ => throw new NotImplementedError("Mac: B1x F2 step 1")
    ),
    Datapath(
      "SerialAccumulate",
      Seq(
        ("inType", DtypeParam, "inputs.dtype"),
        ("outType", DtypeParam, "outputs.dtype"),
        ("lanes", IntParam(1), "inputs.lanes")
      ),
      Seq(Port("in", "in"), Port("out", "out")),
      ujson.Num(1),
      ujson.Str("lanes"),
      Seq(Pin("read"), Pin("lane"), Pin("first")),
      _ => throw new NotImplementedError("SerialAccumulate: B1x F2 step 1")
    ),
    Datapath(
      "LinkBuffer",
      Seq(
        ("depth", IntParam(1), "link.depth"),
        ("lanes", IntParam(1), "link.lanes"),
        ("dtype", DtypeParam, "link.dtype")
      ),
      Seq(Port("in", "in"), Port("out", "out")),
      ujson.Num(1),
      ujson.Num(1),
      Seq(Pin("load"), Pin("wslot"), Pin("rslot")),
      _ => throw new NotImplementedError("LinkBuffer: B1x F2 step 1")
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
        "ports"   -> ujson.Obj.from(d.ports.map { x =>
          x.name -> (if (x.when.isEmpty) ujson.Str(x.dir)
                     else ujson.Obj("dir" -> x.dir, "when" -> ujson.Obj.from(x.when)))
        }),
        "latency" -> d.latency,
        "span"    -> d.span,
        "pins"    -> ujson.Arr.from(d.pins.map { x =>
          if (x.when.isEmpty) ujson.Str(x.name) else ujson.Obj("name" -> x.name, "when" -> ujson.Obj.from(x.when))
        }),
        "fill"    -> ujson.Obj.from(d.params.map { case (n, _, rule) => n -> ujson.Str(rule) })
      )
    })

  private def paramJson(t: ParamType): ujson.Value =
    t match {
      case IntParam(min)   => ujson.Obj("type" -> "int", "min" -> ujson.Num(min))
      case OpParam(values) => ujson.Obj("type" -> "op", "values" -> ujson.Arr.from(values))
      case BoolParam       => ujson.Obj("type" -> "bool")
      case DtypeParam      => ujson.Obj("type" -> "dtype", "values" -> ujson.Arr.from(dtypes))
      case DtypesParam     => ujson.Obj("type" -> "dtypes", "values" -> ujson.Arr.from(dtypes))
    }
}
