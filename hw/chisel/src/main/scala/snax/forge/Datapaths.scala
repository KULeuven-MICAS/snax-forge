package snax.forge

import chisel3._

import snax.forge.datapath.{DatapathModule, LinkBuffer, Mac, SerialAccumulate, SimdLanes, TreeAccumulate}
import snax.forge.elementwise.ElementwiseOp
import snax.forge.shell.AcceleratorSpec

/** The datapath catalogue: name -> signature and builder (D129, D130, D147, D149; C§17's datapath catalogue).
  *
  * The signature is what `hw/chisel/blocks.json` holds and `Emit --describe` prints: the params with their types and
  * limits, the data ports with their directions, the latency L and the span its control unit runs it at, its register
  * bits (`flops`), its pins (the enables it takes from the control unit: `fire`, `first`, `read`, `lane`, `en`, `load`,
  * the slots) and `fill`, how each param comes from the unit (or, for a `LinkBuffer`, the link) of `accelerator.json`:
  *
  *   - `inputs.dtype`: the unit's input ports' dtypes, in order (a `dtypes` param), or the first input's (a `dtype`);
  *   - `outputs.dtype`: the output port's dtype;
  *   - `inputs.lanes`: the first input port's lanes;
  *   - `inputs.count`: the number of input ports (SimdLanes' `streams`);
  *   - `unit.opType`, `unit.params.<name>`: the unit's `opType` and its `params.<name>`;
  *   - `unit.scalars.<slot>`: whether the unit reads that slot from its CSR (a `bool`: its `scalars` has it);
  *   - `link.depth`, `link.lanes`, `link.dtype`: a buffer's link's depth and its ends' lanes and dtype.
  *
  * A port is `"in"` or `"out"`, or `{"dir": ..., "when": {param: value, ...}}`; a pin is a name, or `{"name": ...,
  * "when": {...}}`. `when` is a match: the port or pin exists only when every listed param has that value, decided at
  * elaboration (`holds`); one without `when` always does. A latency, a span or `flops` is an int or an expression over
  * the params in `snax_forge/expr.py`'s grammar, a bool as 0 or 1 and `width(t)` a dtype param's bits, which the Python
  * side evaluates (`snax_forge/gen/rtl.py`). `flops` is the block's register bits on its params, every flop it
  * declares: CHAR1's cost model prices a register as its bits (D154); `BlockFlopsSpec` counts them in the emitted SV.
  *
  * The Python side (`snax_forge/gen/rtl.py`) checks each unit's timing and ports against it; `Emit --spec` builds from
  * the unit with `build`. A spec checks `describe` equal to `blocks.json`.
  *
  * Param types: `int` (`min`), `bool`, `op` (one op name of `values`: SimdLanes takes every ElementwiseOp,
  * TreeAccumulate `add` only), `dtype` (an element type's NumPy name of `values`, `IntType`'s eight), `dtypes` (a list
  * of them) and `const` (an integer of any size or null, SimdLanes' `k`: null at two streams, where the unit has no
  * such param; `const`). A `unit.params.<name>` the unit does not have fills null.
  */
object Datapaths {

  sealed trait ParamType
  case class IntParam(min: Int)           extends ParamType
  case class OpParam(values: Seq[String]) extends ParamType
  case object BoolParam                   extends ParamType
  case object DtypeParam                  extends ParamType
  case object DtypesParam                 extends ParamType
  case object ConstParam                  extends ParamType // an integer of any size, or null (unset)

  /** A data port: its name, its direction and the params it exists for (empty: always). */
  case class Port(name: String, dir: String, when: Seq[(String, ujson.Value)] = Seq.empty)

  /** A pin: its name and the params it exists for (empty: always). */
  case class Pin(name: String, when: Seq[(String, ujson.Value)] = Seq.empty)

  case class Datapath(
    name:    String,
    params:  Seq[(String, ParamType, String)], // name, type, fill
    ports:   Seq[Port],
    latency: ujson.Value,
    span:    ujson.Value,
    flops:   ujson.Value,
    pins:    Seq[Pin],
    make:    Map[String, ujson.Value] => DatapathModule,
    example: Map[String, ujson.Value]          // the params of Emit's catalogue target
  ) {

    /** Its module on `p`, instantiated: its IO, whose elements are its live ports and pins. */
    def build(p: Map[String, ujson.Value]): Bundle = {
      val m = Module(make(p))
      DatapathModule.io(m)
    }

    /** The params of `unit` by the fill rule, from its input and output BRM ports in port order. */
    def fill(unit: AcceleratorSpec.UnitSpec): Map[String, ujson.Value] =
      params.map { case (n, t, rule) =>
        val inPorts = unit.inPorts.map(unit.port)
        val value: ujson.Value = rule match {
          case "inputs.dtype" if t == DtypesParam => ujson.Arr.from(inPorts.map(_.dtype))
          case "inputs.dtype"                     => ujson.Str(inPorts.head.dtype)
          case "outputs.dtype"                    => ujson.Str(unit.port(unit.outPorts.head).dtype)
          case "inputs.lanes"                     => ujson.Num(inPorts.head.lanes)
          case "inputs.count"                     => ujson.Num(inPorts.size)
          case "unit.opType"                      => ujson.Str(unit.opType.get)
          case r if r.startsWith("unit.params.")  => unit.params.getOrElse(r.stripPrefix("unit.params."), ujson.Null)
          case r if r.startsWith("unit.scalars.") =>
            ujson.Bool(unit.scalars.exists(_._1 == r.stripPrefix("unit.scalars.")))
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

    /** The pins whose `when` holds on `p`. */
    def livePins(p: Map[String, ujson.Value]): Seq[Pin] = pins.filter(x => holds(x.when, p))

    /** `p` as a full set of its params (`Emit --block`): each param once and nothing else, each value of its type (an
      * int of at least `min`, a bool, one of an op's `values`, a dtype, a non-empty list of dtypes), as the Python side
      * checks a unit's (`snax_forge/gen/rtl.py`'s `check_params`); an `IllegalArgumentException` otherwise. What only
      * the constructor knows (an opType narrower than an input, a Mac's form and its types) is its own `require`.
      */
    def check(p: Map[String, ujson.Value]): Unit = {
      val names            = params.map(_._1)
      val (missing, extra) = (names.filterNot(p.contains), p.keys.filterNot(names.contains).toSeq.sorted)
      if (missing.nonEmpty || extra.nonEmpty)
        throw new IllegalArgumentException(
          s"$name: params ${names.mkString(", ")}; missing [${missing.mkString(", ")}], extra [${extra.mkString(", ")}]"
        )
      for ((n, t, _) <- params) {
        val ok = (t, p(n)) match {
          case (IntParam(min), ujson.Num(x)) => x.isWhole   && x >= min && x <= Int.MaxValue
          case (BoolParam, _: ujson.Bool)    => true
          case (OpParam(vs), ujson.Str(s))   => vs.contains(s)
          case (DtypeParam, ujson.Str(s))    => dtypes.contains(s)
          case (DtypesParam, ujson.Arr(xs))  => xs.nonEmpty && xs.forall(x => x.strOpt.exists(dtypes.contains))
          case (ConstParam, v)               => scala.util.Try(const(v)).isSuccess
          case _                             => false
        }
        if (!ok)
          throw new IllegalArgumentException(s"$name.$n: ${ujson.write(p(n))} is not ${ujson.write(paramJson(t))}")
      }
    }
  }

  def holds(when: Seq[(String, ujson.Value)], p: Map[String, ujson.Value]): Boolean =
    when.forall { case (k, v) => p.get(k).contains(v) }

  /** A `const` param's value: null as None, else an integer, exact (a JSON number within ±2^53, the doubles' exact
    * integers, or a decimal string); an `IllegalArgumentException` otherwise.
    */
  def const(v: ujson.Value): Option[BigInt] =
    v match {
      case ujson.Null => None
      case ujson.Num(x) if x.isWhole && math.abs(x) <= Exact => Some(BigInt(x.toLong))
      case ujson.Str(s) if s.matches("-?[0-9]+") => Some(BigInt(s))
      case other                                 =>
        throw new IllegalArgumentException(s"${ujson.write(other)} is not an integer exact in JSON (±2^53) or null")
    }

  private val Exact = math.pow(2, 53)

  private def str(p: Map[String, ujson.Value], n: String): String = p(n).str
  private def int(p: Map[String, ujson.Value], n: String): Int    = p(n).num.toInt

  /** Every dtype a type param takes. */
  private val dtypes = IntType.names

  val all: Seq[Datapath] = Seq(
    Datapath(
      "SimdLanes",
      Seq(
        ("inTypes", DtypesParam, "inputs.dtype"),
        ("opType", DtypeParam, "unit.opType"),
        ("outType", DtypeParam, "outputs.dtype"),
        ("lanes", IntParam(1), "inputs.lanes"),
        ("op", OpParam(ElementwiseOp.all.map(ElementwiseOp.name) :+ SimdLanes.AxpbyName), "unit.params.op"),
        ("streams", IntParam(1), "inputs.count"),
        ("k", ConstParam, "unit.params.k"),
        ("k1", ConstParam, "unit.params.k1"),
        ("k2", ConstParam, "unit.params.k2"),
        ("kCsr", BoolParam, "unit.scalars.k"),
        ("k1Csr", BoolParam, "unit.scalars.k1"),
        ("k2Csr", BoolParam, "unit.scalars.k2")
      ),
      Seq(Port("a", "in"), Port("b", "in", Seq("streams" -> ujson.Num(2))), Port("out", "out")),
      ujson.Num(0),
      ujson.Num(1),
      ujson.Num(0),
      Seq(
        Pin("k", Seq("kCsr" -> ujson.Bool(true))),
        Pin("k1", Seq("k1Csr" -> ujson.Bool(true))),
        Pin("k2", Seq("k2Csr" -> ujson.Bool(true)))
      ),
      p => {
        val inTypes = p("inTypes").arr.toSeq.map(_.str)
        require(
          int(p, "streams") == inTypes.size,
          s"streams ${int(p, "streams")} with inTypes ${inTypes.mkString(", ")}"
        )
        new SimdLanes(
          inTypes,
          str(p, "opType"),
          str(p, "outType"),
          int(p, "lanes"),
          SimdLanes.opByName(str(p, "op")),
          const(p("k")),
          const(p("k1")),
          const(p("k2")),
          p("kCsr").bool,
          p("k1Csr").bool,
          p("k2Csr").bool
        )
      },
      Map(
        "inTypes" -> ujson.Arr("int32", "int32"),
        "opType"  -> ujson.Str("int32"),
        "outType" -> ujson.Str("int32"),
        "lanes"   -> ujson.Num(4),
        "op"      -> ujson.Str("add"),
        "streams" -> ujson.Num(2),
        "k"       -> ujson.Null,
        "k1"      -> ujson.Null,
        "k2"      -> ujson.Null,
        "kCsr"    -> ujson.Bool(false),
        "k1Csr"   -> ujson.Bool(false),
        "k2Csr"   -> ujson.Bool(false)
      )
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
      ujson.Str("width(outType)"),
      Seq(Pin("fire"), Pin("first")),
      p => {
        require(str(p, "op") == "add", s"TreeAccumulate adds, op ${str(p, "op")}")
        new TreeAccumulate(str(p, "inType"), str(p, "outType"), int(p, "lanes"))
      },
      Map(
        "inType"  -> ujson.Str("int32"),
        "outType" -> ujson.Str("int64"),
        "lanes"   -> ujson.Num(4),
        "op"      -> ujson.Str("add")
      )
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
      ujson.Str("lanes * width(outType) * max(accumulate, regOut)"),
      Seq(
        Pin("fire", Seq("accumulate" -> ujson.True)),
        Pin("first", Seq("accumulate" -> ujson.True)),
        Pin("en", Seq("accumulate" -> ujson.False, "regOut" -> ujson.True))
      ),
      p => {
        new Mac(
          p("inTypes").arr.toSeq.map(_.str),
          str(p, "opType"),
          str(p, "outType"),
          int(p, "lanes"),
          p("accumulate").bool,
          p("regOut").bool
        )
      },
      Map(
        "inTypes"    -> ujson.Arr("int64", "int64"),
        "opType"     -> ujson.Str("int64"),
        "outType"    -> ujson.Str("int64"),
        "lanes"      -> ujson.Num(4),
        "accumulate" -> ujson.True,
        "regOut"     -> ujson.False
      )
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
      ujson.Str("width(outType)"),
      Seq(Pin("read"), Pin("lane"), Pin("first")),
      p => {
        new SerialAccumulate(str(p, "inType"), str(p, "outType"), int(p, "lanes"))
      },
      Map("inType" -> ujson.Str("int64"), "outType" -> ujson.Str("int64"), "lanes" -> ujson.Num(4))
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
      ujson.Str("depth * lanes * width(dtype)"),
      Seq(Pin("load"), Pin("wslot"), Pin("rslot")),
      p => {
        new LinkBuffer(int(p, "depth"), int(p, "lanes"), str(p, "dtype"))
      },
      Map("depth" -> ujson.Num(2), "lanes" -> ujson.Num(4), "dtype" -> ujson.Str("int64"))
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
        "flops"   -> d.flops,
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
      case ConstParam      => ujson.Obj("type" -> "const")
    }
}
