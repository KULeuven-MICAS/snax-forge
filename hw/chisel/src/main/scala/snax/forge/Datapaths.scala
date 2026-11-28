package snax.forge

import chisel3._

import snax.forge.datapath.{
  DatapathModule,
  HaloLink,
  LaneAccumulate,
  LinkBuffer,
  Mac,
  SerialAccumulate,
  SimdLanes,
  TreeAccumulate
}
import snax.forge.elementwise.ElementwiseOp
import snax.forge.shell.AcceleratorSpec

/** The datapath catalogue: name -> signature and builder (the datapath catalogue of CONTRACTS, "Accelerator RTL spec").
  *
  * The signature is what `hw/chisel/blocks.json` holds and `Emit --describe` prints: the params with their types and
  * limits, the data ports with their directions, the latency L and the span its control unit runs it at, its register
  * bits (`flops`), its pins (the enables it takes from the control unit: `fire`, `first`, `read`, `lane`, `en`, `load`,
  * `row`, `drain`, the slots) and `fill`, how each param comes from the unit (or, for a `LinkBuffer`, the link) of
  * `accelerator.json`:
  *
  *   - `inputs.dtype`: the unit's input ports' dtypes but a seed's, in order (a `dtypes` param), or the first input's
  *     (a `dtype`);
  *   - `outputs.dtype`: the output port's dtype;
  *   - `inputs.lanes`: the first input port's lanes;
  *   - `inputs.count`: the number of input ports (SimdLanes' `streams`);
  *   - `unit.opType`, `unit.params.<name>`: the unit's `opType` and its `params.<name>` (an `int` the unit does not
  *     have, its `min`: a `Mac`'s `rows` 1, the plain `Mac`);
  *   - `unit.scalars.<slot>`: whether the unit reads that slot from its CSR (a `bool`: its `scalars` has it);
  *   - `unit.ports.seed`: whether the unit has a seed port (a `bool`: a fold's start, read once per job);
  *   - `link.depth`, `link.lanes`, `link.dtype`: a buffer's link's depth and its ends' lanes and dtype;
  *   - `link.taps`: a halo link's taps (`HaloLink`).
  *
  * A port is `"in"` or `"out"`, or `{"dir": ..., "when": {param: value, ...}}`; a pin is a name, or `{"name": ...,
  * "when": {...}}`. `when` is a match: the port or pin exists only when every listed param has that value, or, for a
  * value `{"min": m}`, an int of at least m, decided at elaboration (`holds`); one without `when` always does. A
  * latency, a span or `flops` is an int or an expression over the params in `snax_forge/expr.py`'s grammar, a bool as 0
  * or 1 and `width(t)` a dtype param's bits, which the Python side evaluates (`snax_forge/gen/rtl.py`). `flops` is the
  * block's register bits on its params, every flop it declares: the block cost model (`snax_forge/char`) prices a
  * register as its bits; `BlockFlopsSpec` counts them in the emitted SV.
  *
  * The Python side (`snax_forge/gen/rtl.py`) checks each unit's timing and ports against it; `Emit --spec` builds from
  * the unit with `build`. A spec checks `describe` equal to `blocks.json`.
  *
  * Param types: `int` (`min`), `bool`, `op` (one op name of `values`: SimdLanes takes every ElementwiseOp and its own
  * `SimdLanes.NamedNames`, TreeAccumulate `add` only), `dtype` (an element type's NumPy name of `values`, `IntType`'s
  * eight), `dtypes` (a list of them) and `const` (an integer of any size or null, SimdLanes' slots `k` ... `hi`: null
  * where the unit has no such param, `k` at two streams; `const`). A `unit.params.<name>` the unit does not have fills
  * null, or an `int`'s `min`.
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
          case "inputs.dtype" if t == DtypesParam => // a seed is no operand (`unit.ports.seed` says it is there)
            ujson.Arr.from(unit.inPorts.filterNot(_ == "seed").map(unit.port(_).dtype))
          case "inputs.dtype"                     => ujson.Str(inPorts.head.dtype)
          case "outputs.dtype"                    => ujson.Str(unit.port(unit.outPorts.head).dtype)
          case "inputs.lanes"                     => ujson.Num(inPorts.head.lanes)
          case "inputs.count"                     => ujson.Num(inPorts.size)
          case "unit.opType"                      => ujson.Str(unit.opType.get)
          case r if r.startsWith("unit.params.")  =>
            unit.params.getOrElse(
              r.stripPrefix("unit.params."),
              t match {
                case IntParam(min) => ujson.Num(min)
                case _             => ujson.Null
              }
            )
          case r if r.startsWith("unit.scalars.") =>
            ujson.Bool(unit.scalars.exists(_._1 == r.stripPrefix("unit.scalars.")))
          case "unit.ports.seed"                  => ujson.Bool(unit.inPorts.contains("seed"))
          case other                              => throw new IllegalArgumentException(s"$name.$n: fill rule $other")
        }
        n -> value
      }.toMap

    /** A link's datapath's params by the fill rule: its ends' lanes and dtype, and a buffer's depth or a halo link's
      * taps (the one its rule names; 0 for the other, which it does not read).
      */
    def fillLink(depth: Int, lanes: Int, dtype: String, taps: Int = 0): Map[String, ujson.Value] =
      params.map { case (n, _, rule) =>
        n -> (rule match {
          case "link.depth" => ujson.Num(depth)
          case "link.taps"  => ujson.Num(taps)
          case "link.lanes" => ujson.Num(lanes)
          case "link.dtype" => ujson.Str(dtype)
          case other        => throw new IllegalArgumentException(s"$name.$n: fill rule $other")
        })
      }.toMap

    /** The pins whose `when` holds on `p`. */
    def livePins(p: Map[String, ujson.Value]): Seq[Pin] = pins.filter(x => holds(x.when, p))

    /** `p` with each `int` param filled from `unit.params` that `p` leaves out at its `min`, as `fill` gives a unit
      * without it (a `Mac`'s `rows` 1): what `Emit --block` checks and builds.
      */
    def complete(p: Map[String, ujson.Value]): Map[String, ujson.Value] =
      p ++ params.collect {
        case (n, IntParam(min), rule) if rule.startsWith("unit.params.") && !p.contains(n) => n -> ujson.Num(min)
      }

    /** `p` as a full set of its params (`Emit --block`, after `complete`): each param once and nothing else, each value
      * of its type (an int of at least `min`, a bool, one of an op's `values`, a dtype, a non-empty list of dtypes), as
      * the Python side checks a unit's (`snax_forge/gen/rtl.py`'s `check_params`); an `IllegalArgumentException`
      * otherwise. What only the constructor knows (an opType narrower than an input, a Mac's form and its types) is its
      * own `require`.
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
    when.forall {
      case (k, ujson.Obj(m)) if m.keySet == Set("min") => p.get(k).flatMap(_.numOpt).exists(_ >= m("min").num)
      case (k, v)                                      => p.get(k).contains(v)
    }

  /** A `when` value that holds on an int of at least `m` (`holds`). */
  private def atLeast(m: Int): ujson.Value = ujson.Obj("min" -> ujson.Num(m))

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
        ("op", OpParam(ElementwiseOp.all.map(ElementwiseOp.name) ++ SimdLanes.NamedNames), "unit.params.op"),
        ("streams", IntParam(1), "inputs.count"),
        ("k", ConstParam, "unit.params.k"),
        ("k1", ConstParam, "unit.params.k1"),
        ("k2", ConstParam, "unit.params.k2"),
        ("kCsr", BoolParam, "unit.scalars.k"),
        ("k1Csr", BoolParam, "unit.scalars.k1"),
        ("k2Csr", BoolParam, "unit.scalars.k2"),
        ("k3", ConstParam, "unit.params.k3"),
        ("d", ConstParam, "unit.params.d"),
        ("lo", ConstParam, "unit.params.lo"),
        ("hi", ConstParam, "unit.params.hi"),
        ("k3Csr", BoolParam, "unit.scalars.k3")
      ),
      Seq(
        Port("a", "in"),
        Port("b", "in", Seq("streams" -> atLeast(2))),
        Port("c", "in", Seq("streams" -> ujson.Num(3))),
        Port("out", "out")
      ),
      ujson.Num(0),
      ujson.Num(1),
      ujson.Num(0),
      Seq(
        Pin("k", Seq("kCsr" -> ujson.Bool(true))),
        Pin("k1", Seq("k1Csr" -> ujson.Bool(true))),
        Pin("k2", Seq("k2Csr" -> ujson.Bool(true))),
        Pin("k3", Seq("k3Csr" -> ujson.Bool(true)))
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
          p("k2Csr").bool,
          const(p("k3")),
          p("k3Csr").bool,
          const(p("d")),
          const(p("lo")),
          const(p("hi"))
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
        "k2Csr"   -> ujson.Bool(false),
        "k3"      -> ujson.Null,
        "d"       -> ujson.Null,
        "lo"      -> ujson.Null,
        "hi"      -> ujson.Null,
        "k3Csr"   -> ujson.Bool(false)
      )
    ),
    Datapath(
      "TreeAccumulate",
      Seq(
        ("inType", DtypeParam, "inputs.dtype"),
        ("outType", DtypeParam, "outputs.dtype"),
        ("lanes", IntParam(1), "inputs.lanes"),
        ("op", OpParam(Seq("add")), "unit.params.op"),
        ("seed", BoolParam, "unit.ports.seed")
      ),
      Seq(Port("in", "in"), Port("seed", "in", Seq("seed" -> ujson.True)), Port("out", "out")),
      ujson.Num(1),
      ujson.Num(1),
      ujson.Str("width(outType)"),
      Seq(Pin("fire"), Pin("first")),
      p => {
        require(str(p, "op") == "add", s"TreeAccumulate adds, op ${str(p, "op")}")
        new TreeAccumulate(str(p, "inType"), str(p, "outType"), int(p, "lanes"), p("seed").bool)
      },
      Map(
        "inType"  -> ujson.Str("int32"),
        "outType" -> ujson.Str("int64"),
        "lanes"   -> ujson.Num(4),
        "op"      -> ujson.Str("add"),
        "seed"    -> ujson.False
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
        ("regOut", BoolParam, "unit.params.regOut"),
        ("seed", BoolParam, "unit.ports.seed"),
        ("rows", IntParam(1), "unit.params.rows")
      ),
      Seq(
        Port("a", "in"),
        Port("b", "in"),
        Port("c", "in", Seq("accumulate" -> ujson.False)),
        Port("seed", "in", Seq("seed" -> ujson.True)),
        Port("out", "out")
      ),
      ujson.Str("max(accumulate, regOut)"),
      ujson.Num(1),
      ujson.Str("lanes * width(outType) * max(accumulate, regOut) * rows"),
      Seq(
        Pin("fire", Seq("accumulate" -> ujson.True)),
        Pin("first", Seq("accumulate" -> ujson.True)),
        Pin("en", Seq("accumulate" -> ujson.False, "regOut" -> ujson.True)),
        Pin("row", Seq("rows" -> atLeast(2))),
        Pin("drain", Seq("rows" -> atLeast(2)))
      ),
      p => {
        new Mac(
          p("inTypes").arr.toSeq.map(_.str),
          str(p, "opType"),
          str(p, "outType"),
          int(p, "lanes"),
          p("accumulate").bool,
          p("regOut").bool,
          p("seed").bool,
          int(p, "rows")
        )
      },
      Map(
        "inTypes"    -> ujson.Arr("int64", "int64"),
        "opType"     -> ujson.Str("int64"),
        "outType"    -> ujson.Str("int64"),
        "lanes"      -> ujson.Num(4),
        "accumulate" -> ujson.True,
        "regOut"     -> ujson.False,
        "seed"       -> ujson.False,
        "rows"       -> ujson.Num(1)
      )
    ),
    Datapath(
      "SerialAccumulate",
      Seq(
        ("inType", DtypeParam, "inputs.dtype"),
        ("outType", DtypeParam, "outputs.dtype"),
        ("lanes", IntParam(1), "inputs.lanes"),
        ("seed", BoolParam, "unit.ports.seed")
      ),
      Seq(Port("in", "in"), Port("seed", "in", Seq("seed" -> ujson.True)), Port("out", "out")),
      ujson.Num(1),
      ujson.Str("lanes"),
      ujson.Str("width(outType)"),
      Seq(Pin("read"), Pin("lane"), Pin("first")),
      p => {
        new SerialAccumulate(str(p, "inType"), str(p, "outType"), int(p, "lanes"), p("seed").bool)
      },
      Map(
        "inType"  -> ujson.Str("int64"),
        "outType" -> ujson.Str("int64"),
        "lanes"   -> ujson.Num(4),
        "seed"    -> ujson.False
      )
    ),
    Datapath(
      "LaneAccumulate",
      Seq(
        ("inType", DtypeParam, "inputs.dtype"),
        ("outType", DtypeParam, "outputs.dtype"),
        ("lanes", IntParam(1), "inputs.lanes")
      ),
      Seq(Port("a", "in"), Port("out", "out")),
      ujson.Num(1),
      ujson.Num(1),
      ujson.Str("lanes * width(outType)"),
      Seq(Pin("fire"), Pin("first")),
      p => {
        new LaneAccumulate(str(p, "inType"), str(p, "outType"), int(p, "lanes"))
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
    ),
    Datapath(
      "HaloLink",
      Seq(
        ("taps", IntParam(2), "link.taps"),
        ("lanes", IntParam(1), "link.lanes"),
        ("dtype", DtypeParam, "link.dtype")
      ),
      Seq(Port("in", "in"), Port("taps", "out")),
      ujson.Num(0),
      ujson.Num(1),
      ujson.Str("(taps + lanes - 2) // lanes * lanes * width(dtype)"), // H = ⌈(taps − 1) / lanes⌉ beats
      Seq(Pin("load")),
      p => new HaloLink(int(p, "taps"), int(p, "lanes"), str(p, "dtype")),
      Map("taps" -> ujson.Num(3), "lanes" -> ujson.Num(8), "dtype" -> ujson.Str("uint8"))
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
