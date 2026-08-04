package snax.forge

import java.nio.file.Files

import scala.util.{Failure, Success, Try}

import chisel3.RawModule
import snax.forge.elementwise.{ElementwiseLoop, ElementwiseOp, ElementwiseSpatial, ElementwiseTiledSpatial}

/** upickle configured so that Option[T] is null-or-absent rather than a JSON array.
  *
  * upickle's default encoding for Option is a sequence: None is `[]` and Some(x) is `[x]`. That is unambiguous and
  * completely unlike how anyone writes JSON by hand, and it would force the Python emitter to wrap every optional
  * field in a single-element list. This is the recipe from upickle's own documentation for the conventional encoding:
  * absent or null means None, a bare value means Some.
  *
  * Every ReadWriter below must be derived from THIS object, not from upickle.default, or the override does not apply.
  */
object Json extends upickle.AttributeTagged {
  override implicit def OptionWriter[T: Writer]: Writer[Option[T]] =
    implicitly[Writer[T]].comap[Option[T]] {
      case None    => null.asInstanceOf[T]
      case Some(x) => x
    }

  override implicit def OptionReader[T: Reader]: Reader[Option[T]] =
    new Reader.Delegate[Any, Option[T]](implicitly[Reader[T]].map(Some(_))) {
      override def visitNull(index: Int): Option[T] = None
    }
}

/** Descriptor-driven generator: JSON in, SystemVerilog out.
  *
  * The other half of the boundary whose Python side emits `out/descriptors`, validated against
  * `hw/descriptors/schema/hw_schema.json`. `Emit` builds from a hardcoded catalogue and exists for inspecting the
  * module library; this builds from a descriptor and is the path that will carry real kernels.
  *
  * {{{
  * pixi run chisel-hwgen -- experiments/vecadd_tiled_spatial.json
  * pixi run chisel-hwgen -- experiments/vecadd_loop.json --dry-run
  * pixi run chisel-hwgen -- desc.json --selectable --loop-count-width 16
  * }}}
  *
  * ==What is checked here, and what is not==
  *
  * upickle's decoding already enforces the structural half of the schema: a missing required field or a wrong JSON
  * type fails to decode. What it cannot see are the value constraints -- enums, `bounded: const true`, `lanes >= 1` --
  * so those are explicit `require`s in `validate`. They produce better messages than a schema validator would anyway,
  * because they can say which unit and which field.
  *
  * That is deliberately not the same as validating against the schema file. A hand-edited or third-party descriptor
  * could still satisfy every check here and violate the schema in a way nothing notices. Adding a real JSON Schema
  * validator (networknt is the usual Java choice) is the eventual fix; it is one more dependency and was not worth
  * blocking the first iteration on.
  *
  * ==Descriptor versus policy==
  *
  * The descriptor says what the SDFG says: which operation, how many lanes, how wide an element. It says nothing about
  * whether the ALU should be fixed-function or selectable, or how many bits the trip counter needs, because an SDFG
  * has no opinion on either. Those are build decisions and they arrive on the command line, in `Policy`. Keeping the
  * split sharp is what stops the descriptor from slowly accumulating generator flags.
  */
object HwGen {

  // -------------------------------------------------------------------------
  // Descriptor model -- field names must match the JSON exactly
  // -------------------------------------------------------------------------

  /** An expression node: a port reference, an integer literal, or a binary operation.
    *
    * Custom ReadWriter because the shapes are discriminated by which key is present, not by a type tag. upickle's
    * derived sealed-trait support writes a `$type` field, which the schema does not have and should not grow.
    */
  sealed trait Expr
  case class PortRef(port: String) extends Expr
  case class ConstInt(const: Long) extends Expr
  case class BinOp(op: String, args: Seq[Expr]) extends Expr

  object Expr {
    implicit val rw: Json.ReadWriter[Expr] = Json.readwriter[ujson.Value].bimap[Expr](
      {
        case PortRef(p)   => ujson.Obj("port" -> p)
        case ConstInt(c)  => ujson.Obj("const" -> ujson.Num(c.toDouble))
        case BinOp(o, as) => ujson.Obj("op" -> o, "args" -> ujson.Arr.from(as.map(Json.writeJs(_))))
      },
      json => {
        val o = json.obj
        if (o.contains("port")) PortRef(o("port").str)
        else if (o.contains("const")) ConstInt(o("const").num.toLong)
        else if (o.contains("op")) BinOp(o("op").str, o("args").arr.map(Json.read[Expr](_)).toSeq)
        else fail(s"expression node has none of port/const/op: ${ujson.write(json)}")
      }
    )
  }

  case class Port(
    name:      String,
    direction: String,
    bind:      String,
    dtype:     String,
    bits:      Int,
    signed:    Boolean,
    subset:    Option[String] = None,
    shape:     Option[Seq[String]] = None
  )
  object Port { implicit val rw: Json.ReadWriter[Port] = Json.macroRW }

  case class Shape(
    lanes:    Int,
    trips:    String,
    elements: String,
    bounded:  Boolean,
    ragged:   Option[Boolean] = None
  )
  object Shape { implicit val rw: Json.ReadWriter[Shape] = Json.macroRW }

  case class Datapath(expr: Expr, label: Option[String] = None, source: Option[String] = None)
  object Datapath { implicit val rw: Json.ReadWriter[Datapath] = Json.macroRW }

  /** Named UnitDesc rather than Unit, which is taken by scala.Unit. */
  case class UnitDesc(
    module_name: String,
    family:      String,
    variant:     String,
    shape:       Shape,
    datapath:    Datapath,
    ports:       Seq[Port]
  )
  object UnitDesc { implicit val rw: Json.ReadWriter[UnitDesc] = Json.macroRW }

  case class Cluster(name: String, units: Seq[UnitDesc])
  object Cluster { implicit val rw: Json.ReadWriter[Cluster] = Json.macroRW }

  case class Provenance(
    tool:      Option[String] = None,
    version:   Option[String] = None,
    recipe:    Option[String] = None,
    sdfg:      Option[String] = None,
    timestamp: Option[String] = None
  )
  object Provenance { implicit val rw: Json.ReadWriter[Provenance] = Json.macroRW }

  case class Descriptor(
    schema_version: String,
    cluster:        Cluster,
    generator:      Option[Provenance] = None
  )
  object Descriptor { implicit val rw: Json.ReadWriter[Descriptor] = Json.macroRW }

  // -------------------------------------------------------------------------
  // Generator policy -- decisions the SDFG does not make
  // -------------------------------------------------------------------------

  /** @param selectable
    *   emit an ALU that can perform every operation, rather than only the one the descriptor names. The descriptor is
    *   always fixed -- a tasklet is one computation -- so this is purely a build decision, taken when several units
    *   are expected to share a datapath or when the same RTL must serve more than one kernel.
    * @param loopCountWidth
    *   bits of trip counter. The descriptor's `trips` may be symbolic (ceiling(N/64)) and is a runtime CSR value, so
    *   nothing in it determines this.
    */
  case class Policy(selectable: Boolean = false, loopCountWidth: Int = 32) {
    def opsFor(op: Int): Seq[Int] = if (selectable) ElementwiseOp.all else Seq(op)
  }

  /** Operation name to encoding, derived from the Chisel-side table so the two cannot disagree. The schema's `op` enum
    * is the third copy of this list; generating one from the other is the eventual fix.
    */
  private val encoding: Map[String, Int] = ElementwiseOp.names.map(_.swap)

  private val MajorVersion = "1"

  // -------------------------------------------------------------------------
  // Loading and validation
  // -------------------------------------------------------------------------

  private def fail(msg: String): Nothing = throw new IllegalArgumentException(msg)

  def load(path: String): Descriptor = {
    // Repo-root anchored: sbt's working directory is hw/chisel, so a path
    // typed at the repo root would otherwise not be found. See RepoPaths.
    val file = RepoPaths.resolve(path)
    if (!Files.exists(file)) {
      fail(s"descriptor not found: $path (looked in ${file.getParent})")
    }
    val text = new String(Files.readAllBytes(file), "UTF-8").stripPrefix("\uFEFF")
    if (text.trim.isEmpty) fail(s"descriptor is empty: $path")
    Try(Json.read[Descriptor](text)) match {
      case Success(d) => d
      case Failure(e) => fail(s"$path is not a readable descriptor: ${e.getMessage}")
    }
  }

  /** Everything the schema asserts that upickle cannot, plus the cross-field checks a schema cannot express at all. */
  def validate(d: Descriptor): Descriptor = {
    val major = d.schema_version.takeWhile(_ != '.')
    require(
      major == MajorVersion,
      s"schema_version ${d.schema_version} has major version $major; this generator understands $MajorVersion.x"
    )
    require(d.cluster.units.nonEmpty, "cluster has no units")

    val names = d.cluster.units.map(_.module_name)
    require(
      names.distinct.size == names.size,
      s"module_name must be unique within a cluster: ${names.diff(names.distinct).distinct.mkString(", ")}"
    )

    d.cluster.units.foreach(validateUnit)
    d
  }

  private def validateUnit(u: UnitDesc): Unit = {
    def bad(msg: String): Nothing = fail(s"${u.module_name}: $msg")

    // The one check that is a design decision rather than type-checking: lanes
    // is an elaboration-time width, so a symbolic one has no value to pass a
    // constructor. Specialize the SDFG first.
    if (!u.shape.bounded) bad("shape.bounded is false -- lanes did not resolve to a concrete width")
    if (u.shape.lanes < 1) bad(s"lanes must be at least 1, got ${u.shape.lanes}")

    if (u.family != "elementwise") bad(s"unknown family '${u.family}'")
    if (!Set("loop", "spatial", "tiled_spatial").contains(u.variant)) {
      bad(s"unknown elementwise variant '${u.variant}'")
    }
    // W = 1 is what makes a loop a loop; any other width is a spatial variant
    // wearing the wrong label.
    if (u.variant == "loop" && u.shape.lanes != 1) {
      bad(s"variant 'loop' requires lanes = 1, got ${u.shape.lanes}")
    }

    val ins  = u.ports.filter(_.direction == "in")
    val outs = u.ports.filter(_.direction == "out")
    u.ports.foreach { p =>
      if (!Set("in", "out").contains(p.direction)) bad(s"port ${p.name}: unknown direction '${p.direction}'")
      if (!Set(8, 16, 32, 64).contains(p.bits)) bad(s"port ${p.name}: bits must be a power of two, got ${p.bits}")
    }
    if (ins.size != 2 || outs.size != 1) {
      bad(s"an elementwise binary unit needs 2 inputs and 1 output, got ${ins.size} and ${outs.size}")
    }

    // Mixed precision is a real case -- int8 operands accumulating into int32 --
    // but no datapath here handles it, so it is refused rather than silently
    // truncated to whichever port happened to be read first.
    val widths = u.ports.map(_.bits).distinct
    if (widths.size != 1) bad(s"ports disagree on element width: ${widths.sorted.mkString(", ")}")
    val signs = u.ports.map(_.signed).distinct
    if (signs.size != 1) bad("ports disagree on signedness")

    // A leaf naming a port that does not exist means the descriptor cannot be
    // wired, which a schema cannot catch because it is a cross-field property.
    val binds = u.ports.map(_.bind).toSet
    leaves(u.datapath.expr).foreach { leaf =>
      if (!binds.contains(leaf)) bad(s"expression references '$leaf', which no port binds to")
    }
  }

  private def leaves(e: Expr): Seq[String] = e match {
    case PortRef(p)  => Seq(p)
    case ConstInt(_) => Seq.empty
    case BinOp(_, a) => a.flatMap(leaves)
  }

  // -------------------------------------------------------------------------
  // Descriptor to module
  // -------------------------------------------------------------------------

  /** Reduce the expression tree to one ALU operation.
    *
    * Today this handles exactly one binary node over two port references, which is every elementwise kernel we can
    * currently raise. `c = a + b * 2` decodes and validates fine and fails here, which is the honest place for it to
    * fail: the tree is expressible, the hardware for it is not yet. Widening this means composing lanes rather than
    * selecting one, so it is a real piece of work and not a missing case.
    */
  def opOf(u: UnitDesc): Int = u.datapath.expr match {
    case BinOp(op, Seq(PortRef(_), PortRef(_))) =>
      encoding.getOrElse(op, fail(s"${u.module_name}: no Chisel encoding for operation '$op'"))
    case BinOp(op, _) =>
      fail(s"${u.module_name}: '$op' over a nested expression is not yet supported; only two port operands")
    case _ =>
      fail(s"${u.module_name}: datapath must be a single binary operation")
  }

  def build(u: UnitDesc, policy: Policy): RawModule = {
    val bits   = u.ports.head.bits
    val signed = u.ports.head.signed
    val ops    = policy.opsFor(opOf(u))

    u.variant match {
      case "loop" =>
        new ElementwiseLoop(bits, policy.loopCountWidth, ops, signed)
      // lanes is the only thing separating these two from each other; the
      // trip counter is what separates them structurally. Both carry the same
      // datapath width, which is why `variant` selects a module rather than a
      // circuit.
      case "spatial" =>
        new ElementwiseSpatial(bits, u.shape.lanes, ops, signed)
      case "tiled_spatial" =>
        new ElementwiseTiledSpatial(bits, u.shape.lanes, policy.loopCountWidth, ops, signed)
      case other =>
        fail(s"${u.module_name}: unreachable variant '$other'")
    }
  }

  // -------------------------------------------------------------------------
  // CLI
  // -------------------------------------------------------------------------

  private def describe(u: UnitDesc, policy: Policy): String = {
    // Every non-trivial expression is hoisted into a val rather than embedded
    // in the string. Nested interpolators inside a triple-quoted block parse,
    // but they defeat editor brace matching and are miserable to edit; there
    // is nothing here that needs to be inline.
    val op       = ElementwiseOp.name(opOf(u))
    val kind     = if (policy.selectable) s"selectable (fixed op would be $op)" else s"fixed $op"
    val sign     = if (u.ports.head.signed) "signed" else "unsigned"
    val bits     = u.ports.head.bits
    val portList = u.ports.map(p => p.name + "(" + p.direction + "->" + p.bind + ")").mkString(", ")

    s"""  ${u.module_name}
       |    family/variant  ${u.family} / ${u.variant}
       |    lanes (W)       ${u.shape.lanes}
       |    trips (T)       ${u.shape.trips}   [runtime CSR, not elaborated]
       |    element         $bits bits, $sign
       |    datapath        $kind
       |    ports           $portList""".stripMargin
  }

  def main(args: Array[String]): Unit = {
    val positional = args.filterNot(_.startsWith("--"))
    val flags      = args.filter(_.startsWith("--")).toSet

    if (positional.isEmpty || flags.contains("--help")) {
      println("usage: HwGen <descriptor.json> [--out DIR] [--selectable] [--loop-count-width N] [--dry-run]")
      println("  --dry-run           decode, validate and report; do not elaborate")
      println("  --selectable        emit a full ALU rather than only the operation named")
      println("  --loop-count-width  bits of trip counter (default 32)")
      sys.exit(if (positional.isEmpty) 1 else 0)
    }

    // The descriptor path is always explicit. It lives in a gitignored
    // directory today and will move to out/descriptors once Python emits it,
    // so a baked-in default would be wrong within the week.
    val path = positional.head
    val policy = Policy(
      selectable = flags.contains("--selectable"),
      loopCountWidth = args.sliding(2).collectFirst { case Array("--loop-count-width", n) => n.toInt }.getOrElse(32)
    )

    val descriptor =
      try validate(load(path))
      catch {
        case e: IllegalArgumentException =>
          Console.err.println(s"[snax-forge] $path rejected: ${e.getMessage}")
          sys.exit(1)
      }

    val provenance = descriptor.generator.flatMap(_.recipe).getOrElse("<unknown recipe>")
    println(s"[snax-forge] ${RepoPaths.display(RepoPaths.resolve(path))}")
    println(s"[snax-forge] cluster ${descriptor.cluster.name} from $provenance, ${descriptor.cluster.units.size} unit(s)")
    descriptor.cluster.units.foreach(u => println(describe(u, policy)))

    if (flags.contains("--dry-run")) {
      println("[snax-forge] dry run, nothing elaborated")
      return
    }

    val outDir = Emit.outDirFrom(args)
    descriptor.cluster.units.foreach { u =>
      // Emit's helper, not a copy of it: two elaboration paths with two copies
      // of the firtool flags would eventually disagree.
      val emitted = Emit.emit(build(u, policy), outDir)
      println(f"[snax-forge] ${u.module_name}%-34s -> $emitted%s.sv")
    }
    println(s"[snax-forge] emitted ${descriptor.cluster.units.size} module(s) to $outDir")
  }
}