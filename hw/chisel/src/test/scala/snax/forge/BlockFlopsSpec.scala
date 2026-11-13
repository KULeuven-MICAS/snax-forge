package snax.forge

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.datapath.SimdLanes
import snax.forge.elementwise.ElementwiseOp

/** `flops` (`blocks.json`, the registry's `Datapath.flops`): per seed of `Seeds`, a block of every datapath at random
  * params (the Mac in its three forms) emitted alone through one `Emit --block` call, and the register bits its
  * SystemVerilog declares (`reg [n:0] name;` lines of every file, firtool's form for a flop, a Vec register one per
  * element) equal to the expression evaluated on the params (`FlopsExpr`, `snax_forge/expr.py`'s grammar as Python
  * evaluates it).
  */
class BlockFlopsSpec extends AnyFlatSpec {

  private def dtype(r: Rand, atLeast: Int = 0): String = r.pick(IntType.names.filter(IntType(_).width >= atLeast))

  /** One random block per datapath, the Mac in each form: its name and params. */
  private def blocks(r: Rand): Seq[(String, ujson.Obj)] = {
    def lanes       = r.between(1, 8)
    val elementwise = {
      val (a, b)       = (dtype(r), dtype(r))
      // two streams at an ElementwiseOp; one stream with k baked or from its CSR; or axpby, k1 baked or from its CSR
      // and k2 baked, from its CSR or neither. A baked value is in 0..100, which every type holds.
      val form         = r.between(0, 2)
      val (one, axpby) = (form == 1, form == 2)
      def baked        = ujson.Num(r.between(0, 100))
      def pick(csr: Boolean, none: Boolean): ujson.Value = if (csr || none) ujson.Null else baked
      val kCsr  = one   && r.between(0, 1) == 0
      val k1Csr = axpby && r.between(0, 1) == 0
      val k2    = if (axpby) r.between(0, 2) else 0 // 0 none, 1 baked, 2 from its CSR
      ujson.Obj(
        "inTypes" -> (if (one) ujson.Arr(a) else ujson.Arr(a, b)),
        "opType"  -> dtype(r, IntType(a).width max (if (one) 0 else IntType(b).width)),
        "outType" -> dtype(r),
        "lanes"   -> lanes,
        "op"      -> (if (axpby) SimdLanes.AxpbyName else r.pick(ElementwiseOp.all.map(ElementwiseOp.name))),
        "streams" -> (if (one) 1 else 2),
        "k"       -> pick(kCsr, !one),
        "k1"      -> pick(k1Csr, !axpby),
        "k2"      -> pick(k2 == 2, k2 == 0),
        "kCsr"    -> kCsr,
        "k1Csr"   -> k1Csr,
        "k2Csr"   -> (k2 == 2)
      )
    }
    val tree        = {
      val in = dtype(r)
      ujson.Obj(
        "inType"  -> in,
        "outType" -> dtype(r, IntType(in).width),
        "lanes"   -> lanes,
        "op"      -> "add",
        "seed"    -> r.chance(0.5)
      )
    }
    val macs        = Seq((true, false), (false, false), (false, true)).map { case (acc, reg) =>
      val out = dtype(r)
      ujson.Obj(
        "inTypes"    -> ujson.Arr.from(Seq(dtype(r), dtype(r)) ++ (if (acc) Nil else Seq(out))),
        "opType"     -> dtype(r),
        "outType"    -> out,
        "lanes"      -> lanes,
        "accumulate" -> acc,
        "regOut"     -> reg
      )
    }
    val serial      = {
      val in = dtype(r)
      ujson.Obj("inType" -> in, "outType" -> dtype(r, IntType(in).width), "lanes" -> lanes, "seed" -> r.chance(0.5))
    }
    val lane        = { // LaneAccumulate: its output at least as wide as its input
      val in = dtype(r)
      ujson.Obj("inType" -> in, "outType" -> dtype(r, IntType(in).width), "lanes" -> lanes)
    }
    val link        = ujson.Obj("depth" -> r.between(1, 6), "lanes" -> lanes, "dtype" -> dtype(r))
    Seq("SimdLanes" -> elementwise, "TreeAccumulate" -> tree) ++ macs.map("Mac" -> _) ++
      Seq("SerialAccumulate" -> serial, "LaneAccumulate" -> lane, "LinkBuffer" -> link)
  }

  private val Reg = raw"\s*reg\s+(?:\[(\d+):0\]\s+)?(\w+)\s*(?:\[(\d+):(\d+)\])?\s*;.*".r

  /** The register bits every SystemVerilog file of `dir/sv/` declares. */
  private def regBits(dir: Path): Int =
    Files
      .list(dir.resolve("sv"))
      .iterator()
      .asScala
      .filter(_.toString.endsWith(".sv"))
      .flatMap(f => Files.readAllLines(f).asScala)
      .collect { case Reg(hi, _, a, b) =>
        val width = if (hi == null) 1 else hi.toInt + 1
        val depth = if (a == null) 1 else (a.toInt - b.toInt).abs + 1
        width * depth
      }
      .sum

  behavior of "flops"

  it should "be a value for every catalogue datapath" in {
    for (d <- Datapaths.all) assert(FlopsExpr(d.flops, d.example) >= 0, d.name)
  }

  for (seed <- Seeds())
    it should s"equal the register bits of every datapath's block alone, seed $seed" in Seeds.run(seed) { r =>
      val cases = blocks(r)
      assert(cases.map(_._1).toSet == Datapaths.all.map(_.name).toSet)
      val root  = Files.createTempDirectory("block-flops")
      val dirs  = cases.indices.map(i => root.resolve(s"$i-${cases(i)._1}"))
      Emit.main(
        cases
          .zip(dirs)
          .flatMap { case ((name, p), d) =>
            Seq("--block", name, "--params", ujson.write(p), "--out", d.toString)
          }
          .toArray
      )
      for (((name, p), d) <- cases.zip(dirs)) {
        val want = FlopsExpr(Datapaths(name).flops, p.value.toMap)
        assert(regBits(d) == want, s"$name ${ujson.write(p)}: the SV declares ${regBits(d)} bits, flops is $want")
      }
    }
}

/** A catalogue expression evaluated on a block's params, as `snax_forge/expr.py` evaluates it: ints, names, `+ - * //`,
  * parentheses, `max(a, b)`, `min(a, b)` and `width(t)` (a dtype param's bits); an int param its value, a bool 0 or 1.
  */
object FlopsExpr {

  def apply(v: ujson.Value, params: Map[String, ujson.Value]): Int =
    v match {
      case ujson.Num(x) => x.toInt
      case ujson.Str(s) => new Parser(s, params).all()
      case other        => throw new IllegalArgumentException(s"not an expression: $other")
    }

  private class Parser(text: String, params: Map[String, ujson.Value]) {
    private val tokens = raw"\d+|\w+|//|[-+*(),]".r.findAllIn(text).toVector
    require(tokens.mkString == text.replace(" ", ""), s"cannot tokenize $text")
    private var at     = 0

    private def peek:   Option[String] = tokens.lift(at)
    private def take(): String         = { at += 1; tokens(at - 1) }
    private def expect(t: String): Unit = { val got = take(); require(got == t, s"$text: $t expected, got $got") }

    def all(): Int = { val x = sum(); require(at == tokens.size, s"$text: trailing ${tokens.drop(at)}"); x }

    private def sum(): Int = {
      var x = product()
      while (peek.exists(Set("+", "-"))) x = if (take() == "+") x + product() else x - product()
      x
    }

    private def product(): Int = {
      var x = unary()
      while (peek.exists(Set("*", "//"))) x = if (take() == "*") x * unary() else Math.floorDiv(x, unary())
      x
    }

    private def unary(): Int =
      if (peek.contains("-")) { take(); -unary() }
      else atom()

    private def atom(): Int = {
      val t = take()
      if (t == "(") { val x = sum(); expect(")"); x }
      else if (t.forall(_.isDigit)) t.toInt
      else if (peek.contains("(")) {
        take()
        val x = t match {
          case "width"       => IntType(params(take()).str).width
          case "max" | "min" =>
            val a = sum(); expect(","); val b = sum()
            if (t == "max") a max b else a min b
          case _             => throw new IllegalArgumentException(s"$text: call $t")
        }
        expect(")")
        x
      } else
        params(t) match {
          case ujson.Num(x)  => x.toInt
          case ujson.Bool(b) => if (b) 1 else 0
          case other         => throw new IllegalArgumentException(s"$text: $t is $other, not an int")
        }
    }
  }
}
