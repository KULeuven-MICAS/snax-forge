package snax.forge

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

import org.scalatest.flatspec.AnyFlatSpec

/** `Emit --block NAME --params JSON --out DIR`: one block of every catalogue datapath alone, all in one call as
  * gen-block makes it, each with the ports and pins `blocks.json` gives it on its params and an asynchronous reset on
  * every flop; and the refusals.
  *
  * The ports, read off the top's SystemVerilog header (`SvHeader`): `clock` and `reset`, then per live port of
  * `blocks.json` (its `when` holding on the params) `io_<port>` (a scalar) or `io_<port>_0` to `io_<port>_<lanes - 1>`
  * (a Vec of the params' lanes), in its direction, and per live pin an input `io_<pin>`, but for a pin of no bits,
  * which firtool leaves out: `LinkBuffer`'s `wslot` and `rslot` at depth 1, `SerialAccumulate`'s `lane` at lanes 1. A
  * registered block (all but `SimdLanes` and the plain `Mac`) has every `always` block of its files on `posedge clock
  * or posedge reset`, a flop with an asynchronous reset; the others none.
  */
class BlockEmitSpec extends AnyFlatSpec {

  private val Lanes = 8

  /** A block: its datapath, params, the pins of no bits, and whether it holds a register. */
  case class Case(datapath: String, params: ujson.Obj, empty: Set[String], registered: Boolean) {
    val label: String = s"$datapath ${ujson.write(params)}"
  }

  private def i64(n: Int) = ujson.Arr.from(Seq.fill(n)(ujson.Str("int64")))

  /** A SimdLanes block at int64 and `Lanes` lanes, its params in `blocks.json`'s order (as `emitted.json` lists them):
    * each constant baked (a number) or absent (null), each CSR flag.
    */
  private def simd(
    streams: Int,
    op:      String,
    k:       ujson.Value = ujson.Null,
    k1:      ujson.Value = ujson.Null,
    k2:      ujson.Value = ujson.Null,
    kCsr:    Boolean     = false,
    k1Csr:   Boolean     = false,
    k2Csr:   Boolean     = false
  ): Case =
    Case(
      "SimdLanes",
      ujson.Obj(
        "inTypes" -> i64(streams),
        "opType"  -> "int64",
        "outType" -> "int64",
        "lanes"   -> Lanes,
        "op"      -> op,
        "streams" -> streams,
        "k"       -> k,
        "k1"      -> k1,
        "k2"      -> k2,
        "kCsr"    -> kCsr,
        "k1Csr"   -> k1Csr,
        "k2Csr"   -> k2Csr
      ),
      Set.empty,
      registered = false
    )

  private val cases = Seq(
    simd(2, "add"),
    simd(1, "max", k       = 0),
    simd(1, "max", kCsr    = true), // the pin k
    simd(2, "axpby", k1Csr = true, k2Csr = true), // the pins k1 and k2
    simd(2, "axpby", k1    = 3) // no pin
  ) ++ Seq(false, true).map { seed => // seeded, the port seed (D163)
    Case(
      "TreeAccumulate",
      ujson.Obj("inType" -> "int64", "outType" -> "int64", "lanes" -> Lanes, "op" -> "add", "seed" -> seed),
      Set.empty,
      registered = true
    )
  } ++ Seq((true, false, true), (false, false, false), (false, true, true)).map { case (acc, reg, registered) =>
    Case(
      "Mac",
      ujson.Obj(
        "inTypes"    -> i64(if (acc) 2 else 3),
        "opType"     -> "int64",
        "outType"    -> "int64",
        "lanes"      -> Lanes,
        "accumulate" -> acc,
        "regOut"     -> reg
      ),
      Set.empty,
      registered
    )
  } ++ Seq((1, false), (Lanes, false), (Lanes, true)).map { case (n, seed) =>
    Case(
      "SerialAccumulate",
      ujson.Obj("inType" -> "int64", "outType" -> "int64", "lanes" -> n, "seed" -> seed),
      if (n == 1) Set("lane") else Set.empty,
      registered = true
    )
  } ++ Seq(
    Case(
      "LaneAccumulate",
      ujson.Obj("inType" -> "int64", "outType" -> "int64", "lanes" -> Lanes),
      Set.empty,
      registered = true
    )
  ) ++ Seq(1, 2).map { d =>
    Case(
      "LinkBuffer",
      ujson.Obj("depth" -> d, "lanes" -> Lanes, "dtype" -> "int64"),
      if (d == 1) Set("wslot", "rslot") else Set.empty,
      registered = true
    )
  }

  /** Every case through one `Emit` call, each into a folder of its own. */
  private lazy val out: Map[Case, Path] = {
    val root = Files.createTempDirectory("emit-block")
    val dirs = cases.zipWithIndex.map { case (c, i) => c -> root.resolve(s"$i-${c.datapath}") }
    Emit.main(dirs.flatMap { case (c, d) =>
      Seq("--block", c.datapath, "--params", ujson.write(c.params), "--out", d.toString)
    }.toArray)
    dirs.toMap
  }

  private val Indexed = raw"(\w+)_(\d+)".r

  behavior of "Emit --block"

  it should "cover every catalogue datapath" in
    assert(cases.map(_.datapath).toSet == Datapaths.all.map(_.name).toSet)

  for (c <- cases)
    it should s"give ${c.label} blocks.json's ports and pins, an async reset and its file list" in {
      val d       = Datapaths(c.datapath)
      val p       = c.params.value.toMap
      val dir     = out(c)
      val listing = ujson.read(Files.readString(dir.resolve("emitted.json")))
      val top     = listing("top").str
      assert(listing("block").str == c.datapath)
      assert(ujson.write(listing("params")) == ujson.write(c.params))
      val files   = listing("files").arr.map(f => dir.resolve(f.str)).toSeq
      assert(files.forall(Files.isRegularFile(_)) && files.contains(dir.resolve(s"sv/$top.sv")), s"$files")

      // The header, each io_ port grouped by its name without a lane index
      val ports   = SvHeader.ports(dir.resolve(s"sv/$top.sv"), top)
      assert(ports.filter(x => Set("clock", "reset")(x._1)) == Set(("clock", "input", 1), ("reset", "input", 1)))
      val io      = ports.collect { case (n, dir, _) if n.startsWith("io_") => (n.stripPrefix("io_"), dir) }
      val grouped = io.groupBy { case (n, _) =>
        n match {
          case Indexed(base, _) if !d.ports.exists(_.name == n) && !d.pins.exists(_.name == n) => base
          case _ => n
        }
      }
      val want    =
        d.ports.filter(x => Datapaths.holds(x.when, p)).map(x => x.name -> x.dir) ++
          d.livePins(p).map(_.name).filterNot(c.empty).map(_ -> "in")
      assert(grouped.keySet == want.map(_._1).toSet, s"$top: ${ports.toSeq.sorted}")
      for ((name, dirn) <- want) {
        val got = grouped(name)
        assert(got.map(_._2) == Set(s"${dirn}put"), s"$name: $got")
        val idx = got.map(_._1).collect { case Indexed(_, i) => i.toInt }
        assert(got == Set((name, s"${dirn}put")) || idx == (0 until p("lanes").num.toInt).toSet, s"$name: $got")
      }

      // Every flop on an asynchronous reset
      val always = files
        .filter(_.toString.endsWith(".sv"))
        .flatMap(f => Files.readAllLines(f).asScala.map(_.trim).filter(_.startsWith("always")))
      if (c.registered) {
        assert(always.nonEmpty, s"$top: no flop")
        assert(always.forall(_ == "always @(posedge clock or posedge reset) begin"), s"$top: ${always.distinct}")
      } else assert(always.isEmpty, s"$top: ${always.distinct}")
    }

  private val tree = """{"inType":"int64","outType":"int64","lanes":8,"op":"add","seed":false}"""

  for (
    (why, name, json) <- Seq(
      ("an unknown name", "Adder", tree),
      ("a missing param", "TreeAccumulate", """{"inType":"int64","outType":"int64","lanes":8}"""),
      ("an extra param", "TreeAccumulate", """{"inType":"int64","outType":"int64","lanes":8,"op":"add","T":4}"""),
      ("a dtype outside its type", "TreeAccumulate", tree.replace("\"inType\":\"int64\"", "\"inType\":\"int128\"")),
      ("an op outside its type", "TreeAccumulate", tree.replace("add", "mul")),
      ("lanes below its min", "TreeAccumulate", tree.replace("8", "0")),
      ("lanes that are not an int", "TreeAccumulate", tree.replace("8", "2.5")),
      ("lanes that are a bool", "SerialAccumulate", """{"inType":"int64","outType":"int64","lanes":true}"""),
      (
        "a bool that is not one",
        "Mac",
        """{"inTypes":["int64","int64"],"opType":"int64","outType":"int64","lanes":1,"accumulate":1,"regOut":false}"""
      ),
      ("params that are not an object", "TreeAccumulate", "[]")
    )
  )
    it should s"refuse $why" in assertThrows[IllegalArgumentException](Emit.block(name, json))
}
