package snax.forge

import java.nio.file.{Files, Paths}

/** snax-gvsoc's datapath vectors (`tests/snax/datapath/vectors.txt`, written from NumPy by that folder's `gen.py`; its
  * README gives the format), the numbers both sides check their datapaths on: per test a catalogue datapath, its params
  * and per step its enables, its inputs and the output its eval gives before the clock edge. Read from the folder
  * `SNAX_GVSOC_DATAPATH_VECTORS` names (e.g. `gvsoc/snax-gvsoc/tests/snax/datapath`); without it a spec cancels.
  * snax-gvsoc's `tb.cpp` reads the same file the same way.
  */
object SnaxDatapathVectors {

  val Env = "SNAX_GVSOC_DATAPATH_VECTORS"

  /** One step: fire, first, read, lane, en, load, wslot, rslot; each input's lanes; the output's lanes. */
  case class Step(
    fire:  Boolean,
    first: Boolean,
    read:  Boolean,
    lane:  Int,
    en:    Boolean,
    load:  Boolean,
    wslot: Int,
    rslot: Int,
    ins:   Seq[Seq[BigInt]],
    out:   Seq[BigInt]
  )

  case class Test(name: String, datapath: String, header: Map[String, Seq[String]], steps: Seq[Step]) {

    /** The datapath's params as the catalogue names them (`Datapaths`), from the test's header. */
    def params: Map[String, ujson.Value] = {
      def one(k: String) = header(k).head
      val ins = header("in")
      datapath match {
        case "ElementwiseLanes" =>
          Map(
            "inTypes" -> ujson.Arr.from(ins),
            "opType"  -> ujson.Str(one("optype")),
            "outType" -> ujson.Str(one("out")),
            "lanes"   -> ujson.Num(one("lanes").toInt),
            "op"      -> ujson.Str(one("op"))
          )
        case "TreeAccumulate"   =>
          Map(
            "inType"  -> ujson.Str(ins.head),
            "outType" -> ujson.Str(one("out")),
            "lanes"   -> ujson.Num(one("lanes").toInt),
            "op"      -> ujson.Str(one("op"))
          )
        case "Mac"              =>
          Map(
            "inTypes"    -> ujson.Arr.from(ins),
            "opType"     -> ujson.Str(one("optype")),
            "outType"    -> ujson.Str(one("out")),
            "lanes"      -> ujson.Num(one("lanes").toInt),
            "accumulate" -> ujson.Bool(one("accumulate") == "1"),
            "regOut"     -> ujson.Bool(one("regOut") == "1")
          )
        case "SerialAccumulate" =>
          Map(
            "inType"  -> ujson.Str(ins.head),
            "outType" -> ujson.Str(one("out")),
            "lanes"   -> ujson.Num(one("lanes").toInt)
          )
        case "LinkBuffer"       =>
          Map(
            "depth" -> ujson.Num(one("depth").toInt),
            "lanes" -> ujson.Num(one("lanes").toInt),
            "dtype" -> ujson.Str(one("out"))
          )
        case other              => throw new IllegalArgumentException(s"$name: no datapath $other")
      }
    }
  }

  private def numbers(text: String): Seq[BigInt] = text.trim.split("\\s+").filter(_.nonEmpty).map(BigInt(_)).toSeq

  def parse(text: String): Seq[Test] = {
    var tests  = Vector.empty[Test]
    var header = Map.empty[String, Seq[String]]
    var steps  = Vector.empty[Step]
    var name   = ""
    def close(): Unit = if (name.nonEmpty) tests :+= Test(name, header("datapath").head, header, steps)
    for (raw <- text.linesIterator) {
      val line = raw.takeWhile(_ != '#').trim
      if (line.nonEmpty) line.split("\\s+").toList match {
        case "test" :: n :: Nil => close(); name = n; header = Map.empty; steps = Vector.empty
        case "step" :: _        =>
          val Array(head, rest) = line.stripPrefix("step").split(":", 2)
          val Array(ins, out)   = rest.split("=>", 2)
          val e                 = head.trim.split("\\s+").map(_.toInt)
          steps :+= Step(
            e(0) == 1,
            e(1) == 1,
            e(2) == 1,
            e(3),
            e(4) == 1,
            e(5) == 1,
            e(6),
            e(7),
            ins.split(";").toSeq.map(numbers),
            numbers(out)
          )
        case k :: vs            => header += k -> vs
        case Nil                =>
      }
    }
    close()
    tests
  }

  /** Every test of `vectors.txt` in the folder `Env` names; None when it is not set. */
  def all: Option[Seq[Test]] =
    sys.env.get(Env).map(dir => parse(Files.readString(Paths.get(dir).resolve("vectors.txt"))))
}
