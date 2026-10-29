package snax.forge

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._

/** snax-gvsoc's group cases (`tests/snax/group/cases/`, the format of that folder's README), the shared half of the
  * group cross-check between snax-gvsoc and Chisel: written from C§17's Control by that folder's `gen.py`, checked
  * against `SnaxGroupRule` by its `tb.cpp`, and driven through `ctrl.GroupCtrl` here, cycle for cycle. Read from the
  * folder `SNAX_GVSOC_GROUP_CASES` names (e.g. `gvsoc/snax-gvsoc/tests/snax/group/cases`); without it a spec cancels.
  */
object SnaxGroupCases {

  val Env = "SNAX_GVSOC_GROUP_CASES"

  case class UnitLine(name: String, datapath: String, latency: Int, ii: Int, span: Int, t: Int, inputs: Int)

  /** A cycle line: the outer inputs' valids (unit by unit), the output's ready; per unit fire and give; per buffer its
    * occupancy after the cycle; busy.
    */
  case class Cycle(
    valids:    Seq[Boolean],
    ready:     Boolean,
    fire:      Seq[Boolean],
    give:      Seq[Boolean],
    occupancy: Seq[Int],
    busy:      Boolean
  )

  case class RunLine(firings: Int, busy: Int, stallIn: Int, stallOut: Int, flush: Int)

  /** A link line: its producer's and its consumer's names, its depth. */
  case class LinkLine(from: String, to: String, depth: Int)

  case class Case(
    name:   String,
    units:  Seq[UnitLine],
    links:  Seq[LinkLine],
    jobs:   Int,
    start:  (Int, String),
    runs:   Seq[RunLine],
    cycles: Seq[Cycle]
  ) {
    def depths: Seq[Int] = links.map(_.depth)
  }

  private val RunKeys = List("index", "firings", "busy", "stall_in", "stall_out", "flush")

  def parse(path: Path): Case = {
    val lines = Files.readAllLines(path).asScala.toSeq.map(_.takeWhile(_ != '#').trim).filter(_.nonEmpty)
    val at    = lines.indexOf("cycles")
    require(at >= 0, s"$path: no `cycles` line")
    var units = Vector.empty[UnitLine]
    var links = Vector.empty[LinkLine]
    var jobs  = 1
    var start = (-1, "")
    var runs  = Vector.empty[RunLine]
    for (line <- lines.take(at)) line.split("\\s+").toList match {
      case "unit" :: n :: d :: "latency" :: l :: "ii" :: ii :: "span" :: s :: "t" :: t :: "inputs" :: i :: Nil =>
        units :+= UnitLine(n, d, l.toInt, ii.toInt, s.toInt, t.toInt, i.toInt)
      case "link" :: f :: t :: "depth" :: d :: Nil => links :+= LinkLine(f, t, d.toInt)
      case "jobs" :: j :: Nil                      => jobs  = j.toInt
      case "start" :: c :: a :: Nil                => start = (c.toInt, a)
      case "expect" :: "run" :: fields if fields.map(_.takeWhile(_ != '=')) == RunKeys =>
        val v = fields.map(_.dropWhile(_ != '=').drop(1).toInt)
        runs :+= RunLine(v(1), v(2), v(3), v(4), v(5))
      case _ => throw new IllegalArgumentException(s"$path: `$line` is not a header line")
    }
    val buffers = links.count(_.depth > 0)
    val ins    = units.map(_.inputs).sum
    val cycles = lines.drop(at + 1).map { line =>
      val cols = line.split("\\|").map(_.trim.split("\\s+").filter(_.nonEmpty).map(_.toInt).toSeq).toSeq
      require(cols.size == 2 + units.size + (if (buffers > 0) 1 else 0), s"$path: `$line`")
      val head = cols.head
      val fg   = cols.slice(1, 1 + units.size)
      Cycle(
        head.take(ins).map(_ == 1),
        head(ins) == 1,
        fg.map(_(0) == 1),
        fg.map(_(1) == 1),
        if (buffers > 0) cols(1 + units.size) else Seq.empty,
        cols.last.head == 1
      )
    }
    Case(path.getFileName.toString.stripSuffix(".txt"), units, links, jobs, start, runs, cycles)
  }

  /** Every case of the folder `Env` names, sorted by file name; None when it is not set. */
  def all: Option[Seq[Case]] =
    sys.env.get(Env).map { dir =>
      Files.list(Paths.get(dir)).iterator.asScala.filter(_.toString.endsWith(".txt")).toSeq.sorted.map(parse)
    }
}
