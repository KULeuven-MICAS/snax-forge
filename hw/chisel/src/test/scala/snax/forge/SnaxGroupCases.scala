package snax.forge

import java.nio.file.Path

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

  def parse(path: Path): Case = {
    val (header, rows) = CaseFile.read(path)
    var units          = Vector.empty[UnitLine]
    var links          = Vector.empty[LinkLine]
    var jobs           = 1
    var start          = (-1, "")
    var runs           = Vector.empty[RunLine]
    for (line <- header) line.split("\\s+").toList match {
      case "unit" :: n :: d :: "latency" :: l :: "ii" :: ii :: "span" :: s :: "t" :: t :: "inputs" :: i :: Nil =>
        units :+= UnitLine(n, d, l.toInt, ii.toInt, s.toInt, t.toInt, i.toInt)
      case "link" :: f :: t :: "depth" :: d :: Nil     => links :+= LinkLine(f, t, d.toInt)
      case "jobs" :: j :: Nil                          => jobs  = j.toInt
      case "start" :: c :: a :: Nil                    => start = (c.toInt, a)
      case "expect" :: "run" :: RunLine.Fields(_, run) => runs :+= run
      case _ => throw new IllegalArgumentException(s"$path: `$line` is not a header line")
    }
    val buffers = links.count(_.depth > 0)
    val ins    = units.map(_.inputs).sum
    val cycles = rows.map { line =>
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
    Case(CaseFile.name(path), units, links, jobs, start, runs, cycles)
  }

  /** Every case of the folder `Env` names (`CaseFile.all`). */
  def all: Option[Seq[Case]] = CaseFile.all(Env)(parse)
}
