package snax.forge

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._

/** snax-gvsoc's `SnaxUnit` cases, the hand-worked half of the unit cross-check between snax-gvsoc and Chisel.
  *
  * The case files (`tests/snax/unit/cases/`, one `.txt` each, in `tb.cpp`'s format: header lines, then `cycles` and one
  * line per cycle) are worked by hand from `SnaxUnit`'s rule (`snax/snax_unit.hpp`); snax-gvsoc's `tb.cpp` checks the
  * rule against them, and the Chisel specs drive their blocks through the same cases. They are read from the folder
  * `SNAX_GVSOC_UNIT_CASES` names (e.g. `gvsoc/snax-gvsoc/tests/snax/unit/cases`); without it a spec cancels, since CI
  * never fetches snax-gvsoc (D128).
  */
object SnaxUnitCases {

  val Env = "SNAX_GVSOC_UNIT_CASES"

  /** A start offered before a cycle's step: its words in `config`'s order and the answer the case expects. */
  case class Start(words: Seq[Int], answer: String)

  /** One cycle line: each input's valid, the output's ready, then the expected fire, push and busy. */
  case class Cycle(valids: Seq[Boolean], ready: Boolean, fire: Boolean, push: Boolean, busy: Boolean)

  /** An `expect run` line: the trace line at the end of a run. */
  case class RunLine(firings: Int, busy: Int, stallIn: Int, stallOut: Int, flush: Int)

  case class Case(
    name:     String,
    waitKind: String,
    latency:  Int,
    ii:       Int,
    inputs:   Int,
    lanes:    Int,
    config:   Seq[String],
    starts:   Map[Int, Start],
    runs:     Seq[RunLine],
    cycles:   Seq[Cycle]
  ) {

    /** The start's word of `role`, or `default` when `config` has no such role. */
    def word(start: Start, role: String, default: Int): Int =
      config.indexOf(role) match {
        case -1 => default
        case i  => start.words(i)
      }
  }

  /** One case file; a file of `like` lines only (`pipelined_l0`, the lockstep cases rerun as pipelined) has no cycles
    * of its own and gives None.
    */
  def parse(path: Path): Option[Case] = {
    val lines    = Files.readAllLines(path).asScala.map(_.split("#")(0).trim).filter(_.nonEmpty).toSeq
    val at       = lines.indexOf("cycles")
    var waitKind = "lockstep"
    var latency  = 0
    var ii       = 1
    var inputs   = 2
    var lanes    = 1
    var config   = Seq("count")
    var starts   = Map.empty[Int, Start]
    var runs     = Seq.empty[RunLine]
    if (at < 0) {
      require(lines.exists(_.startsWith("like ")), s"$path: no `cycles` line and no `like` line")
      return None
    }
    for (line <- lines.take(at)) line.split("\\s+").toList match {
      case "wait" :: w :: Nil                                   => waitKind = w
      case "latency" :: n :: Nil                                => latency  = n.toInt
      case "ii" :: n :: Nil                                     => ii       = n.toInt
      case "inputs" :: n :: Nil                                 => inputs   = n.toInt
      case "lanes" :: n :: Nil                                  => lanes    = n.toInt
      case "config" :: roles if roles.nonEmpty                  => config   = roles
      case "start" :: c :: rest if rest.size == config.size + 1 =>
        starts += c.toInt -> Start(rest.init.map(_.toInt), rest.last)
      case "expect" :: "run" :: _ :: "firings" :: f :: "busy" :: b :: "stall_in" :: si :: "stall_out" :: so ::
          "flush" :: fl :: Nil =>
        runs :+= RunLine(f.toInt, b.toInt, si.toInt, so.toInt, fl.toInt)
      case "like" :: _ :: Nil                                   => // only in files without cycles
      case _ => throw new IllegalArgumentException(s"$path: `$line` is not a header line")
    }
    val cycles = lines.drop(at + 1).map { line =>
      val v = line.split("\\s+").map(_ == "1").toSeq
      require(v.size == inputs + 4, s"$path: `$line` has ${v.size} columns, not ${inputs + 4}")
      Cycle(v.take(inputs), v(inputs), v(inputs + 1), v(inputs + 2), v(inputs + 3))
    }
    Some(
      Case(
        path.getFileName.toString.stripSuffix(".txt"),
        waitKind,
        latency,
        ii,
        inputs,
        lanes,
        config,
        starts,
        runs,
        cycles
      )
    )
  }

  /** Every case of the folder `Env` names, sorted by file name; None when it is not set. */
  def all: Option[Seq[Case]] =
    sys.env.get(Env).map { dir =>
      Files.list(Paths.get(dir)).iterator.asScala.filter(_.toString.endsWith(".txt")).toSeq.sorted.flatMap(parse)
    }
}
