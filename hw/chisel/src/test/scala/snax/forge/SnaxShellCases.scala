package snax.forge

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._

/** snax-gvsoc's `SnaxShell` cases, the hand-worked half of the shell cross-check between snax-gvsoc and Chisel (open
  * item 84), as `SnaxUnitCases` is the unit's.
  *
  * The case files (`tests/snax/shell/cases/`, one `.txt` each, in that folder's `tb.cpp` format: header lines, then
  * `cycles` and one line per cycle; the folder's README) are worked by hand from `SnaxShell`'s rule
  * (`snax/snax_shell.hpp`, one group's CSRs); snax-gvsoc's `tb.cpp` steps the rule with the unit's against them, and
  * `ShellCasesSpec` drives a Chisel shell wrapper through the same files. They are read from the folder
  * `SNAX_GVSOC_SHELL_CASES` names (e.g. `gvsoc/snax-gvsoc/tests/snax/shell/cases`); without it a spec cancels (the
  * submodule not checked out). CI's chisel job sets it to the submodule's cases.
  */
object SnaxShellCases {

  val Env = "SNAX_GVSOC_SHELL_CASES"

  /** A cycle's CSR access: a write of `value` to `offset`, a read of `offset`, none, or the write of an earlier line
    * still held.
    */
  sealed trait Access
  case class Write(offset: Int, value: BigInt) extends Access
  case class Read(offset: Int)                 extends Access
  case object NoAccess                         extends Access
  case object Held                             extends Access

  /** One cycle line: each input's valid, the output's ready, the access, its expected result (a write's answer cycle, a
    * read's value; None for `-` and `.`) and the expected STATUS (bit 0 busy, bit 1 done).
    */
  case class Cycle(valids: Seq[Boolean], ready: Boolean, access: Access, result: Option[BigInt], status: Int)

  /** A case: the group's read-write CSRs (`csrs`: START, JOBS, the units'; STATUS and COUNTER follow) and its unit. */
  case class Case(name: String, csrs: Seq[String], latency: Int, ii: Int, inputs: Int, cycles: Seq[Cycle]) {
    def status:  Int = csrs.size
    def counter: Int = csrs.size + 1
  }

  /** A number as `tb.cpp` reads it: decimal, or hex with `0x`. */
  private def number(token: String): BigInt =
    if (token.startsWith("0x") || token.startsWith("0X")) BigInt(token.drop(2), 16) else BigInt(token)

  private def bit(path: Path, line: String, token: String): Boolean =
    token match {
      case "0" => false
      case "1" => true
      case _   => throw new IllegalArgumentException(s"$path: `$line`: $token is not 0 or 1")
    }

  def parse(path: Path): Case = {
    val lines   = Files.readAllLines(path).asScala.map(_.split("#")(0).trim).filter(_.nonEmpty).toSeq
    val at      = lines.indexOf("cycles")
    var csrs    = Seq("START", "JOBS")
    var latency = 0
    var ii      = 1
    var inputs  = 2
    require(at >= 0, s"$path: no `cycles` line")
    for (line <- lines.take(at)) line.split("\\s+").toList match {
      case "csrs" :: "START" :: "JOBS" :: rest => csrs    = Seq("START", "JOBS") ++ rest
      case "latency" :: n :: Nil               => latency = n.toInt
      case "ii" :: n :: Nil                    => ii      = n.toInt
      case "inputs" :: n :: Nil                => inputs  = n.toInt
      case _ => throw new IllegalArgumentException(s"$path: `$line` is not a header line")
    }
    val cycles = lines.drop(at + 1).map { line =>
      val t = line.split("\\s+").toList
      require(t.size >= inputs + 4, s"$path: `$line` is too short")
      val valids           = t.take(inputs).map(bit(path, line, _))
      val ready            = bit(path, line, t(inputs))
      val (access, rest)   = t.drop(inputs + 1) match {
        case "w" :: o :: v :: r => (Write(o.toInt, number(v)), r)
        case "r" :: o :: r      => (Read(o.toInt), r)
        case "-" :: r           => (NoAccess, r)
        case "." :: r           => (Held, r)
        case _                  => throw new IllegalArgumentException(s"$path: `$line`: no access column")
      }
      val (result, status) = rest match {
        case r :: s :: Nil =>
          val res = access match {
            case NoAccess | Held =>
              require(r == "-", s"$path: `$line`: `-` and `.` have no result")
              None
            case _               => Some(number(r))
          }
          (res, s.toInt)
        case _ => throw new IllegalArgumentException(s"$path: `$line`: a result and a STATUS column expected")
      }
      Cycle(valids, ready, access, result, status)
    }
    Case(path.getFileName.toString.stripSuffix(".txt"), csrs, latency, ii, inputs, cycles)
  }

  /** Every case of the folder `Env` names, sorted by file name; None when it is not set. */
  def all: Option[Seq[Case]] =
    sys.env.get(Env).map { dir =>
      Files.list(Paths.get(dir)).iterator.asScala.filter(_.toString.endsWith(".txt")).toSeq.sorted.map(parse)
    }
}
