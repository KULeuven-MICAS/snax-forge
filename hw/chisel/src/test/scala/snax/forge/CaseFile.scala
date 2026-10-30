package snax.forge

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters._

/** A snax-gvsoc case file, as its folder's `tb.cpp` reads it (`tests/snax/unit/`, `shell/`, `group/`: each folder's
  * README): header lines, a `cycles` line, one line per cycle; `#` starts a comment.
  */
object CaseFile {

  /** Its header lines and its cycle lines, comments and blank lines dropped. */
  def read(path: Path): (Seq[String], Seq[String]) = {
    val lines = Files.readAllLines(path).asScala.toSeq.map(_.takeWhile(_ != '#').trim).filter(_.nonEmpty)
    val at    = lines.indexOf("cycles")
    require(at >= 0, s"$path: no `cycles` line")
    (lines.take(at), lines.drop(at + 1))
  }

  /** A case's name: its file's, without `.txt`. */
  def name(path: Path): String = path.getFileName.toString.stripSuffix(".txt")

  /** Every `.txt` case of the folder the variable `env` names, sorted by file name, parsed; None when it is not set
    * (the submodule not checked out: a spec cancels). CI's chisel job sets each to the submodule's folder.
    */
  def all[T](env: String)(parse: Path => T): Option[Seq[T]] =
    sys.env.get(env).map { dir =>
      Files.list(Paths.get(dir)).iterator.asScala.filter(_.toString.endsWith(".txt")).toSeq.sorted.map(parse)
    }

  /** A number as `tb.cpp` reads it: decimal, or hex with `0x`. */
  def number(token: String): BigInt =
    if (token.startsWith("0x") || token.startsWith("0X")) BigInt(token.drop(2), 16) else BigInt(token)
}

/** An `expect run` line: the run line at the end of a run (snax-gvsoc's `docs/EVENTS.md`), its cycles per class. */
case class RunLine(firings: Int, busy: Int, stallIn: Int, stallOut: Int, flush: Int)

object RunLine {
  private val Keys = List("index", "firings", "busy", "stall_in", "stall_out", "flush")

  /** `index=I firings=F busy=B stall_in=S stall_out=O flush=X`, the fields after `expect run`: (I, the line). */
  object Fields {
    def unapply(fields: List[String]): Option[(Int, RunLine)] =
      if (fields.map(_.takeWhile(_ != '=')) != Keys) None
      else {
        val v = fields.map(_.dropWhile(_ != '=').drop(1).toInt)
        Some(v.head -> RunLine(v(1), v(2), v(3), v(4), v(5)))
      }
  }
}

/** The run lines a run of cycles gives: per busy cycle its class (a control unit's `cls`: busy 0, stall_in 1, stall_out
  * 2, flush 3) and whether the head fired; a line each time busy falls, and for a run still busy at the end.
  */
class RunLines {
  private val classes = Array(0, 0, 0, 0)
  private var firings = 0
  private var wasBusy = false
  private var lines   = Vector.empty[RunLine]

  def cycle(busy: Boolean, cls: => Int, fired: => Boolean): Unit = {
    if (wasBusy && !busy) close()
    wasBusy = busy
    if (busy) {
      classes(cls) += 1
      if (fired) firings += 1
    }
  }

  private def close(): Unit = {
    lines :+= RunLine(firings, classes(0), classes(1), classes(2), classes(3))
    java.util.Arrays.fill(classes, 0)
    firings = 0
  }

  /** Every run's line. */
  def done: Vector[RunLine] = {
    if (wasBusy) { close(); wasBusy = false }
    lines
  }
}
