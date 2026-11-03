package snax.forge

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

/** A module's ports read off its emitted SystemVerilog header, firtool's: `input` or `output`, an optional `[n:0]`, a
  * name; a line with a name only has the previous line's direction and width.
  */
object SvHeader {

  type Ports = Set[(String, String, Int)]

  private val Port = raw"\s*(input|output)?\s*(?:\[(\d+):0\])?\s*(\w+),?\s*".r

  /** The ports (name, direction, width) of `module`'s header in `file`. */
  def ports(file: Path, module: String): Ports = {
    val lines = Files.readAllLines(file).asScala.dropWhile(!_.startsWith(s"module $module(")).drop(1)
    var dir   = ""
    var width = 1
    lines
      .takeWhile(!_.trim.startsWith(");"))
      .map {
        case Port(d, w, name) =>
          if (d != null) { dir = d; width = 1 }
          if (w != null) width = w.toInt + 1
          (name, dir, width)
        case line             => throw new AssertionError(s"$file: `$line` is not a port line")
      }
      .toSet
  }
}
