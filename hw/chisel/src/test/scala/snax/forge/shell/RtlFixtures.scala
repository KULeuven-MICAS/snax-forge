package snax.forge.shell

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

import snax.forge.RepoPaths

/** The fixtures of the shell specs, `hw/chisel/src/test/resources/rtl/<design>/`, written by the Python side
  * (`snax_forge/gen/rtl.py`'s `write_fixture`, checked byte for byte by `tests/gen/test_rtl.py`): `accelerator.json`
  * (the same bytes gen-gvsoc writes), the design's streamer hjson (`<tag>_streamer.hjson`) and, for a design of one
  * accelerator task, `vectors.json` (`driven`; dot_1's two groups have none: `TwoGroupsSpec` drives each itself).
  */
object RtlFixtures {

  val dir: Path = RepoPaths.root.resolve("hw/chisel/src/test/resources/rtl")

  case class Port(port: String, array: String, lanes: Int, beats: Int, values: Seq[BigInt])
  case class Task(name: String, csrs: Seq[(String, Long)], firings: Int, counter: Long)

  case class Fixture(name: String, folder: Path) {
    lazy val spec: AcceleratorSpec = AcceleratorSpec.load(folder.resolve("accelerator.json"))

    /** The streamer hjson: `//` comment lines, then JSON. */
    lazy val streamer: ujson.Value = ujson.read(
      Files
        .readAllLines(folder.resolve(s"${spec.tag}_streamer.hjson"))
        .asScala
        .filterNot(_.startsWith("//"))
        .mkString("\n")
    )

    lazy val vectors: ujson.Value = ujson.read(Files.readString(folder.resolve("vectors.json")))

    lazy val ports: Seq[Port] = vectors("ports").arr.toSeq.map { p =>
      Port(
        p("port").str,
        p("array").str,
        p("lanes").num.toInt,
        p("beats").num.toInt,
        p("values").arr.toSeq.map(v => BigInt(v.num.toLong))
      )
    }

    lazy val tasks: Seq[Task] = vectors("tasks").arr.toSeq.map { t =>
      Task(
        t("name").str,
        t("csrs").obj.toSeq.map { case (k, v) => k -> v.num.toLong },
        t("firings").num.toInt,
        t("counter").num.toLong
      )
    }

    /** Each mover's num_ch (the hjson's `num_channel`) of a group, `data_reader_params` or `data_writer_params`. */
    def numCh(group: String): Seq[Int] =
      streamer.obj.get(group).map(_("num_channel").arr.toSeq.map(_.num.toInt)).getOrElse(Seq.empty)
  }

  def all: Seq[Fixture] =
    Files.list(dir).iterator.asScala.filter(Files.isDirectory(_)).toSeq.sortBy(_.toString).map { f =>
      Fixture(f.getFileName.toString, f)
    }

  /** The fixtures with `vectors.json`: one group, one task. */
  def driven: Seq[Fixture] = all.filter(f => Files.isRegularFile(f.folder.resolve("vectors.json")))
}
