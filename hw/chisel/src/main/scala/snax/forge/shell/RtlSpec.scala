package snax.forge.shell

import java.nio.file.{Files, Path}

/** `rtl.json` as `Emit --spec` reads it (C§17, D130): concrete values only, resolved and checked by the Python side
  * (`snax_forge/gen/rtl.py`) against `hw/chisel/blocks.json`. Nothing is checked again here beyond what construction
  * needs; the blocks keep their `require`s.
  *
  * @param ports
  *   BRM port -> block port, in the BRM's port order (the streamer's reader and writer order)
  * @param rw
  *   the configuration fields in CSR order from offset 0, each a block pin and its width; start follows
  * @param tie
  *   block pins tied to a constant
  */
case class RtlUnit(
  name:   String,
  block:  String,
  params: Map[String, ujson.Value],
  ports:  Seq[(String, String)],
  rw:     Seq[(String, Int)],
  start:  String,
  tie:    Seq[(String, Int)]
)

case class RtlSpec(designPoint: String, tag: String, units: Seq[RtlUnit])

object RtlSpec {

  def parse(text: String): RtlSpec = {
    val d = ujson.read(text)
    RtlSpec(
      d("design_point").str,
      d("tag").str,
      d("units").arr.toSeq.map { u =>
        val c = u("csrs")
        RtlUnit(
          u("name").str,
          u("block").str,
          u("params").obj.toMap,
          u("ports").arr.toSeq.map(p => p("brm").str -> p("block").str),
          c("rw").arr.toSeq.map(r => r("pin").str -> r("width").num.toInt),
          c("start").str,
          c("tie").obj.toSeq.map { case (k, v) => k -> v.num.toInt }
        )
      }
    )
  }

  def load(path: Path): RtlSpec = parse(Files.readString(path))
}
