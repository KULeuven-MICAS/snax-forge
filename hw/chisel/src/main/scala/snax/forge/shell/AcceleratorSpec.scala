package snax.forge.shell

import java.nio.file.{Files, Path}

/** `accelerator.json` as `Emit --spec` reads it (C§17, D145): the one description of a design point's accelerator,
  * written by snax-forge (`snax_forge/gen/accelerator.py`) and checked by it against the datapath catalogue
  * (`hw/chisel/blocks.json`, D130). Nothing is checked again here beyond what construction needs: the offsets as
  * stored, contiguous in CSR order.
  */
object AcceleratorSpec {

  /** A CSR: its offset from the accelerator's first CSR; `width` for a read-write one. */
  case class CsrSpec(name: String, access: String, offset: Int, width: Option[Int])

  /** A unit's BRM port: the datapath's port, its dtype (NumPy's name), its lanes. */
  case class PortSpec(port: String, dtype: String, lanes: Int)

  case class UnitSpec(
    name:     String,
    brm:      String,
    datapath: String,
    ports:    Seq[(String, PortSpec)],
    opType:   Option[String],
    params:   Map[String, ujson.Value],
    latency:  Int,
    ii:       Int,
    csrs:     Seq[CsrSpec]
  ) {
    def port(brm: String): PortSpec = ports.find(_._1 == brm).get._2
  }

  case class GroupSpec(name: String, csrs: Seq[CsrSpec], units: Seq[UnitSpec]) {
    private def own(n: String): CsrSpec = csrs.find(_.name == n).get

    /** Its CSRs in CSR order: START, JOBS, its units' read-write ones, STATUS, COUNTER. */
    def window: Seq[CsrSpec] =
      Seq(own("START"), own("JOBS")) ++ units.flatMap(_.csrs) ++ Seq(own("STATUS"), own("COUNTER"))

    def csr(n: String): Option[CsrSpec] = window.find(_.name == n)
  }

  def parse(text: String): AcceleratorSpec = {
    val d                   = ujson.read(text)
    def csr(c: ujson.Value) =
      CsrSpec(c("name").str, c("access").str, c("offset").num.toInt, c.obj.get("width").map(_.num.toInt))
    def outer(key: String)  =
      d(key).arr.toSeq.map { e =>
        val Array(u, p) = e.str.split("\\.", 2)
        u -> p
      }
    val spec                = AcceleratorSpec(
      d("name").str,
      d("design_point").str,
      d("tag").str,
      outer("inputs"),
      outer("outputs"),
      d("groups").arr.toSeq.map { g =>
        GroupSpec(
          g("name").str,
          g("csrs").arr.toSeq.map(csr),
          g("units").arr.toSeq.map { u =>
            UnitSpec(
              u("name").str,
              u("brm").str,
              u("datapath").str,
              u("ports").obj.toSeq.map { case (k, p) =>
                k -> PortSpec(p("port").str, p("dtype").str, p("lanes").num.toInt)
              },
              u.obj.get("opType").map(_.str),
              u("params").obj.toMap,
              u("timing")("latency").num.toInt,
              u("timing")("ii").num.toInt,
              u("csrs").arr.toSeq.map(csr)
            )
          }
        )
      }
    )
    val offsets             = spec.csrs.map(_.offset)
    require(offsets == offsets.indices, s"CSR offsets $offsets in CSR order are not contiguous from 0")
    spec
  }

  def load(path: Path): AcceleratorSpec = parse(Files.readString(path))
}

case class AcceleratorSpec(
  name:        String,
  designPoint: String,
  tag:         String,
  inputs:      Seq[(String, String)],
  outputs:     Seq[(String, String)],
  groups:      Seq[AcceleratorSpec.GroupSpec]
) {

  /** Every CSR in CSR order. */
  def csrs: Seq[AcceleratorSpec.CsrSpec] = groups.flatMap(_.window)

  def units: Seq[AcceleratorSpec.UnitSpec] = groups.flatMap(_.units)

  def unit(name: String): AcceleratorSpec.UnitSpec = units.find(_.name == name).get

  /** The unit's input (or output) BRM ports in the streamer's order, each with its index among the outer ports. */
  def ins(unit:  String): Seq[(Int, String)] = inputs.zipWithIndex.collect { case ((u, p), k) if u == unit => k -> p }
  def outs(unit: String): Seq[(Int, String)] = outputs.zipWithIndex.collect { case ((u, p), k) if u == unit => k -> p }
}
