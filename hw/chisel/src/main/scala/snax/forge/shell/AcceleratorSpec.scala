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

  /** A unit's BRM port: the datapath's port, its direction (`in` or `out`), its dtype (NumPy's name), its lanes. */
  case class PortSpec(port: String, dir: String, dtype: String, lanes: Int)

  /** A link: from a unit's output BRM port to another unit's input, `(unit, port)` each; depth 0 a wire, >= 1 a buffer
    * of that many beats (`LinkBuffer`, `<tag>_<name>`). The group's graph is checked by `ctrl.GroupDesc`.
    */
  case class LinkSpec(name: String, from: (String, String), to: (String, String), depth: Int)

  case class UnitSpec(
    name:     String,
    brm:      String,
    datapath: String,
    ports:    Seq[(String, PortSpec)],
    opType:   Option[String],
    params:   Map[String, ujson.Value],
    latency:  Int,
    ii:       Int,
    span:     Int,
    csrs:     Seq[CsrSpec]
  ) {
    def port(brm: String): PortSpec = ports.find(_._1 == brm).get._2

    /** Its input (output) BRM ports in port order, outer or linked. */
    def inPorts:  Seq[String] = ports.collect { case (b, p) if p.dir == "in" => b }
    def outPorts: Seq[String] = ports.collect { case (b, p) if p.dir == "out" => b }

    /** Its T when fixed at design time: `params.T`, or 1 for a unit without a JOB_BEATS (no fold); None when the start
      * carries it (JOB_BEATS).
      */
    def fixedT: Option[Int] =
      if (csrs.exists(_.name == "JOB_BEATS")) None
      else Some(params.get("T").map(_.num.toInt).getOrElse(1))
  }

  case class GroupSpec(name: String, csrs: Seq[CsrSpec], units: Seq[UnitSpec], links: Seq[LinkSpec]) {
    private def own(n: String): CsrSpec = csrs.find(_.name == n).get

    /** Its CSRs in CSR order: START, JOBS, its units' read-write ones, STATUS, COUNTER. */
    def window: Seq[CsrSpec] =
      Seq(own("START"), own("JOBS")) ++ units.flatMap(_.csrs) ++ Seq(own("STATUS"), own("COUNTER"))

    def csr(n: String): Option[CsrSpec] = window.find(_.name == n)

    /** Each JOB_BEATS in unit order: the start carries them after JOBS. */
    def jobBeats: Seq[CsrSpec] = units.flatMap(_.csrs.filter(_.name == "JOB_BEATS"))
  }

  def parse(text: String): AcceleratorSpec = {
    val d                   = ujson.read(text)
    def csr(c: ujson.Value) =
      CsrSpec(c("name").str, c("access").str, c("offset").num.toInt, c.obj.get("width").map(_.num.toInt))
    def end(e: String)      = {
      val Array(u, p) = e.split("\\.", 2)
      u -> p
    }
    def outer(key: String) = d(key).arr.toSeq.map(e => end(e.str))
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
                k -> PortSpec(p("port").str, p("dir").str, p("dtype").str, p("lanes").num.toInt)
              },
              u.obj.get("opType").map(_.str),
              u("params").obj.toMap,
              u("timing")("latency").num.toInt,
              u("timing")("ii").num.toInt,
              u("timing")("span").num.toInt,
              u("csrs").arr.toSeq.map(csr)
            )
          },
          g("links").arr.toSeq.map { l =>
            LinkSpec(l("name").str, end(l("from").str), end(l("to").str), l("depth").num.toInt)
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
