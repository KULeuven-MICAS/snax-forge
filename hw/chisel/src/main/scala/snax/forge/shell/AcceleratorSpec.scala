package snax.forge.shell

import java.nio.file.{Files, Path}

/** `accelerator.json` as `Emit --spec` reads it (C§17, D145, D157): the one description of a design point's
  * accelerators, written by snax-forge (`snax_forge/gen/accelerator.py`) and checked by it against the datapath
  * catalogue (`hw/chisel/blocks.json`, D130). The file nests accelerators ⊃ groups ⊃ units; the shell is one for the
  * whole design, so the reader flattens it once: the groups in CSR order across the accelerators, the outer inputs and
  * outputs set after set. Nothing is checked again here beyond what construction needs: the offsets as stored,
  * contiguous in CSR order from the shell's first CSR.
  *
  * A set whose groups share its movers (`share_ports`, D169) has `movers` and `port_mux`: its outer ports at the
  * wrapper are its movers, readers then writers, each `num_channel * 64` bits, in place of its groups' inputs and
  * outputs, which reach the movers through `<tag>_<set>_port_mux` (`PortMux`); its PORT_MUX_SEL is its last CSR.
  */
object AcceleratorSpec {

  /** A CSR: its offset from the shell's first CSR and its width, 1 to 32 bits (D152: START 1, JOBS and JOB_BEATS
    * job_width, a scalar its dtype's bits, 32 a word at 64, STATUS 2, BUSY_COUNTER busy_counter_width); a read gives
    * its bits, the upper ones 0. `pin`: a scalar's datapath pin (`k1`), which the CSR drives (two at 64 bits, low word
    * first).
    */
  case class CsrSpec(name: String, access: String, offset: Int, width: Int, pin: Option[String] = None) {
    require(1 <= width && width <= 32, s"CSR $name: width $width, must be 1 to 32")
  }

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
    csrs:     Seq[CsrSpec],
    scalars:  Seq[(String, String)] = Seq.empty
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

    /** A scalar's CSRs, the pin's low word first. */
    def pinned(pin: String): Seq[CsrSpec] = csrs.filter(_.pin.contains(pin))
  }

  case class GroupSpec(name: String, csrs: Seq[CsrSpec], units: Seq[UnitSpec], links: Seq[LinkSpec]) {
    private def own(n: String): CsrSpec = csrs.find(_.name == n).get

    /** Its CSRs in CSR order: START, JOBS, its units' read-write ones, STATUS, BUSY_COUNTER. */
    def window: Seq[CsrSpec] =
      Seq(own("START"), own("JOBS")) ++ units.flatMap(_.csrs) ++ Seq(own("STATUS"), own("BUSY_COUNTER"))

    def csr(n: String): Option[CsrSpec] = window.find(_.name == n)

    /** Each JOB_BEATS in unit order: the control unit's `settings` after JOBS. */
    def jobBeats: Seq[CsrSpec] = units.flatMap(_.csrs.filter(_.name == "JOB_BEATS"))
  }

  /** A port of a group on a shared mover: the group, the port `(unit, port)`, its channels (at most the mover's). */
  case class MuxUser(group: String, port: (String, String), numChannel: Int)

  /** A mover of a shared set (D169): `reader<k>` or `writer<k>`, its channels, whether it has `configurable_channel`
    * (`masked`), its users, at most one per group.
    */
  case class MoverSpec(name: String, role: String, numChannel: Int, masked: Boolean, users: Seq[MuxUser]) {
    require(role == "read" || role == "write", s"mover $name: role $role")
    def reader: Boolean = role == "read"
    def width:  Int     = numChannel * 64
  }

  /** A shared set's port mux: its groups in PORT_MUX_SEL's order and PORT_MUX_SEL, the set's last CSR. */
  case class PortMuxSpec(groups: Seq[String], csr: CsrSpec)

  /** A streamer set, one of the file's accelerators (D157): its name, its outer ports, its groups by name, and, shared,
    * its movers and port mux.
    */
  case class SetSpec(
    name:    String,
    inputs:  Seq[(String, String)],
    outputs: Seq[(String, String)],
    groups:  Seq[String],
    movers:  Seq[MoverSpec]      = Seq.empty,
    portMux: Option[PortMuxSpec] = None
  ) {
    def shared:  Boolean        = portMux.nonEmpty
    def readers: Seq[MoverSpec] = movers.filter(_.reader)
    def writers: Seq[MoverSpec] = movers.filterNot(_.reader)
  }

  def parse(text: String): AcceleratorSpec = {
    val d                   = ujson.read(text)
    def csr(c: ujson.Value) =
      CsrSpec(c("name").str, c("access").str, c("offset").num.toInt, c("width").num.toInt, c.obj.get("pin").map(_.str))
    def end(e: String)      = {
      val Array(u, p) = e.split("\\.", 2)
      u -> p
    }
    val accs                = d("accelerators").arr.toSeq
    def outer(key: String) = accs.flatMap(a => a(key).arr.toSeq.map(e => end(e.str)))
    val spec    = AcceleratorSpec(
      d("design_point").str,
      d("tag").str,
      outer("inputs"),
      outer("outputs"),
      accs.flatMap(_("groups").arr.toSeq).map { g =>
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
              u("csrs").arr.toSeq.map(csr),
              u.obj.get("scalars").map(_.obj.toSeq.map { case (k, v) => k -> v.str }).getOrElse(Seq.empty)
            )
          },
          g("links").arr.toSeq.map { l =>
            LinkSpec(l("name").str, end(l("from").str), end(l("to").str), l("depth").num.toInt)
          }
        )
      },
      accs.map { a =>
        val movers = a.obj.get("movers").map(_.arr.toSeq).getOrElse(Seq.empty).map { m =>
          MoverSpec(
            m("name").str,
            m("role").str,
            m("num_channel").num.toInt,
            m("masked").bool,
            m("users").arr.toSeq.map(u => MuxUser(u("group").str, end(u("port").str), u("num_channel").num.toInt))
          )
        }
        val mux    = a.obj.get("port_mux").map(x => PortMuxSpec(x("groups").arr.toSeq.map(_.str), csr(x("csr"))))
        require(movers.isEmpty == mux.isEmpty, s"set ${a("name").str}: movers and port_mux come together (D169)")
        SetSpec(
          a("name").str,
          a("inputs").arr.toSeq.map(e => end(e.str)),
          a("outputs").arr.toSeq.map(e => end(e.str)),
          a("groups").arr.toSeq.map(_("name").str),
          movers,
          mux
        )
      }
    )
    val offsets = spec.csrs.map(_.offset)
    require(offsets == offsets.indices, s"CSR offsets $offsets in CSR order are not contiguous from 0")
    spec.jobWidth
    spec.busyCounterWidth
    spec
  }

  def load(path: Path): AcceleratorSpec = parse(Files.readString(path))
}

/** The design's accelerators, flattened (the companion's doc): every group in CSR order, every outer port set after
  * set.
  */
case class AcceleratorSpec(
  designPoint: String,
  tag:         String,
  inputs:      Seq[(String, String)],
  outputs:     Seq[(String, String)],
  groups:      Seq[AcceleratorSpec.GroupSpec],
  sets:        Seq[AcceleratorSpec.SetSpec] = Seq.empty
) {

  /** The shared sets' PORT_MUX_SEL CSRs, in set order. */
  def portMuxCsrs: Seq[AcceleratorSpec.CsrSpec] = sets.flatMap(_.portMux.map(_.csr))

  /** Every CSR in CSR order: the groups' windows and each shared set's PORT_MUX_SEL after its last group's. */
  def csrs: Seq[AcceleratorSpec.CsrSpec] = (groups.flatMap(_.window) ++ portMuxCsrs).sortBy(_.offset)

  private def width(names: String*): Int = {
    val ws = csrs.filter(c => names.contains(c.name)).map(_.width).distinct
    require(ws.size == 1, s"${names.mkString(" and ")}: widths $ws, one width for all (D152)")
    ws.head
  }

  /** job_width (D152): the width of JOBS and every JOB_BEATS, one for all, and of the control units' counters. */
  lazy val jobWidth: Int = width("JOBS", "JOB_BEATS")

  /** busy_counter_width: every BUSY_COUNTER's width. */
  lazy val busyCounterWidth: Int = width("BUSY_COUNTER")

  /** The widest CSR a read can give: the width of `<tag>_csr`'s kept answer. */
  def readWidth: Int = csrs.map(_.width).max

  def units: Seq[AcceleratorSpec.UnitSpec] = groups.flatMap(_.units)

  def unit(name: String): AcceleratorSpec.UnitSpec = units.find(_.name == name).get

  /** The unit's input (or output) BRM ports in the streamer's order, each with its index among the outer ports. */
  def ins(unit:  String): Seq[(Int, String)] = inputs.zipWithIndex.collect { case ((u, p), k) if u == unit => k -> p }
  def outs(unit: String): Seq[(Int, String)] = outputs.zipWithIndex.collect { case ((u, p), k) if u == unit => k -> p }

  /** The set that holds group `g`'s ports, when it shares its movers. */
  def sharedSet(g: String): Option[AcceleratorSpec.SetSpec] = sets.find(s => s.shared && s.groups.contains(g))
}
