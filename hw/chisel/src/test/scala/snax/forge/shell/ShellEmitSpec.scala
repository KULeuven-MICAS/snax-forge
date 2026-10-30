package snax.forge.shell

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.Emit

/** Every fixture through `Emit --spec accelerator.json` once, and three checks on what it wrote.
  *
  * The ports: `<tag>_shell_wrapper` has exactly the ports `snax_acc_wrapper.sv.tpl` connects with
  * `snax_disable_csr_manager` (snax_cluster `hw/templates/`, at 05615e4), for the fixture's streamer hjson, and per
  * group `<group>_busy_o` and `<group>_done_o` (D146), which the template leaves unconnected. The template's ports,
  * read off its `snax_disable_csr_manager` branch: `clk_i`, `rst_ni`; per writer mover k (`data_writer_params`)
  * `acc2stream_<k>_data_o` of `num_channel[k] * tcdm_data_width` bits, `_valid_o`, `_ready_i`; per reader mover k
  * `stream2acc_<k>_data_i`, `_valid_i`, `_ready_o`; then `csr_req_addr_i`, `csr_req_data_i` (`RegAddrWidth`,
  * `RegDataWidth`: 32), `csr_req_write_i`, `csr_req_valid_i`, `csr_req_ready_o`, `csr_rsp_data_o` (32),
  * `csr_rsp_valid_o`, `csr_rsp_ready_i`. `tcdm_data_width` is the cluster's `data_width`, 64 on snax_forge. No fixture
  * has reader-writer movers. The wrapper's ports are read from the emitted SystemVerilog's module header (firtool's:
  * `input` or `output`, an optional `[n:0]`, a name; a line with a name only has the previous line's direction and
  * width).
  *
  * The hierarchy is `bundle.json`'s `modules` (C§18; snax-forge's `snax_forge/gen/bundle.py`, `module_tree`, which
  * writes it from `accelerator.json` without elaborating): `<tag>_shell_wrapper` holds `<tag>_csr`, `<tag>_ctrl` and
  * `<tag>_datapath`; `<tag>_csr` nothing; `<tag>_ctrl` a `<tag>_<group>_ctrl` per group; `<tag>_datapath` a
  * `<tag>_<unit>` per unit, then a `<tag>_<link>` per buffer, group by group; each `<tag>_<unit>` its one catalogue
  * datapath. Read off each module's instances in the SystemVerilog (firtool's ` <module> <instance> (` lines); dot 1's
  * (two groups), dot 4's (a wire) and dot 6's (a buffer) among them.
  *
  * `emitted.json` lists a file for every module the wrapper reaches, each file exists, and `sv/` holds those only.
  */
class ShellEmitSpec extends AnyFlatSpec {

  private val TcdmDataWidth = 64

  type Ports = Set[(String, String, Int)]

  def template(f: RtlFixtures.Fixture): Ports = {
    val clock   = Set(("clk_i", "input", 1), ("rst_ni", "input", 1))
    val writers = f.numCh("data_writer_params").zipWithIndex.flatMap { case (n, k) =>
      Seq(
        (s"acc2stream_${k}_data_o", "output", n * TcdmDataWidth),
        (s"acc2stream_${k}_valid_o", "output", 1),
        (s"acc2stream_${k}_ready_i", "input", 1)
      )
    }
    val readers = f.numCh("data_reader_params").zipWithIndex.flatMap { case (n, k) =>
      Seq(
        (s"stream2acc_${k}_data_i", "input", n * TcdmDataWidth),
        (s"stream2acc_${k}_valid_i", "input", 1),
        (s"stream2acc_${k}_ready_o", "output", 1)
      )
    }
    val csr     = Set(
      ("csr_req_addr_i", "input", 32),
      ("csr_req_data_i", "input", 32),
      ("csr_req_write_i", "input", 1),
      ("csr_req_valid_i", "input", 1),
      ("csr_req_ready_o", "output", 1),
      ("csr_rsp_data_o", "output", 32),
      ("csr_rsp_valid_o", "output", 1),
      ("csr_rsp_ready_i", "input", 1)
    )
    val pins = f.spec.groups.flatMap(g => Seq((s"${g.name}_busy_o", "output", 1), (s"${g.name}_done_o", "output", 1)))
    clock ++ writers ++ readers ++ csr ++ pins
  }

  private val Port = raw"\s*(input|output)?\s*(?:\[(\d+):0\])?\s*(\w+),?\s*".r

  /** The ports of `module`'s header in `file`. */
  def header(file: Path, module: String): Ports = {
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
        case line             => fail(s"$file: `$line` is not a port line")
      }
      .toSet
  }

  private val Instance = raw"  (\w+) \w+ \(".r

  /** The modules `module` instantiates, in order, read off `sv/<module>.sv`. */
  def children(sv: Path, module: String): Seq[String] =
    Files.readAllLines(sv.resolve(s"$module.sv")).asScala.toSeq.collect { case Instance(m) => m }

  /** `bundle.py`'s `module_tree`. */
  def tree(spec: AcceleratorSpec): Seq[(String, Seq[String])] = {
    val t = spec.tag
    Seq(
      s"${t}_csr"      -> Seq.empty,
      s"${t}_ctrl"     -> spec.groups.map(g => s"${t}_${g.name}_ctrl"),
      s"${t}_datapath" -> spec.groups.flatMap { g =>
        g.units.map(u => s"${t}_${u.name}") ++ g.links.filter(_.depth > 0).map(l => s"${t}_${l.name}")
      }
    )
  }

  behavior of "Emit --spec"

  for (f <- RtlFixtures.all)
    it should s"give ${f.name} the template's ports, bundle.json's hierarchy and its file list" in {
      val out  = Files.createTempDirectory(s"emit-${f.name}")
      val done = Emit.emitSpec(f.folder.resolve("accelerator.json"), out)
      val sv   = out.resolve("sv")
      assert(done.top == s"${f.spec.tag}_shell_wrapper")

      val got  = header(sv.resolve(s"${done.top}.sv"), done.top)
      val want = template(f)
      assert(
        got == want,
        s"\nonly the wrapper: ${(got -- want).toSeq.sorted}\nonly the template: ${(want -- got).toSeq.sorted}"
      )

      val modules = tree(f.spec)
      assert(children(sv, done.top) == modules.map(_._1), s"${done.top}")
      for ((m, kids) <- modules) assert(children(sv, m) == kids, s"$m")
      for (u <- f.spec.units) assert(children(sv, s"${f.spec.tag}_${u.name}").size == 1, s"${u.name}: one datapath")
      assert(modules.flatMap { case (m, k) => m +: k }.forall(done.modules.contains), s"emitted.json: ${done.modules}")

      def reach(m: String): Seq[String] = m +: children(sv, m).distinct.flatMap(reach)
      val listed = ujson.read(Files.readString(out.resolve("emitted.json")))("files").arr.map(_.str).toSet
      assert(listed.forall(p => Files.isRegularFile(out.resolve(p))), s"emitted.json: $listed")
      assert(reach(done.top).distinct.forall(m => listed(s"sv/$m.sv")), s"emitted.json: $listed")
      val svs    = Files.list(sv).iterator.asScala.map(p => s"sv/${p.getFileName}").filter(_.endsWith(".sv")).toSet
      assert(svs == listed, s"sv/ holds ${svs -- listed} besides emitted.json's files")
    }

  it should "have the fixtures of two groups, a wire and a buffer" in
    assert(Set("dot_1", "dot_4", "dot_6").subsetOf(RtlFixtures.all.map(_.name).toSet))
}
