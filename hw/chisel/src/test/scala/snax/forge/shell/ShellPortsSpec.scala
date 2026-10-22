package snax.forge.shell

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.Emit

/** Every fixture elaborates through `Emit --spec` into `<tag>_shell_wrapper` with exactly the ports
  * `snax_acc_wrapper.sv.tpl` connects with `snax_disable_csr_manager` (snax_cluster `hw/templates/`, at 05615e4), for
  * the fixture's streamer hjson; and `emitted.json` lists files that exist, the wrapper's among them.
  *
  * The template's ports, read off its `snax_disable_csr_manager` branch: `clk_i`, `rst_ni`; per writer mover k
  * (`data_writer_params`) `acc2stream_<k>_data_o` of `num_channel[k] * tcdm_data_width` bits, `_valid_o`, `_ready_i`;
  * per reader mover k `stream2acc_<k>_data_i`, `_valid_i`, `_ready_o`; then `csr_req_addr_i`, `csr_req_data_i`
  * (`RegAddrWidth`, `RegDataWidth`: 32), `csr_req_write_i`, `csr_req_valid_i`, `csr_req_ready_o`, `csr_rsp_data_o`
  * (32), `csr_rsp_valid_o`, `csr_rsp_ready_i`. `tcdm_data_width` is the cluster's `data_width`, 64 on snax_forge. No
  * fixture has reader-writer movers.
  *
  * The wrapper's ports are read from the emitted SystemVerilog's module header (firtool's: `input` or `output`, an
  * optional `[n:0]`, a name; a line with a name only has the previous line's direction and width).
  */
class ShellPortsSpec extends AnyFlatSpec {

  private val TcdmDataWidth = 64

  type Ports = Set[(String, String, Int)]

  def template(f: RtlFixtures.Fixture): Ports = {
    val clock   = Set(("clk_i", "input", 1), ("rst_ni", "input", 1))
    val writers = f.channels("data_writer_params").zipWithIndex.flatMap { case (n, k) =>
      Seq(
        (s"acc2stream_${k}_data_o", "output", n * TcdmDataWidth),
        (s"acc2stream_${k}_valid_o", "output", 1),
        (s"acc2stream_${k}_ready_i", "input", 1)
      )
    }
    val readers = f.channels("data_reader_params").zipWithIndex.flatMap { case (n, k) =>
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
    clock ++ writers ++ readers ++ csr
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

  behavior of "Emit --spec"

  for (f <- RtlFixtures.all) {
    it should s"give ${f.name} the template's ports" in {
      val out    = Files.createTempDirectory(s"emit-${f.name}")
      val done   = Emit.emitSpec(f.folder.resolve("rtl.json"), out)
      assert(done.top == s"${f.spec.tag}_shell_wrapper")
      val listed = ujson.read(Files.readString(out.resolve("emitted.json")))("files").arr.map(_.str)
      assert(listed.nonEmpty && listed.forall(p => Files.isRegularFile(out.resolve(p))), s"emitted.json: $listed")
      val got  = header(out.resolve(s"sv/${done.top}.sv"), done.top)
      val want = template(f)
      assert(
        got == want,
        s"\nonly the wrapper: ${(got -- want).toSeq.sorted}\nonly the template: ${(want -- got).toSeq.sorted}"
      )
    }
  }
}
