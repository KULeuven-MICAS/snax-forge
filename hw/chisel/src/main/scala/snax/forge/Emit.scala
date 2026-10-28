package snax.forge

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

import chisel3.RawModule
import circt.stage.ChiselStage
import snax.forge.ctrl.NestCtrl
import snax.forge.datapath.{ElementwiseLanes, TreeAccumulate}
import snax.forge.elementwise.ElementwiseOp
import snax.forge.shell.{AcceleratorSpec, ShellWrapper}
import snax.forge.stream.{FifoPacker, FifoSlicer}

/** Elaboration entry point: Chisel -> CHIRRTL -> firtool -> SystemVerilog.
  *
  * Run with `pixi run chisel-gen`. Output goes to `$SNAX_FORGE_HW_OUT` (`out/hw`, already covered by .gitignore) so
  * that elaborated RTL is never committed -- the Scala is the source of truth.
  *
  * ==Why emission lives here and not in each module==
  *
  * A `main` inside every module would work, and would spare this file a catalogue. One catalogue keeps the mapping from
  * a configuration to a module and its constructor arguments in one place, which is where a later generator (from a
  * BRM's hardware binding) plugs in. It also keeps one copy of the firtool flags rather than one per module, and one
  * `runMain` target rather than N.
  *
  * ==Usage==
  *
  * {{{
  * pixi run chisel-gen                        # every target
  * pixi run chisel-gen -- --list              # names only, no elaboration
  * pixi run chisel-gen -- NestCtrl            # substring match, so both controller configs
  * pixi run chisel-gen -- --out /tmp/rtl Lanes
  * pixi run chisel-gen -- --describe          # the datapath catalogue (Datapaths), as blocks.json holds it
  * pixi run chisel-gen -- --spec out/rtl/vecadd/accelerator.json   # the accelerator of an accelerator.json (gen-rtl)
  * }}}
  */
object Emit {

  /** firtool flags shared by every SNAX-FORGE emission.
    *
    * Not private: a later generator (from a BRM's hardware binding) elaborates through the same flags. Two entry points
    * with two copies of the flag list would eventually produce RTL that differs depending on which one built it.
    *
    *   - `-disable-all-randomization` removes the `RANDOMIZE_*` ifdef soup, so one .sv file is consumable by Verilator
    *     and by a synthesis flow without a per-tool define set.
    *   - `-strip-debug-info` drops Scala source locators. Without it the emitted SV churns on every unrelated edit
    *     above it in the file, which makes RTL diffs useless for review.
    */
  val firtoolOpts = Array(
    "-disable-all-randomization",
    "-strip-debug-info"
  )

  /** One emittable configuration.
    *
    * `gen` is a thunk rather than an instance because Chisel requires the module to be constructed *inside* the
    * elaboration context; a pre-built instance throws.
    *
    * `label` is how the target is selected on the command line. It is not the emitted module name -- that comes from
    * the module's own `desiredName`, which bakes in the parameters, so two configurations of one class land in two
    * files instead of overwriting each other.
    */
  private case class Target(label: String, note: String, gen: () => RawModule)

  private val Int32 = Seq("int32", "int32")

  /** Everything the emitter knows how to build on its own, apart from `--spec`: the catalogue's datapaths (D147), the
    * nest controller at the latencies they run at, and the parked stream blocks.
    */
  private val catalogue: Seq[Target] = Seq(
    Target(
      "ElementwiseLanes",
      "datapath: 4 lanes of int32, add (L 0, no pins)",
      () =>
        new ElementwiseLanes(inTypes = Int32, opType = "int32", outType = "int32", lanes = 4, op = ElementwiseOp.Add)
    ),
    Target(
      "TreeAccumulate",
      "datapath: 4 lanes of int32 into an int64 sum (L 1, pins fire and first)",
      () => new TreeAccumulate(inType = "int32", outType = "int64", lanes = 4)
    ),
    Target(
      "NestCtrl-l0",
      "the nest controller of an elementwise unit: L 0, II 1, two inputs",
      () => new NestCtrl(0, 1, 2)
    ),
    Target("NestCtrl-l1", "the nest controller of a fold: L 1, II 1, one input", () => new NestCtrl(1, 1, 1)),
    // Built and parked (D142): not in the datapath catalogue, emitted on their own.
    Target(
      "FifoSlicer",
      "parked: 2 words of int8 in, 4 lanes out per pop (r = 4)",
      () => new FifoSlicer(words = 2, laneType = "int8", outLanes = 4)
    ),
    Target(
      "FifoPacker",
      "parked: 4 lanes of int8 in per push, 2 words out (r = 4)",
      () => new FifoPacker(inLanes = 4, laneType = "int8", words = 2)
    )
  )

  def main(args: Array[String]): Unit = {
    val (flags, selectors) = args.toSeq.partition(_.startsWith("--"))

    if (flags.contains("--describe")) {
      println(describe)
      return
    }

    args.sliding(2).collectFirst { case Array("--spec", path) => path } match {
      case Some(path) =>
        val spec = RepoPaths.resolve(path)
        val dir  = args.sliding(2).collectFirst { case Array("--out", d) => RepoPaths.resolve(d) }
        val done = emitSpec(spec, dir.getOrElse(spec.toAbsolutePath.getParent))
        done.files.foreach(f => println(s"[snax-forge] wrote ${RepoPaths.display(f)}"))
        println(s"[snax-forge] ${done.top} from ${RepoPaths.display(spec)}: ${done.modules.size} module(s)")
        return
      case None       =>
    }

    if (flags.contains("--list")) {
      println("[snax-forge] emittable targets:")
      catalogue.foreach(t => println(f"  ${t.label}%-30s ${t.note}"))
      return
    }

    val outDir = outDirFrom(args)

    // No selector means everything, which is what `pixi run chisel-gen` does
    // and what CI depends on. A selector is a case-insensitive substring, so
    // "NestCtrl" picks up every controller configuration without exact spelling.
    val chosen =
      if (selectors.isEmpty) catalogue
      else catalogue.filter(t => selectors.exists(s => t.label.toLowerCase.contains(s.toLowerCase)))

    if (chosen.isEmpty) {
      Console.err.println(s"[snax-forge] no target matches ${selectors.mkString(", ")}")
      Console.err.println("[snax-forge] run with --list to see the available targets")
      sys.exit(1)
    }

    chosen.foreach { target =>
      val module = emit(target.gen(), outDir)
      println(f"[snax-forge] ${target.label}%-30s -> $module%s.sv")
    }
    println(s"[snax-forge] emitted ${chosen.size} module(s) to $outDir")
  }

  /** What `emitSpec` wrote: the top module, every module, and the files (those in `sv/`, then `emitted.json`). */
  case class Emitted(top: String, modules: Seq[String], files: Seq[Path])

  /** `--spec <accelerator.json> [--out DIR]`: the accelerator of `accelerator.json` (C§17, D145), `<tag>_shell_wrapper`
    * and what it holds, one SystemVerilog file per module in `DIR/sv/` (default `DIR` the folder of the file), and
    * `DIR/emitted.json`: `rtl` (the file it was built from), `top`, `modules` and `files` (paths from `DIR`). The files
    * a previous `emitted.json` lists are removed first, so `sv/` holds this spec's files only.
    */
  def emitSpec(specPath: Path, dir: Path): Emitted = {
    val spec    = AcceleratorSpec.load(specPath)
    val sv      = dir.resolve("sv")
    val listing = dir.resolve("emitted.json")
    if (Files.exists(listing))
      ujson.read(Files.readString(listing))("files").arr.foreach(f => Files.deleteIfExists(dir.resolve(f.str)))
    var top     = "<unknown>"
    ChiselStage.emitSystemVerilogFile(
      {
        val m = new ShellWrapper(spec)
        top = m.name
        m
      },
      args = Array("--target-dir", sv.toString, "--split-verilog"),
      firtoolOpts = firtoolOpts
    )
    val files   = Files.list(sv).iterator.asScala.filter(_.toString.endsWith(".sv")).toSeq.sortBy(_.toString)
    val modules = files.flatMap(f => Files.readAllLines(f).asScala.collect { case ModuleLine(m) => m })
    Files.writeString(
      listing,
      ujson.write(
        ujson.Obj(
          "rtl"     -> RepoPaths.display(specPath),
          "top"     -> top,
          "modules" -> ujson.Arr.from(modules),
          "files"   -> ujson.Arr.from(files.map(f => dir.relativize(f).toString))
        ),
        indent = 1
      ) + "\n"
    )
    Emitted(top, modules, files :+ listing)
  }

  private val ModuleLine = raw"module (\w+)\b.*".r

  /** `--describe`: the datapath catalogue from the registry (`Datapaths`), as `hw/chisel/blocks.json` holds it (D130).
    */
  def describe: String = ujson.write(Datapaths.describe, indent = 1)

  /** `--out <dir>`, else `$SNAX_FORGE_HW_OUT`, else `out/hw` under the repo root.
    *
    * Anchored rather than written as ../../out/hw, which only happened to be right because sbt starts in hw/chisel. A
    * --out passed by the user is resolved the same way, so `--out out/scratch` means what it looks like.
    */
  def outDirFrom(args: Array[String]): String = {
    val flagged = args.sliding(2).collectFirst { case Array("--out", dir) => dir }
    flagged
      .orElse(sys.env.get("SNAX_FORGE_HW_OUT"))
      .map(RepoPaths.resolve)
      .getOrElse(RepoPaths.out.resolve("hw"))
      .toString
  }

  /** Elaborate one module and return the emitted top-level name.
    *
    * `gen` is by-name for the same reason `Target.gen` is a thunk: construction has to happen inside the elaboration
    * context. The returned name is the module's `desiredName`, which is also the .sv filename ChiselStage writes.
    */
  def emit(gen: => RawModule, outDir: String): String = {
    var name = "<unknown>"
    ChiselStage.emitSystemVerilogFile(
      {
        val m = gen
        name = m.name
        m
      },
      args = Array("--target-dir", outDir),
      firtoolOpts = firtoolOpts
    )
    name
  }
}
