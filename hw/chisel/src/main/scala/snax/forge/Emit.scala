package snax.forge

import chisel3.RawModule
import circt.stage.ChiselStage
import snax.forge.elementwise.{ElementwiseLoop, ElementwiseOp, ElementwiseSpatial, ElementwiseTiledSpatial}
import snax.forge.samples.{Accumulator, SimpleAdder}

/** Elaboration entry point: Chisel -> CHIRRTL -> firtool -> SystemVerilog.
  *
  * Run with `pixi run chisel-gen`. Output goes to `$SNAX_FORGE_HW_OUT` (`out/hw`, already covered by .gitignore) so
  * that elaborated RTL is never committed -- the Scala is the source of truth, exactly as the SDFG is the source of
  * truth on the Python side.
  *
  * ==Why emission lives here and not in each module==
  *
  * A `main` inside every module would work, and would spare this file a catalogue. It is the wrong shape for what this
  * becomes. This object is hand-written today and generated tomorrow: once the Accelerator Descriptor schema freezes in
  * W5, the emitter reads a descriptor and picks the module and its parameters from it. That mapping -- descriptor to
  * module to constructor arguments -- belongs in one generator, not scattered across the modules being generated. A
  * module that knows how to emit itself has to know something about the descriptor, and then there is no single place
  * left to change when the schema moves.
  *
  * The smaller reasons point the same way: one copy of the firtool flags rather than one per module, and one `runMain`
  * target rather than N.
  *
  * ==Usage==
  *
  * {{{
  * pixi run chisel-gen                        # every target
  * pixi run chisel-gen -- --list              # names only, no elaboration
  * pixi run chisel-gen -- ElementwiseLoop     # substring match, so both loop configs
  * pixi run chisel-gen -- --out /tmp/rtl Tiled
  * }}}
  */
object Emit {

  /** firtool flags shared by every SNAX-FORGE emission.
    *
    * Not private: HwGen elaborates through the same helpers. Two entry points
    * with two copies of the flag list would eventually produce RTL that
    * differs depending on which one built it.
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

  private val DataWidth = 32
  private val LoopWidth = 32

  /** Everything the emitter knows how to build.
    *
    * The elementwise entries deliberately come in selectable/fixed pairs. Diffing the two emitted files is the
    * clearest way to see what `supportedOps` actually does to the hardware: the operation mux and its decode vanish,
    * leaving the bare operator. That is the generator policy made visible.
    */
  private val catalogue: Seq[Target] = Seq(
    Target(
      "SimpleAdder",
      "sample: combinational lanes, fixed add",
      () => new SimpleAdder(width = DataWidth, lanes = 4)
    ),
    Target(
      "Accumulator",
      "sample: WCR-style accumulation, for contrast with the elementwise units",
      () => new Accumulator(width = DataWidth, accWidth = DataWidth)
    ),
    Target(
      "ElementwiseLoop-sel",
      "loop variant, W=1, all eight operations selectable",
      () => new ElementwiseLoop(dataWidth = DataWidth, loopCountWidth = LoopWidth)
    ),
    Target(
      "ElementwiseLoop-add",
      "loop variant, W=1, fixed add -- diff against -sel to see the mux disappear",
      () =>
        new ElementwiseLoop(
          dataWidth      = DataWidth,
          loopCountWidth = LoopWidth,
          supportedOps   = Seq(ElementwiseOp.Add)
        )
    ),
    Target(
      "ElementwiseSpatial-sel",
      "spatial variant, W=4, selectable",
      () => new ElementwiseSpatial(dataWidth = DataWidth, lanes = 4)
    ),
    Target(
      "ElementwiseSpatial-add",
      "spatial variant, W=4, fixed add",
      () => new ElementwiseSpatial(dataWidth = DataWidth, lanes = 4, supportedOps = Seq(ElementwiseOp.Add))
    ),
    Target(
      "ElementwiseTiledSpatial-sel",
      "tiled_spatial variant, W=4, selectable",
      () => new ElementwiseTiledSpatial(dataWidth = DataWidth, lanes = 4, loopCountWidth = LoopWidth)
    ),
    Target(
      "ElementwiseTiledSpatial-add",
      "tiled_spatial variant, W=4, fixed add",
      () =>
        new ElementwiseTiledSpatial(
          dataWidth      = DataWidth,
          lanes          = 4,
          loopCountWidth = LoopWidth,
          supportedOps   = Seq(ElementwiseOp.Add)
        )
    ),
    // The configuration the vecadd_tiled_spatial recipe actually describes:
    // lanes = 64 from the MapTiling step, int32 from the DaCe dtype, fixed add
    // from the tasklet. Emitted at full width on purpose -- 64 lanes is a big
    // file, and seeing how big is part of the point.
    Target(
      "vecadd-tiled-spatial",
      "matches transforms/vecadd_tiled_spatial.py: W=64, int32, fixed add",
      () =>
        new ElementwiseTiledSpatial(
          dataWidth      = DataWidth,
          lanes          = 64,
          loopCountWidth = LoopWidth,
          supportedOps   = Seq(ElementwiseOp.Add)
        )
    )
  )

  def main(args: Array[String]): Unit = {
    val (flags, selectors) = args.toSeq.partition(_.startsWith("--"))

    if (flags.contains("--list")) {
      println("[snax-forge] emittable targets:")
      catalogue.foreach(t => println(f"  ${t.label}%-30s ${t.note}"))
      return
    }

    val outDir = outDirFrom(args)

    // No selector means everything, which is what `pixi run chisel-gen` does
    // and what CI depends on. A selector is a case-insensitive substring, so
    // "Tiled" picks up every tiled configuration without exact spelling.
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

  /** `--out <dir>`, else `$SNAX_FORGE_HW_OUT`, else `out/hw` under the repo root.
    *
    * Anchored rather than written as ../../out/hw, which only happened to be
    * right because sbt starts in hw/chisel. A --out passed by the user is
    * resolved the same way, so `--out out/scratch` means what it looks like.
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
      args        = Array("--target-dir", outDir),
      firtoolOpts = firtoolOpts
    )
    name
  }
}