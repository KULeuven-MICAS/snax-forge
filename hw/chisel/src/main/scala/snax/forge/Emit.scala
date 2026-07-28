package snax.forge

import chisel3.RawModule
import circt.stage.ChiselStage
import snax.forge.samples.{Accumulator, SimpleAdder}   // ← new

/** Elaboration entry point: Chisel -> CHIRRTL -> firtool -> SystemVerilog.
  *
  * Run with `pixi run chisel-gen`. Output goes to `$SNAX_FORGE_HW_OUT`
  * (`out/hw`, already covered by .gitignore) so that elaborated RTL is never
  * committed -- the Scala is the source of truth, exactly as the SDFG is the
  * source of truth on the Python side.
  *
  * This object is hand-written today and generated tomorrow. Once the
  * Accelerator Descriptor schema freezes in W5, the emitter will read a
  * descriptor and choose the module and its parameters from it; the shape of
  * this file is what that generated code should look like.
  */
object Emit {

  /** firtool flags shared by every SNAX-FORGE emission.
    *
    *   - `-disable-all-randomization` removes the `RANDOMIZE_*` ifdef soup, so
    *     one .sv file is consumable by Verilator and by a synthesis flow
    *     without a per-tool define set.
    *   - `-strip-debug-info` drops Scala source locators. Without it the
    *     emitted SV churns on every unrelated edit above it in the file, which
    *     makes RTL diffs useless for review.
    */
  private val firtoolOpts = Array(
    "-disable-all-randomization",
    "-strip-debug-info"
  )

  def main(args: Array[String]): Unit = {
    val outDir = args.headOption
      .orElse(sys.env.get("SNAX_FORGE_HW_OUT"))
      .getOrElse("../../out/hw")

    emit(new SimpleAdder(width = 32, lanes = 4), outDir)
    emit(new Accumulator(width = 32, accWidth = 32), outDir)

    println(s"[snax-forge] emitted SystemVerilog to $outDir")
  }

  /** `gen` is by-name: Chisel requires the module to be constructed *inside*
    * the elaboration context, so passing an already-built instance throws.
    */
  private def emit(gen: => RawModule, outDir: String): Unit =
    ChiselStage.emitSystemVerilogFile(
      gen,
      args = Array("--target-dir", outDir),
      firtoolOpts = firtoolOpts
    )
}