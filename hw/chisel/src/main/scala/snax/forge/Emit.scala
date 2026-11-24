package snax.forge

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

import chisel3.{RawModule, RequireAsyncReset}
import circt.stage.ChiselStage
import snax.forge.datapath.DatapathModule
import snax.forge.shell.{AcceleratorSpec, ShellWrapper}
import snax.forge.stream.{FifoPacker, FifoSlicer}

/** Elaboration entry point: Chisel -> CHIRRTL -> firtool -> SystemVerilog.
  *
  * Run with `pixi run chisel-gen`. Output goes to `$SNAX_FORGE_HW_OUT` (`out/hw`, already covered by .gitignore) so
  * that elaborated RTL is never committed -- the Scala is the source of truth.
  *
  * ==Why emission lives here and not in each module==
  *
  * A `main` inside every module would work, and would spare this file a catalogue. One entry point keeps one copy of
  * the firtool flags rather than one per module, one `runMain` target rather than N, and builds both the catalogue
  * (from the `Datapaths` registry) and a design point's accelerator (`--spec`) through the same flags.
  *
  * ==Usage==
  *
  * {{{
  * pixi run chisel-gen                        # every target
  * pixi run chisel-gen -- --list              # names only, no elaboration
  * pixi run chisel-gen -- Fifo                # substring match, so both stream blocks
  * pixi run chisel-gen -- --out /tmp/rtl Lanes
  * pixi run chisel-gen -- --describe          # the datapath catalogue (Datapaths), as blocks.json holds it
  * pixi run chisel-gen -- --spec out/rtl/vecadd/accelerator.json   # the accelerator of an accelerator.json (gen-rtl)
  * pixi run chisel-gen -- --spec A.json --out DIR_A --spec B.json --out DIR_B   # several, in one call
  * pixi run chisel-gen -- --block TreeAccumulate --params '{"inType":"int64","outType":"int64","lanes":8,"op":"add"}' --out DIR
  * pixi run chisel-gen -- --block FifoPacker --params '{"inLanes":1,"laneType":"int32","words":1}' --out DIR
  * pixi run chisel-gen -- --block FifoSlicer --params '{"words":1,"laneType":"int32","outLanes":1}' --out DIR
  * }}}
  *
  * `--block NAME --params JSON --out DIR` repeats like `--spec` (gen-block); `--spec` and `--block` go in one call.
  * `NAME` is a catalogue datapath or a stream block of `streamBlocks` (FifoPacker, FifoSlicer).
  */
object Emit {

  /** firtool flags shared by every SNAX-FORGE emission.
    *
    * One list for the catalogue and `--spec`: two copies would eventually produce RTL that differs depending on which
    * one built it.
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

  /** Everything the emitter knows how to build on its own, apart from `--spec` and `--block`: each catalogue datapath
    * (`Datapaths.all`) at its `example` params, and the stream blocks.
    */
  private val catalogue: Seq[Target] =
    Datapaths.all.map { d =>
      Target(d.name, s"datapath: ${ujson.write(ujson.Obj.from(d.example))}", () => d.make(d.example))
    } ++ Seq(
      // Not in the datapath catalogue, emitted on their own: the slicer and the packer, placed by ShellWrapper on a
      // sliced input and a packed output, and emittable alone at any params with --block
      Target(
        "FifoSlicer",
        "stream block: 2 words of int8 in, 4 lanes out per pop (r = 4)",
        () => new FifoSlicer(words = 2, laneType = "int8", outLanes = 4)
      ),
      Target(
        "FifoPacker",
        "stream block: 4 lanes of int8 in per push, 2 words out (r = 4)",
        () => new FifoPacker(inLanes = 4, laneType = "int8", words = 2)
      )
    )

  /** The command line, read once: its flags, the catalogue's `--out DIR`, each `--spec FILE` with the `--out DIR` that
    * follows it (none: the file's folder), each `--block NAME --params JSON --out DIR`, and the selectors (every other
    * word).
    */
  private case class Args(
    flags:     Set[String],
    out:       Option[Path],
    specs:     Vector[(Path, Option[Path])],
    blocks:    Vector[(String, String, Path)],
    selectors: Vector[String]
  )

  private def parse(args: Seq[String]): Args = {
    var a    = Args(Set.empty, None, Vector.empty, Vector.empty, Vector.empty)
    var rest = args.toList
    while (rest.nonEmpty) {
      rest = rest match {
        case "--block" :: n :: "--params" :: j :: "--out" :: d :: t =>
          a = a.copy(blocks = a.blocks :+ ((n, j, RepoPaths.resolve(d)))); t
        case "--block" :: _                     => refuse("--block NAME --params JSON --out DIR, in that order")
        case "--spec" :: f :: "--out" :: d :: t =>
          a = a.copy(specs = a.specs :+ (RepoPaths.resolve(f) -> Some(RepoPaths.resolve(d)))); t
        case "--spec" :: f :: t                 => a = a.copy(specs = a.specs :+ (RepoPaths.resolve(f) -> None)); t
        case "--out" :: d :: t                  => a = a.copy(out = Some(RepoPaths.resolve(d))); t
        case f :: t if f.startsWith("--")       => a = a.copy(flags = a.flags + f); t
        case x :: t                             => a = a.copy(selectors = a.selectors :+ x); t
        case Nil                                => Nil
      }
    }
    a
  }

  def main(argv: Array[String]): Unit = {
    val args = parse(argv.toSeq)

    if (args.flags("--describe")) {
      println(describe)
      return
    }

    if (args.specs.nonEmpty || args.blocks.nonEmpty) {
      // Every block checked before anything is built, so a refused one leaves no other's output half written
      val blocks =
        try args.blocks.map { case (n, j, out) => (block(n, j), out) }
        catch { case e: IllegalArgumentException => refuse(e.getMessage) }
      for ((spec, out) <- args.specs) {
        val done = emitSpec(spec, out.getOrElse(spec.toAbsolutePath.getParent))
        done.files.foreach(f => println(s"[snax-forge] wrote ${RepoPaths.display(f)}"))
        println(s"[snax-forge] ${done.top} from ${RepoPaths.display(spec)}: ${done.modules.size} module(s)")
      }
      for ((b, out) <- blocks) {
        val done = emitBlock(b, out)
        done.files.foreach(f => println(s"[snax-forge] wrote ${RepoPaths.display(f)}"))
        println(s"[snax-forge] ${done.top} from ${b.name}: ${done.modules.size} module(s)")
      }
      return
    }

    if (args.flags("--list")) {
      println("[snax-forge] emittable targets:")
      catalogue.foreach(t => println(f"  ${t.label}%-30s ${t.note}"))
      return
    }

    // `--out`, else `$SNAX_FORGE_HW_OUT`, else `out/hw` under the repo root, each resolved from the repo root (not from
    // hw/chisel, where sbt starts), so `--out out/scratch` means what it looks like.
    val outDir = args.out
      .orElse(sys.env.get("SNAX_FORGE_HW_OUT").map(RepoPaths.resolve))
      .getOrElse(RepoPaths.out.resolve("hw"))

    // No selector means everything, which is what `pixi run chisel-gen` does
    // and what CI depends on. A selector is a case-insensitive substring, so
    // "Fifo" picks up both stream blocks without exact spelling.
    val chosen =
      if (args.selectors.isEmpty) catalogue
      else catalogue.filter(t => args.selectors.exists(s => t.label.toLowerCase.contains(s.toLowerCase)))

    if (chosen.isEmpty) {
      Console.err.println(s"[snax-forge] no target matches ${args.selectors.mkString(", ")}")
      Console.err.println("[snax-forge] run with --list to see the available targets")
      sys.exit(1)
    }

    chosen.foreach { target =>
      val module = elaborate(target.gen(), outDir, split = false)
      println(f"[snax-forge] ${target.label}%-30s -> $module%s.sv")
    }
    println(s"[snax-forge] emitted ${chosen.size} module(s) to $outDir")
  }

  private def refuse(why: String): Nothing = {
    Console.err.println(s"[snax-forge] $why")
    sys.exit(1)
  }

  /** What `emitSpec` or `emitBlock` wrote: the top module, every module, and the files (those in `sv/`, then
    * `emitted.json`).
    */
  case class Emitted(top: String, modules: Seq[String], files: Seq[Path])

  /** `--spec <accelerator.json> [--out DIR]`: the accelerator of `accelerator.json` (CONTRACTS, "Accelerator RTL
    * spec"), `<tag>_shell_wrapper` and what it holds, one SystemVerilog file per module in `DIR/sv/` (default `DIR` the
    * folder of the file), and `DIR/emitted.json`: `rtl` (the file it was built from), `top`, `modules` and `files`
    * (paths from `DIR`). The files a previous `emitted.json` lists are removed first, so `sv/` holds this spec's files
    * only.
    */
  def emitSpec(specPath: Path, dir: Path): Emitted = {
    val spec = AcceleratorSpec.load(specPath)
    emitInto(dir, "rtl" -> ujson.Str(RepoPaths.display(specPath)))(sv => elaborate(new ShellWrapper(spec), sv, true))
  }

  /** What `--block` builds, its params checked: its name, its params in its order (as `emitted.json` lists them) and
    * its module.
    */
  case class Block(name: String, params: ujson.Obj, gen: () => RawModule)

  /** `--block NAME --params JSON`, checked: `NAME` a datapath of `Datapaths.all` or a stream block of `streamBlocks`,
    * `JSON` its params (a JSON object; a datapath's as `blocks.json` lists them, an `int` filled from `unit.params`
    * left out at its `min` by `Datapath.complete`, then `Datapath.check`'s refusals); an unknown name, JSON that is not
    * an object or a refused param an `IllegalArgumentException`.
    */
  def block(name: String, json: String): Block = {
    val make = streamBlocks.get(name).orElse(Datapaths.all.find(_.name == name).map(d => datapath(d, _))).getOrElse {
      val (ds, ss) = (Datapaths.all.map(_.name).mkString(", "), streamBlocks.keys.mkString(", "))
      throw new IllegalArgumentException(s"no block $name: name a datapath ($ds) or a stream block ($ss)")
    }
    val p    =
      try ujson.read(json).obj.toMap
      catch { case e: Exception => throw new IllegalArgumentException(s"$name: --params $json is not a JSON object") }
    make(p)
  }

  /** The datapath `d` on `params`, completed (`Datapath.complete`) and checked (`Datapath.check`), as a `Block`. */
  def datapath(d: Datapaths.Datapath, params: Map[String, ujson.Value]): Block = {
    val p = d.complete(params)
    d.check(p)
    Block(d.name, ujson.Obj.from(d.params.map { case (n, _, _) => n -> p(n) }), () => d.make(p))
  }

  /** The stream blocks `--block` builds beside the catalogue's datapaths, by name, each from its params (checked, an
    * `IllegalArgumentException` that says the fix). They are not in `Datapaths`, which holds what a unit binds: the
    * flow places a stream block on a port (FifoPacker on a packed outer output, FifoSlicer on a sliced outer input).
    */
  val streamBlocks: Map[String, Map[String, ujson.Value] => Block] = Map(
    "FifoPacker" -> (p =>
      stream("FifoPacker", Seq("inLanes", "laneType", "words"), p)((lanes, t, words) =>
        new FifoPacker(lanes, t, words) with RequireAsyncReset
      )
    ),
    "FifoSlicer" -> (p =>
      stream("FifoSlicer", Seq("words", "laneType", "outLanes"), p)((lanes, t, words) =>
        new FifoSlicer(words, t, lanes) with RequireAsyncReset
      )
    )
  )

  /** The stream block `name` on `p`, its params `names` in its constructor's order: its lanes param (`inLanes` or
    * `outLanes`) and `words` positive ints, `laneType` an int type of `IntType`, the lanes dividing the beat's `words`
    * * 64 / width elements (the constructor's `require`, checked first so its refusal says the fix). `make(lanes,
    * laneType, words)` builds it with an asynchronous, active-high reset (`RequireAsyncReset`), its clock and reset as
    * inside an accelerator, where `ShellWrapper` builds it under `withClockAndReset(clk_i, (!rst_ni).asAsyncReset)`.
    */
  private def stream(name: String, names: Seq[String], p: Map[String, ujson.Value])(
    make: (Int, String, Int) => RawModule
  ): Block = {
    val lanes            = names.find(_.endsWith("Lanes")).get
    val (missing, extra) = (names.filterNot(p.contains), p.keys.filterNot(names.contains).toSeq.sorted)
    if (missing.nonEmpty || extra.nonEmpty)
      throw new IllegalArgumentException(
        s"$name: give params ${names.mkString(", ")}; missing [${missing.mkString(", ")}], extra [${extra.mkString(", ")}]"
      )
    def count(n: String): Int =
      p(n) match {
        case ujson.Num(x) if x.isWhole && x >= 1 && x <= Int.MaxValue => x.toInt
        case v => throw new IllegalArgumentException(s"$name.$n: ${ujson.write(v)}; give an int of at least 1")
      }
    val (n, words) = (count(lanes), count("words"))
    val laneType = p("laneType").strOpt.filter(IntType.names.contains).getOrElse {
      val t = ujson.write(p("laneType"))
      throw new IllegalArgumentException(s"$name.laneType: $t; give one of ${IntType.names.mkString(", ")}")
    }
    val total    = words.toLong * 64 / IntType(laneType).width
    if (total % n != 0)
      throw new IllegalArgumentException(
        s"$name: $lanes $n does not divide the beat's $total elements ($words word(s) of $laneType); " +
          s"give an $lanes that divides $total, or more words"
      )
    Block(name, ujson.Obj.from(names.map(k => k -> p(k))), () => make(n, laneType, words))
  }

  /** `--block NAME --params JSON --out DIR`: the block `b` alone as the top module with its ports as they are (`clock`,
    * `reset`, `io_*`) and an asynchronous, active-high reset as it has inside an accelerator (`ShellWrapper`'s
    * `withClockAndReset`): a datapath's from `DatapathModule.asyncTop`, a stream block's from its own builder. So the
    * module is the one an accelerator holds; into `DIR/sv/`, one file per module, with `DIR/emitted.json` (`block`,
    * `params`, `top`, `modules`, `files`), the files of a previous one removed first, as `emitSpec` does.
    */
  def emitBlock(b: Block, dir: Path): Emitted =
    emitInto(dir, "block" -> ujson.Str(b.name), "params" -> b.params) { sv =>
      DatapathModule.asyncTop(elaborate(b.gen(), sv, split = true))
    }

  /** The datapath `d` on `params` alone: `emitBlock` of its `datapath`. */
  def emitBlock(d: Datapaths.Datapath, params: Map[String, ujson.Value], dir: Path): Emitted =
    emitBlock(datapath(d, params), dir)

  /** `build` into `dir/sv/` (giving the top's name), after removing the files a previous `dir/emitted.json` lists, then
    * `dir/emitted.json`: `head`, `top`, `modules`, `files` (paths from `dir`).
    */
  private def emitInto(dir: Path, head: (String, ujson.Value)*)(build: Path => String): Emitted = {
    val sv      = dir.resolve("sv")
    val listing = dir.resolve("emitted.json")
    if (Files.exists(listing))
      ujson.read(Files.readString(listing))("files").arr.foreach(f => Files.deleteIfExists(dir.resolve(f.str)))
    val top     = build(sv)
    val files   = Files.list(sv).iterator.asScala.filter(_.toString.endsWith(".sv")).toSeq.sortBy(_.toString)
    val modules = files.flatMap(f => Files.readAllLines(f).asScala.collect { case ModuleLine(m) => m })
    val body    = head ++ Seq(
      "top"     -> ujson.Str(top),
      "modules" -> ujson.Arr.from(modules),
      "files"   -> ujson.Arr.from(files.map(f => dir.relativize(f).toString))
    )
    Files.writeString(listing, ujson.write(ujson.Obj.from(body), indent = 1) + "\n")
    Emitted(top, modules, files :+ listing)
  }

  private val ModuleLine = raw"module (\w+)\b.*".r

  /** `--describe`: the datapath catalogue from the registry (`Datapaths`), as `hw/chisel/blocks.json` holds it.
    */
  def describe: String = ujson.write(Datapaths.describe, indent = 1)

  /** Elaborate one module into `dir`, one file per module with `split` (else `<name>.sv` with everything): its name.
    *
    * `gen` is by-name for the same reason `Target.gen` is a thunk: construction has to happen inside the elaboration
    * context. The returned name is the module's `desiredName`, which is also the .sv filename ChiselStage writes.
    */
  def elaborate(gen: => RawModule, dir: Path, split: Boolean): String = {
    var name = "<unknown>"
    ChiselStage.emitSystemVerilogFile(
      {
        val m = gen
        name = m.name
        m
      },
      args = Array("--target-dir", dir.toString) ++ (if (split) Seq("--split-verilog") else Nil),
      firtoolOpts = firtoolOpts
    )
    name
  }
}
