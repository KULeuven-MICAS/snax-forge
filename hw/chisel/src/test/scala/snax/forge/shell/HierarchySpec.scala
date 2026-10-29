package snax.forge.shell

import java.nio.file.{Files, Path}

import scala.jdk.CollectionConverters._

import org.scalatest.flatspec.AnyFlatSpec

import snax.forge.Emit

/** The emitted hierarchy is `bundle.json`'s `modules` (C§18; snax-forge's `snax_forge/gen/bundle.py`, `module_tree`,
  * which writes it from `accelerator.json` without elaborating): `<tag>_shell_wrapper` holds `<tag>_csr`, `<tag>_ctrl`
  * and `<tag>_datapath`; `<tag>_csr` nothing; `<tag>_ctrl` a `<tag>_<group>_ctrl` per group; `<tag>_datapath` a
  * `<tag>_<unit>` per unit, then a `<tag>_<link>` per buffer, group by group. Checked on every fixture, dot 4's (a
  * wire) and dot 6's (a buffer) among them, by reading each module's instances off the emitted SystemVerilog (firtool's
  * ` <module> <instance> (` lines).
  */
class HierarchySpec extends AnyFlatSpec {

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

  behavior of "Emit --spec's hierarchy"

  for (f <- RtlFixtures.all)
    it should s"be bundle.json's modules for ${f.name}" in {
      val out  = Files.createTempDirectory(s"emit-${f.name}")
      val done = Emit.emitSpec(f.folder.resolve("accelerator.json"), out)
      val sv   = out.resolve("sv")
      val want = tree(f.spec)
      assert(children(sv, done.top) == want.map(_._1), s"${done.top}")
      for ((m, kids) <- want) assert(children(sv, m) == kids, s"$m")
      assert(want.flatMap { case (m, k) => m +: k }.forall(done.modules.contains), s"emitted.json: ${done.modules}")
    }

  it should "have the fixtures of a wire and a buffer" in
    assert(Set("dot_4", "dot_6").subsetOf(RtlFixtures.all.map(_.name).toSet))
}
