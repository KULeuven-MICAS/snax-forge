package snax.forge

import java.nio.file.{Files, Path, Paths}

import scala.annotation.tailrec

/** Repo-root anchored paths, so a relative path means the same thing wherever the JVM was started.
  *
  * The Scala counterpart of `snax_forge/sdfg/paths.py`. sbt runs with its working directory at `hw/chisel`, because
  * that is where build.sbt lives, so a path typed at the repo root -- `experiments/vecadd_loop.json`, which is what a
  * shell tab-completes -- resolves to `hw/chisel/experiments/...` and is not found. Requiring `../../experiments/...`
  * instead would work and would be wrong: the correct path would then depend on which task you happened to run.
  *
  * Resolution order, deliberately in this sequence:
  *
  *   1. absolute paths are taken as given
  *   1. paths that exist relative to the current directory win, so running from inside `hw/chisel` behaves the way a
  *      shell would lead you to expect
  *   1. otherwise, relative to the repo root
  */
object RepoPaths {

  /** Marker that identifies the repo root. pixi.toml rather than .git, so this still works in a source tarball or a
    * worktree without a .git directory.
    */
  private val Marker = "pixi.toml"

  lazy val root: Path = {
    val start = Paths.get(sys.props.getOrElse("user.dir", ".")).toAbsolutePath.normalize()

    @tailrec
    def climb(dir: Path): Option[Path] =
      if (dir == null) None
      else if (Files.exists(dir.resolve(Marker))) Some(dir)
      else climb(dir.getParent)

    climb(start).getOrElse {
      // Not fatal: an absolute path still works, and so does a relative one
      // that happens to resolve from here. Worth saying out loud, because
      // every "not found" after this point will be confusing otherwise.
      Console.err.println(
        s"[snax-forge] warning: no $Marker found above $start; relative paths will resolve against it"
      )
      start
    }
  }

  /** Resolve a user-supplied path. See the class comment for the order. */
  def resolve(path: String): Path = {
    // `given` would be the natural name and is a Scala 3 keyword.
    val candidate = Paths.get(path)
    if (candidate.isAbsolute) candidate.normalize()
    else if (Files.exists(candidate)) candidate.toAbsolutePath.normalize()
    else root.resolve(path).normalize()
  }

  /** `out/`, the generated-artifact tree. Gitignored, and shared with the Python side. */
  def out: Path = root.resolve("out")

  /** Path for display: relative to the repo root when it is under it, absolute otherwise. Absolute paths in logs are
    * accurate and unreadable; this keeps them short without lying about anything outside the tree.
    */
  def display(path: Path): String = {
    val abs = path.toAbsolutePath.normalize()
    if (abs.startsWith(root)) root.relativize(abs).toString else abs.toString
  }
}