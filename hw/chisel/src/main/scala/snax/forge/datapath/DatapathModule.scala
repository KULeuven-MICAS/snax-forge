package snax.forge.datapath

import scala.util.DynamicVariable

import chisel3._
import chisel3.reflect.DataMirror

/** A catalogue datapath (`Datapaths`) as a module: data and enables only, its live ports and pins the elements of its
  * `io`.
  *
  * Its reset is Chisel's default (a top's `Bool`, else inferred from its parent: inside an accelerator the
  * `ShellWrapper`'s asynchronous one), but within `DatapathModule.asyncTop` an asynchronous reset of its own, which is
  * how `Emit --block` builds one alone as the module it is inside an accelerator, with no wrapper around it.
  */
abstract class DatapathModule extends Module {
  override def resetType: Module.ResetType.Type = DatapathModule.reset.value
}

object DatapathModule {

  private val reset = new DynamicVariable[Module.ResetType.Type](Module.ResetType.Default)

  /** `body` with every datapath constructed in it given an asynchronous reset (`Module.ResetType.Asynchronous`). */
  def asyncTop[T](body: => T): T = reset.withValue(Module.ResetType.Asynchronous)(body)

  /** The `io` of `m`, instantiated (read once the module is closed). */
  def io(m: DatapathModule): Bundle = DataMirror.modulePorts(m).collectFirst { case ("io", b: Bundle) => b }.get
}
