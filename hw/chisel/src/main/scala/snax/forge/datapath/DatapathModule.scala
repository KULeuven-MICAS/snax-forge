package snax.forge.datapath

import chisel3._
import chisel3.reflect.DataMirror

/** A catalogue datapath (`Datapaths`) as a module: data and enables only, its live ports and pins the elements of its
  * `io`.
  */
abstract class DatapathModule extends Module

object DatapathModule {

  /** The `io` of `m`, instantiated (read once the module is closed). */
  def io(m: DatapathModule): Bundle = DataMirror.modulePorts(m).collectFirst { case ("io", b: Bundle) => b }.get
}
