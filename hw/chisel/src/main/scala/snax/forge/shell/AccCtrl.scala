package snax.forge.shell

import chisel3._
import chisel3.util.MixedVec

import snax.forge.ctrl.{GroupCtrl, GroupCtrlIO, GroupDesc, Link, UnitDesc}

/** `<tag>_ctrl`: the groups' control units (`GroupCtrl`, `<tag>_<group>_ctrl`), one per group of `accelerator.json`
  * (D146, D147), each generated from its group's description (`AccCtrl.desc`). Per group its start from `<tag>_csr`,
  * its STATUS and COUNTER back, its pins; the valids and readies of its units' outer ports (the wrapper splits each
  * Decoupled port: data to `<tag>_datapath`, handshake here), in the streamer's order; each unit's and buffer's enables
  * to `<tag>_datapath`.
  */
class AccCtrl(spec: AcceleratorSpec) extends Module {
  override def desiredName: String = s"${spec.tag}_ctrl"

  private val descs = spec.groups.map(AccCtrl.desc(spec, _))

  val io = IO(new Bundle {
    val groups = MixedVec(descs.map(d => new GroupCtrlIO(d)))
  })

  for ((d, i) <- descs.zipWithIndex) {
    val ctrl = Module(new GroupCtrl(d, spec.tag))
    io.groups(i) <> ctrl.io
  }
}

object AccCtrl {

  /** A group's description (C§17): per unit, in the file's order, its L, II and span, its outer inputs, its T when
    * fixed (`params.T`, or 1 without a JOB_BEATS) and its inputs no outer port takes; per link its ends by unit index,
    * the `to` port's position among its unit's inputs (in port order) and its depth.
    */
  def desc(spec: AcceleratorSpec, g: AcceleratorSpec.GroupSpec): GroupDesc = {
    def at(unit: String) = g.units.indexWhere(_.name == unit)
    GroupDesc(
      g.name,
      g.units.map { u =>
        val outer = spec.ins(u.name).map(_._2)
        UnitDesc(
          u.name,
          u.latency,
          u.ii,
          u.span,
          outer.size,
          u.fixedT,
          u.inPorts.indices.filterNot(j => outer.contains(u.inPorts(j)))
        )
      },
      g.links.map { l =>
        val to = at(l.to._1)
        Link(at(l.from._1), to, if (to < 0) -1 else g.units(to).inPorts.indexOf(l.to._2), l.depth)
      }
    )
  }
}
