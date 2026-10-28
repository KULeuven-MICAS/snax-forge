package snax.forge.shell

import chisel3._

import chisel3.util.Decoupled

import snax.forge.ctrl.{GroupCtrl, Handshake, NestConfig}

/** `<tag>_ctrl`: the groups' controllers (`GroupCtrl`, `<tag>_<group>_ctrl`), one per group of `accelerator.json`
  * (D146, D147). Per group its start from `<tag>_csr`, its STATUS and COUNTER back, its pins; per unit the valids and
  * readies of its outer ports (the wrapper splits each Decoupled port: data to `<tag>_datapath`, handshake here) and
  * the enables to its datapath (`fire`, `first`, `adv`).
  */
class AccCtrl(spec: AcceleratorSpec) extends Module {
  override def desiredName: String = s"${spec.tag}_ctrl"

  /** One unit's side: its inputs' and output's handshakes, in the streamer's order, and its datapath's enables. */
  class UnitIo(inputs: Int) extends Bundle {
    val in    = Vec(inputs, new Handshake)
    val out   = Flipped(new Handshake)
    val fire  = Output(Bool())
    val first = Output(Bool())
    val adv   = Output(Bool())
  }

  class GroupIo extends Bundle {
    val start   = Flipped(Decoupled(new NestConfig))
    val status  = Output(UInt(32.W))
    val counter = Output(UInt(32.W))
    val busy    = Output(Bool())
    val done    = Output(Bool())
  }

  val io = IO(new Bundle {
    val groups = Vec(spec.groups.size, new GroupIo)
    val units  = chisel3.util.MixedVec(spec.units.map(u => new UnitIo(spec.ins(u.name).size)))
  })

  for ((g, i) <- spec.groups.zipWithIndex) {
    require(g.units.size == 1, s"group ${g.name}: ${g.units.size} units; without links a group is one unit (B3)")
    val u    = g.units.head
    val k    = spec.units.indexOf(u)
    val ctrl = Module(new GroupCtrl(u.latency, u.ii, spec.ins(u.name).size, g.name, spec.tag))
    ctrl.io.start <> io.groups(i).start
    io.groups(i).status  := ctrl.io.status
    io.groups(i).counter := ctrl.io.counter
    io.groups(i).busy    := ctrl.io.busy
    io.groups(i).done    := ctrl.io.done
    ctrl.io.in <> io.units(k).in
    ctrl.io.out <> io.units(k).out
    io.units(k).fire     := ctrl.io.fire
    io.units(k).first    := ctrl.io.first
    io.units(k).adv      := ctrl.io.adv
  }
}
