package snax.forge.ctrl

import chisel3._
import chisel3.util._

/** One group's control (D146): its START, STATUS and COUNTER around its shell's `NestCtrl` (in B1 a group is one unit,
  * so one shell). `<tag>_<group>_ctrl`.
  *
  *   - `start` is START written with bit 0 set, carrying JOBS and JOB_BEATS as they stand (`<tag>_csr`). It is ready
  *     when the group is idle, so a start while busy is held (the CSR request not ready) until E + 1; it is accepted in
  *     the cycle it fires, a.
  *   - JOBS 0 or JOB_BEATS 0 is the empty run: it ends in a (E = a) without starting the shell, so busy never rises.
  *   - The done pulse is high in E (the shell's last cycle, or a for the empty run).
  *   - STATUS: bit 0 busy (from a + 1 to E), bit 1 done, sticky: set by the done pulse, so it reads 1 from E + 1;
  *     cleared by an accepted start, so it reads 0 from a + 1; a set wins over a clear in the same cycle (the empty
  *     run: STATUS reads 2 from a + 1).
  *   - COUNTER: reset by an accepted start, counting each busy cycle, held after the run (BlockGemm's performance
  *     counter): at c the busy cycles after a and before c.
  *   - Pins: `busy` (STATUS bit 0) and `done`, the done pulse registered: high in E + 1 (a + 1 for an empty run).
  *
  * @param group
  *   the group's name, in the module's
  */
class GroupCtrl(val latency: Int, val ii: Int, val inputs: Int, group: String, tag: String) extends Module {
  override def desiredName: String = s"${tag}_${group}_ctrl"

  val io = IO(new Bundle {
    val start   = Flipped(Decoupled(new NestConfig))
    val status  = Output(UInt(32.W))
    val counter = Output(UInt(32.W))
    val in      = Vec(inputs, new Handshake)
    val out     = Flipped(new Handshake)
    val fire    = Output(Bool())
    val first   = Output(Bool())
    val adv     = Output(Bool())
    val busy    = Output(Bool())
    val done    = Output(Bool())
  })

  val nest = Module(new NestCtrl(latency, ii, inputs))
  nest.io.in <> io.in
  nest.io.out <> io.out
  io.fire  := nest.io.fire
  io.first := nest.io.first
  io.adv   := nest.io.adv

  private val busy     = nest.io.busy
  private val accepted = io.start.fire
  private val empty    = io.start.bits.jobs === 0.U || io.start.bits.jobBeats === 0.U

  io.start.ready       := !busy
  nest.io.config.valid := accepted && !empty
  nest.io.config.bits  := io.start.bits

  private val pulse   = nest.io.done || (accepted && empty)
  private val done    = RegInit(false.B)
  private val counter = RegInit(0.U(32.W))
  when(pulse) {
    done := true.B
  }.elsewhen(accepted) {
    done := false.B
  }
  when(accepted) {
    counter := 0.U
  }.elsewhen(busy) {
    counter := counter + 1.U
  }

  io.status  := Cat(done, busy)
  io.counter := counter
  io.busy    := busy
  io.done    := RegNext(pulse, false.B)
}
