package snax.forge.shell

import chisel3._
import chisel3.util._

import snax.forge.ctrl.GroupStart

/** A CSR request after `snax_csr_mux_demux`: the offset from the accelerator's first CSR, the data, write or read. */
class CsrReq extends Bundle {
  val addr  = UInt(32.W)
  val data  = UInt(32.W)
  val write = Bool()
}

class CsrRsp extends Bundle {
  val data = UInt(32.W)
}

/** `<tag>_csr`: the generated accelerator's own CSR manager (D130) over `accelerator.json`'s CSRs (C§17),
  * snax_cluster's `ReqRspManager` protocol as it is (`hw/chisel/src/main/scala/snax/reqRspManager/ReqRspManager.scala`
  * at 05615e4), for 32-bit CSRs without strobes. It holds the read-write CSRs and no counter: STATUS and COUNTER come
  * from the groups' controllers (`ro`).
  *
  *   - A write to a read-write CSR is taken when the request fires and reads back as written (32 bits).
  *   - A write to a group's START with bit 0 set fires that group's `starts(g)` with its JOBS and each JOB_BEATS in
  *     unit order as they stand (C§17: before the write, which is to START alone; JOBS and 1 for a group without a
  *     JOB_BEATS), and the request is ready only when `starts(g)` is: a start while the group is busy is held,
  *     `req.ready` low, until it is idle.
  *   - A read is answered in the cycle it fires, from the registers or the read-only inputs in that cycle; when `rsp`
  *     is not ready the answer is kept and `req.ready` stays low until it leaves.
  *   - Writes get no response. A read past the last CSR answers 0 and a write to a read-only CSR or past the last is
  *     dropped (ReqRspManager asserts on both).
  */
class AccCsr(spec: AcceleratorSpec) extends Module {
  override def desiredName: String = s"${spec.tag}_csr"

  private val csrs   = spec.csrs
  private val groups = spec.groups

  val io = IO(new Bundle {
    val req    = Flipped(Decoupled(new CsrReq))
    val rsp    = Decoupled(new CsrRsp)
    val starts = MixedVec(groups.map(g => Decoupled(new GroupStart(g.jobBeats.size))))
    val status = Input(Vec(groups.size, UInt(32.W)))
    val count  = Input(Vec(groups.size, UInt(32.W)))
  })

  private val regs = RegInit(VecInit(Seq.fill(csrs.size)(0.U(32.W))))

  private val readReg  = io.req.fire && !io.req.bits.write
  private val writeReg = io.req.fire && io.req.bits.write

  for (c <- csrs if c.access == "rw")
    when(writeReg && io.req.bits.addr === c.offset.U) {
      regs(c.offset) := io.req.bits.data
    }

  // Per group: a request that is its start, and the start's handshake
  private val startReq = groups.zipWithIndex.map { case (g, i) =>
    val at  = g.csr("START").get.offset
    val hit = io.req.valid && io.req.bits.write && io.req.bits.addr === at.U && io.req.bits.data(0)
    io.starts(i).valid      := writeReg && hit
    io.starts(i).bits.jobs  := regs(g.csr("JOBS").get.offset)
    io.starts(i).bits.beats := VecInit(if (g.jobBeats.isEmpty) Seq(1.U(32.W)) else g.jobBeats.map(c => regs(c.offset)))
    hit                     -> io.starts(i).ready
  }

  // A read's answer, kept while rsp is not ready
  private val readRegBusy   = RegNext(io.rsp.valid && !io.rsp.ready, false.B)
  private val readRegBuffer = RegNext(io.rsp.bits.data, 0.U)

  private val value = csrs.map { c =>
    val g = groups.indexWhere(_.window.contains(c))
    c.offset.U -> (c.name match {
      case "STATUS" if c.access == "ro"  => io.status(g)
      case "COUNTER" if c.access == "ro" => io.count(g)
      case _                             => regs(c.offset)
    })
  }
  private val read  = MuxLookup(io.req.bits.addr, 0.U)(value)

  when(readReg) {
    io.rsp.valid     := true.B
    io.rsp.bits.data := read
  }.elsewhen(readRegBusy) {
    io.rsp.valid     := true.B
    io.rsp.bits.data := readRegBuffer
  }.otherwise {
    io.rsp.valid     := false.B
    io.rsp.bits.data := 0.U
  }

  private val held = startReq.foldLeft(false.B) { case (acc, (hit, ready)) => acc || (hit && !ready) }
  io.req.ready := !readRegBusy && !held
}
