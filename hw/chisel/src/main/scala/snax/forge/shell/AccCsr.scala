package snax.forge.shell

import chisel3._
import chisel3.experimental.noPrefix
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

/** `<tag>_csr`: the generated accelerator's own CSR manager (D130) over `accelerator.json`'s CSRs (C§17), the
  * accelerator's interface (D152), snax_cluster's `ReqRspManager` protocol as it is
  * (`hw/chisel/src/main/scala/snax/reqRspManager/ReqRspManager.scala` at 05615e4), for 32-bit accesses without strobes.
  * Each CSR has its width (`AcceleratorSpec.CsrSpec`); a read gives its bits, the upper ones 0.
  *
  *   - START is a strobe: no register, it reads 0. A write to a group's START with bit 0 set fires that group's
  *     `starts(g)` with its JOBS and each JOB_BEATS in unit order as they stand (JOBS and 1 for a group without a
  *     JOB_BEATS), and the request is ready only when `starts(g)` is: a start while the group is busy is held,
  *     `req.ready` low, until it is idle. A write with bit 0 clear is taken and does nothing.
  *   - Every other read-write CSR is a register of its width (JOBS and JOB_BEATS: job_width, a unit's scalar its
  *     dtype's bits), which takes a write's low bits when the request fires. A scalar's CSRs (those with a `pin`, after
  *     its unit's JOB_BEATS) drive `scalars`, one element per CSR in CSR order, held until the next write: the wrapper
  *     joins a 64-bit one's two (low word first) onto its datapath pin.
  *   - Per group, from its control unit's `busy` and `done` (the done pulse, high in the run's last cycle E):
  *     - STATUS {done, busy}: done is sticky, set by the done pulse, so it reads 1 from E + 1, and cleared by an
  *       accepted start, so it reads 0 from a + 1; a set wins over a clear in the same cycle (the empty run: STATUS
  *       reads 2 from a + 1).
  *     - BUSY_COUNTER, kept by the submodule `<tag>_peripherals` (`AccPeripherals`, D168) from the accepted start and
  *       busy, read here: at c the busy cycles after a and before c, saturating at 2^w - 1 for its width w.
  *   - A read is answered in the cycle it fires, from the registers or the read-only CSRs in that cycle; when `rsp` is
  *     not ready the answer is kept (the widest CSR's bits) and `req.ready` stays low until it leaves.
  *   - Writes get no response. A read past the last CSR answers 0 and a write to a read-only CSR or past the last is
  *     dropped (ReqRspManager asserts on both).
  */
class AccCsr(spec: AcceleratorSpec) extends Module {
  override def desiredName: String = s"${spec.tag}_csr"

  private val csrs   = spec.csrs
  private val groups = spec.groups

  val io = IO(new Bundle {
    val req     = Flipped(Decoupled(new CsrReq))
    val rsp     = Decoupled(new CsrRsp)
    val starts  = MixedVec(groups.map(g => Decoupled(new GroupStart(g.jobBeats.size, spec.jobWidth))))
    val busy    = Input(Vec(groups.size, Bool()))
    val done    = Input(Vec(groups.size, Bool()))
    val scalars = Output(MixedVec(AccCsr.scalars(spec).map(c => UInt(c.width.W))))
  })

  private val reading = io.req.fire && !io.req.bits.write
  private val writing = io.req.fire && io.req.bits.write

  // The read-write CSRs but START, each a register of its width, named `<group>_[<unit>_]<csr>`
  private val regs: Map[Int, UInt] = (for {
    g      <- groups
    (u, c) <- g.units.flatMap(u => u.csrs.map(Some(u) -> _)) :+ (None -> g.csr("JOBS").get)
  } yield {
    val name = (Seq(g.name) ++ u.map(_.name) :+ c.name.toLowerCase).mkString("_")
    val reg  = RegInit(0.U(c.width.W)).suggestName(name)
    when(writing && io.req.bits.addr === c.offset.U) {
      reg := io.req.bits.data(c.width - 1, 0)
    }
    c.offset -> reg
  }).toMap

  // Per group: a request that is its start, the start's handshake, STATUS and BUSY_COUNTER
  private val startReq = groups.zipWithIndex.map { case (g, i) =>
    val at   = g.csr("START").get.offset
    val hit  = io.req.valid && io.req.bits.write && io.req.bits.addr === at.U && io.req.bits.data(0)
    val ones = Seq(1.U(spec.jobWidth.W))
    io.starts(i).valid      := writing && hit
    io.starts(i).bits.jobs  := regs(g.csr("JOBS").get.offset)
    io.starts(i).bits.beats := VecInit(if (g.jobBeats.isEmpty) ones else g.jobBeats.map(c => regs(c.offset)))
    hit                     -> io.starts(i).ready
  }
  private val accepted = io.starts.map(_.fire)

  private val status      = noPrefix(groups.zipWithIndex.map { case (g, i) =>
    val done = RegInit(false.B).suggestName(s"${g.name}_done")
    when(io.done(i)) {
      done := true.B
    }.elsewhen(accepted(i)) {
      done := false.B
    }
    Cat(done, io.busy(i))
  })
  // BUSY_COUNTER, in the submodule `<tag>_peripherals` (D168)
  private val peripherals = Module(AccPeripherals(spec))
  peripherals.io.start := VecInit(accepted)
  peripherals.io.busy  := io.busy
  private val counter = peripherals.io.busyCounter

  // A read's answer, the widest CSR's bits, kept while rsp is not ready
  private val w          = spec.readWidth
  private val answerKept = RegNext(io.rsp.valid && !io.rsp.ready, false.B)
  private val answer     = RegNext(io.rsp.bits.data(w - 1, 0), 0.U(w.W))

  private val value = csrs.map { c =>
    val g = groups.indexWhere(_.window.contains(c))
    c.offset.U -> (c.name match {
      case "START" if c.access == "rw"        => 0.U(w.W)
      case "STATUS" if c.access == "ro"       => status(g).pad(w)
      case "BUSY_COUNTER" if c.access == "ro" => counter(g).pad(w)
      case _                                  => regs(c.offset).pad(w)
    })
  }
  private val read  = MuxLookup(io.req.bits.addr, 0.U(w.W))(value)

  when(reading) {
    io.rsp.valid     := true.B
    io.rsp.bits.data := read
  }.elsewhen(answerKept) {
    io.rsp.valid     := true.B
    io.rsp.bits.data := answer
  }.otherwise {
    io.rsp.valid     := false.B
    io.rsp.bits.data := 0.U
  }

  private val held = startReq.foldLeft(false.B) { case (acc, (hit, ready)) => acc || (hit && !ready) }
  io.req.ready := !answerKept && !held

  for ((c, i) <- AccCsr.scalars(spec).zipWithIndex) io.scalars(i) := regs(c.offset)
}

object AccCsr {

  /** The scalars' CSRs (those with a `pin`), in CSR order: `io.scalars`' elements. */
  def scalars(spec: AcceleratorSpec): Seq[AcceleratorSpec.CsrSpec] = spec.csrs.filter(_.pin.nonEmpty)
}
