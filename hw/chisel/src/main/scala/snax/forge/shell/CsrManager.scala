package snax.forge.shell

import scala.collection.immutable.SeqMap

import chisel3._
import chisel3.util._

/** Named UInt fields, one element per (name, width): the configuration a start hands the block. */
class Fields(val fields: Seq[(String, Int)]) extends Record {
  val elements: SeqMap[String, UInt] = SeqMap(fields.map { case (n, w) => n -> UInt(w.W) }: _*)
  def apply(name: String): UInt = elements(name)
}

/** A CSR request after `snax_csr_mux_demux`: the offset from the accelerator's first CSR, the data, write or read. */
class CsrReq extends Bundle {
  val addr  = UInt(32.W)
  val data  = UInt(32.W)
  val write = Bool()
}

class CsrRsp extends Bundle {
  val data = UInt(32.W)
}

/** The generated accelerator's own CSR manager (D130): snax_cluster's `ReqRspManager` protocol as it is
  * (`hw/chisel/src/main/scala/snax/reqRspManager/ReqRspManager.scala` at 05615e4), for 32-bit CSRs without strobes.
  *
  * CSRs, offsets from 0: the read-write `fields` in order, then start (read-write too), then `numRo` read-only ones.
  *
  *   - A write to a read-write CSR is taken when the request fires and reads back as written (32 bits).
  *   - Start is a write to the last read-write CSR with bit 0 set. It fires `ctrl` with the fields as they stand
  *     (before the write, which is to start alone), and the request is ready only when `ctrl` is: a start while the
  *     block is busy is held, `req.ready` low, until `ctrl.ready` rises.
  *   - A read is answered in the cycle it fires, from the registers or the read-only inputs (`ro`) in that cycle; when
  *     `rsp` is not ready the answer is kept and `req.ready` stays low until it leaves.
  *   - Writes get no response. A read past the last CSR answers 0 and a write past the read-write ones is dropped
  *     (ReqRspManager asserts on both).
  *
  * `ctrl.bits` carries each field's low `width` bits.
  *
  * @param fields
  *   the configuration fields, name and width (at most 32), in CSR order
  * @param numRo
  *   the read-only CSRs after start
  */
class CsrManager(val fields: Seq[(String, Int)], val numRo: Int, tag: String) extends Module {
  require(fields.forall { case (_, w) => w >= 1 && w <= 32 }, s"field widths must be 1 to 32 bits: $fields")

  override def desiredName: String = s"${tag}_csr_manager"

  private val numRw = fields.size + 1

  val io = IO(new Bundle {
    val req  = Flipped(Decoupled(new CsrReq))
    val rsp  = Decoupled(new CsrRsp)
    val ctrl = Decoupled(new Fields(fields))
    val ro   = Input(Vec(numRo, UInt(32.W)))
  })

  private val regs = RegInit(VecInit(Seq.fill(numRw)(0.U(32.W))))

  private val readReg  = io.req.fire  && !io.req.bits.write
  private val writeReg = io.req.fire  && io.req.bits.write
  private val startReq = io.req.valid && io.req.bits.write && io.req.bits.addr === (numRw - 1).U && io.req.bits.data(0)

  for (i <- 0 until numRw) {
    when(writeReg && io.req.bits.addr === i.U) {
      regs(i) := io.req.bits.data
    }
  }

  // A read's answer, kept while rsp is not ready
  private val readRegBusy   = RegNext(io.rsp.valid && !io.rsp.ready, false.B)
  private val readRegBuffer = RegNext(io.rsp.bits.data, 0.U)

  private val all  = regs ++ io.ro
  private val read = MuxLookup(io.req.bits.addr, 0.U)(all.zipWithIndex.map { case (r, i) => i.U -> r })

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

  io.ctrl.valid := writeReg && startReq
  fields.zipWithIndex.foreach { case ((n, w), i) => io.ctrl.bits(n) := regs(i)(w - 1, 0) }

  io.req.ready := Mux(readRegBusy, false.B, Mux(startReq, io.ctrl.ready, true.B))
}
