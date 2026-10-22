package snax.forge.shell

import chisel3._
import chisel3.util._

/** Between the CSR manager's `ctrl` and a block's `start`, configuration pins and `busy` (D130).
  *
  *   - An accepted start (`ctrl.fire`) is the block's `start` for that cycle, with the fields on its configuration pins
  *     (`config`); the blocks sample them only at start.
  *   - `ctrl.ready` is not busy, so a start while the block is busy waits in the manager until busy falls.
  *   - The read-only CSRs: busy, then the counter, which an accepted start resets and each busy cycle increments, held
  *     after the run (GeMMX's performance counter, and `SnaxShell`'s rule). A start the block ignores (a count of 0)
  *     still resets it.
  *   - `ties` drives pins the configuration does not, each to its constant (a fixed op's `opSel`).
  *
  * Plain on purpose: no special case for a zero count (open item 83).
  *
  * @param fields
  *   the configuration fields, name and width, as the manager's `ctrl` carries them
  * @param tied
  *   pins tied to a constant: name, width, value
  */
class CtrlAdapter(val fields: Seq[(String, Int)], val tied: Seq[(String, Int, Int)], tag: String) extends Module {
  override def desiredName: String = s"${tag}_ctrl_adapter"

  val io = IO(new Bundle {
    val ctrl   = Flipped(Decoupled(new Fields(fields)))
    val start  = Output(Bool())
    val config = Output(new Fields(fields))
    val ties   = Output(new Fields(tied.map { case (n, w, _) => n -> w }))
    val busy   = Input(Bool())
    val ro     = Output(Vec(2, UInt(32.W)))
  })

  private val counter = RegInit(0.U(32.W))
  when(io.ctrl.fire) {
    counter := 0.U
  }.elsewhen(io.busy) {
    counter := counter + 1.U
  }

  io.ctrl.ready := !io.busy
  io.start      := io.ctrl.fire
  io.config     := io.ctrl.bits
  tied.foreach { case (n, w, v) => io.ties(n) := v.U(w.W) }
  io.ro(0)      := io.busy
  io.ro(1)      := counter
}
