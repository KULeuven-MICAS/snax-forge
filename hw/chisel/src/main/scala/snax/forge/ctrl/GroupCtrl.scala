package snax.forge.ctrl

import chisel3._
import chisel3.util._

/** A group's start as `<tag>_csr` sends it (C§17): JOBS, then each of its `open` units' JOB_BEATS in unit order
  * (`beats`; one word, 1, when the group has none).
  */
class GroupStart(val open: Int) extends Bundle {
  val jobs  = UInt(32.W)
  val beats = Vec(open max 1, UInt(32.W))
}

/** A group control unit's ports (`GroupCtrl`): its start, STATUS, COUNTER and pins; its outer handshakes; each unit's
  * and buffer's enables; the cycle's class.
  */
class GroupCtrlIO(desc: GroupDesc) extends Bundle {
  val start   = Flipped(Decoupled(new GroupStart(desc.open.size)))
  val status  = Output(UInt(32.W))
  val counter = Output(UInt(32.W))
  val in      = Vec(desc.units.map(_.inputs).sum, new Handshake)
  val out     = Flipped(new Handshake)
  val units   = MixedVec(desc.units.map(u => new UnitEnables(u.span)))
  val links   = MixedVec(desc.buffers.map(i => new LinkEnables(desc.links(i).depth)))
  val busy    = Output(Bool())
  val done    = Output(Bool())
  val cls     = Output(UInt(2.W))
}

/** One group's control unit (D146, D147; C§17's Control), `<tag>_<group>_ctrl`, generated from the group's description
  * (`GroupDesc`): its START, STATUS and COUNTER around its control model, `GroupRule` (snax-gvsoc's SnaxGroupRule).
  *
  *   - `start` is START written with bit 0 set, carrying JOBS and each JOB_BEATS in unit order as `<tag>_csr` sends
  *     them (JOBS and 1 when the group has none); a unit with `params.T` takes that fixed T. It is ready when the group
  *     is idle, so a start while busy is held (the CSR request not ready) until E + 1; it is accepted in the cycle it
  *     fires, a.
  *   - JOBS 0 or any T 0 is the empty run: it ends in a (E = a) without starting the model, so busy never rises.
  *   - The done pulse is high in E (the sink's last push, or a for the empty run).
  *   - STATUS: bit 0 busy (from a + 1 to E), bit 1 done, sticky: set by the done pulse, so it reads 1 from E + 1;
  *     cleared by an accepted start, so it reads 0 from a + 1; a set wins over a clear in the same cycle (the empty
  *     run: STATUS reads 2 from a + 1).
  *   - COUNTER: reset by an accepted start, counting each busy cycle, held after the run (BlockGemm's performance
  *     counter): at c the busy cycles after a and before c.
  *   - Pins: `busy` (STATUS bit 0) and `done`, the done pulse registered: high in E + 1 (a + 1 for an empty run).
  *
  * Its ports: `in`, every unit's outer inputs' handshakes, unit by unit in port order; `out`, the outer output, its
  * owner's (`GroupDesc.output`); `units`, each unit's enables to its datapath; `links`, each buffer's, in link order (a
  * wire has none); `cls`, the cycle's class, as the run line counts it (busy 0, stall_in 1, stall_out 2, flush 3). Test
  * ports, which the shell leaves unread: each unit's `give` and `adv`, each buffer's `occupancy`, and `cls`; the specs
  * compare them with snax-gvsoc's cases and `GroupModel` every cycle.
  *
  * @param desc
  *   the group: its units and links (`shell.AcceleratorSpec`'s group, `AccCtrl.desc`)
  * @param tag
  *   the platform's accelerator slot, in the module's name
  */
class GroupCtrl(val desc: GroupDesc, tag: String) extends Module {

  override def desiredName: String = s"${tag}_${desc.name}_ctrl"

  val io = IO(new GroupCtrlIO(desc))

  private val rule = new GroupRule(desc)

  // The outer inputs, flat in unit order, onto each unit's
  private val flat: Seq[Handshake] = desc.units.indices.flatMap(k => rule.io.in(k).toSeq)
  for ((r, p) <- flat.zip(io.in)) {
    r.valid := p.valid
    p.ready := r.ready
  }
  rule.io.out.ready := io.out.ready
  io.out.valid := rule.io.out.valid
  io.units     := rule.io.units
  io.links     := rule.io.links
  io.cls       := rule.io.cls

  // Each unit's T: its fixed one, else the next JOB_BEATS of the start in unit order
  private val words = io.start.bits.beats
  private val ts    = {
    var next = 0
    desc.units.map { u =>
      u.fixedT.map(_.U(32.W)).getOrElse { next += 1; words(next - 1) }
    }
  }

  private val busy     = rule.io.busy
  private val accepted = io.start.fire
  private val empty    = io.start.bits.jobs === 0.U || ts.map(_ === 0.U).reduce(_ || _)

  io.start.ready           := !busy
  rule.io.config.valid     := accepted && !empty
  rule.io.config.bits.jobs := io.start.bits.jobs
  rule.io.config.bits.ts   := VecInit(ts)

  private val pulse   = rule.io.done || (accepted && empty)
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
