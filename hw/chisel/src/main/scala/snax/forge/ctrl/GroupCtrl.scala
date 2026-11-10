package snax.forge.ctrl

import chisel3._
import chisel3.util._

/** A group's settings as `<tag>_csr` keeps them, wires to its control unit (D168): JOBS, then each of its `open` units'
  * JOB_BEATS in unit order, each `bits` wide (the CSRs' job_width, D152); constant through a run, since the csr holds
  * their writes while the group is busy.
  */
class GroupSettings(val open: Int, bits: Int = 32) extends Bundle {
  val jobs  = UInt(bits.W)
  val beats = Vec(open, UInt(bits.W))
}

/** A group control unit's ports (`GroupCtrl`): its start, a bare strobe, and its settings, the csr's wires; busy, its
  * done pulse and pin; its outer handshakes; each unit's and buffer's enables; the cycle's class.
  */
class GroupCtrlIO(desc: GroupDesc, jobWidth: Int = 32) extends Bundle {
  val start    = Flipped(Decoupled(UInt(0.W)))
  val settings = Input(new GroupSettings(desc.open.size, jobWidth))
  val in       = Vec(desc.units.map(_.inputs).sum, new Handshake)
  val out      = Flipped(new Handshake)
  val units    = MixedVec(desc.units.map(u => new UnitEnables(u.span)))
  val links    = MixedVec(desc.buffers.map(i => new LinkEnables(desc.links(i).depth)))
  val busy     = Output(Bool())
  val pulse    = Output(Bool())
  val done     = Output(Bool())
  val cls      = Output(UInt(2.W))
}

/** One group's control unit (D146, D147; C§17's Control), `<tag>_<group>_ctrl`, generated from the group's description
  * (`GroupDesc`): its start and its busy and done around its control model, `GroupRule` (snax-gvsoc's SnaxGroupRule).
  * STATUS and BUSY_COUNTER are the interface's, `<tag>_csr`'s, from `busy` and `pulse` (D152).
  *
  *   - `start` is START written with bit 0 set, a bare strobe from `<tag>_csr`. It is ready when the group is idle (the
  *     csr holds a START while busy, the CSR request not ready, until E + 1); it is accepted in the cycle it fires, a.
  *   - `settings`: JOBS and each JOB_BEATS in unit order, the csr's CSRs as wires, read live through the run with no
  *     copy (D168): the csr holds their writes while the group is busy, so they stay constant from a to E. A unit with
  *     `params.T` has that fixed T.
  *   - JOBS 0 or any JOB_BEATS 0 is the empty run, tested on the wires in a: it ends in a (E = a) without starting the
  *     model, so busy never rises.
  *   - `busy`: from a + 1 to E (STATUS bit 0).
  *   - `pulse`, the done pulse: high in E (the sink's last push, or a for the empty run), for `<tag>_csr`'s sticky done
  *     bit and BUSY_COUNTER.
  *   - Pins: `busy` and `done`, the done pulse registered: high in E + 1 (a + 1 for an empty run).
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
  * @param jobWidth
  *   the bits of JOBS and each JOB_BEATS (accelerator.json's, D152)
  */
class GroupCtrl(val desc: GroupDesc, tag: String, jobWidth: Int = 32) extends Module {

  override def desiredName: String = s"${tag}_${desc.name}_ctrl"

  val io = IO(new GroupCtrlIO(desc, jobWidth))

  private val rule = new GroupRule(desc, jobWidth)

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

  private val busy     = rule.io.busy
  private val accepted = io.start.fire
  private val empty    = io.settings.jobs === 0.U || io.settings.beats.map(_ === 0.U).foldLeft(false.B)(_ || _)

  io.start.ready   := !busy
  rule.io.config   := accepted && !empty
  rule.io.settings := io.settings

  private val pulse = rule.io.done || (accepted && empty)

  io.busy  := busy
  io.pulse := pulse
  io.done  := RegNext(pulse, false.B)
}
