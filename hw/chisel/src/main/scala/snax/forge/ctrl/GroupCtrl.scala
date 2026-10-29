package snax.forge.ctrl

import chisel3._
import chisel3.util._

/** One group's control unit (D146, D147; C§17's Control), `<tag>_<group>_ctrl`, generated from the group's description
  * (`GroupDesc`): its START, STATUS and COUNTER around its control model, `GroupRule` (snax-gvsoc's SnaxGroupRule).
  *
  *   - `start` is START written with bit 0 set, carrying JOBS and each JOB_BEATS in unit order as `<tag>_csr` sends
  *     them (JOBS and 1 when the group has none); a unit with `params.T` takes that fixed T. It is ready when the group
  *     is idle, so a start while busy is held (the CSR request not ready) until E + 1; it is accepted in the cycle it
  *     fires, a.
  *   - JOBS 0 or any T 0 is the empty run: it ends in a (E = a) without starting the model, so busy never rises.
  *   - The done pulse is high in E (the last unit's last push, or a for the empty run).
  *   - STATUS: bit 0 busy (from a + 1 to E), bit 1 done, sticky: set by the done pulse, so it reads 1 from E + 1;
  *     cleared by an accepted start, so it reads 0 from a + 1; a set wins over a clear in the same cycle (the empty
  *     run: STATUS reads 2 from a + 1).
  *   - COUNTER: reset by an accepted start, counting each busy cycle, held after the run (BlockGemm's performance
  *     counter): at c the busy cycles after a and before c.
  *   - Pins: `busy` (STATUS bit 0) and `done`, the done pulse registered: high in E + 1 (a + 1 for an empty run).
  *
  * Its ports: `in`, every unit's outer inputs' handshakes, unit by unit in port order; `out`, the last unit's outer
  * output; `units`, each unit's enables to its datapath; `links`, each buffer's, in link order (a wire has none);
  * `cls`, the cycle's class, as the run line counts it (busy 0, stall_in 1, stall_out 2, flush 3). `fire`, `first` and
  * `adv` are the head unit's, a one-unit group's whole.
  *
  * @param desc
  *   the group: its units and links (`shell.AcceleratorSpec`'s group, `AccCtrl.desc`)
  * @param tag
  *   the platform's accelerator slot, in the module's name
  */
class GroupCtrl(val desc: GroupDesc, tag: String) extends Module {

  /** A group of one unit: B1's controller, L and II, `inputs` joined, its T from the start. */
  def this(latency: Int, ii: Int, inputs: Int, group: String, tag: String) =
    this(GroupDesc(group, Seq(UnitDesc(group, latency, ii, 1, inputs, None)), Nil), tag)

  override def desiredName: String = s"${tag}_${desc.name}_ctrl"

  private val open = desc.open
  private val ins  = desc.units.map(_.inputs).sum

  val io = IO(new Bundle {
    val start   = Flipped(Decoupled(new NestConfig((open.size - 1) max 0)))
    val status  = Output(UInt(32.W))
    val counter = Output(UInt(32.W))
    val in      = Vec(ins, new Handshake)
    val out     = Flipped(new Handshake)
    val units   = MixedVec(desc.units.map(u => new UnitEnables(u.span)))
    val links   = MixedVec(desc.buffers.map(k => new LinkEnables(desc.depths(k))))
    val fire    = Output(Bool())
    val first   = Output(Bool())
    val adv     = Output(Bool())
    val busy    = Output(Bool())
    val done    = Output(Bool())
    val cls     = Output(UInt(2.W))
  })

  private val nest = new GroupRule(desc)

  // The outer inputs, flat in unit order, onto each unit's
  private val flat: Seq[Handshake] = desc.units.indices.flatMap(k => nest.io.in(k).toSeq)
  for ((r, p) <- flat.zip(io.in)) {
    r.valid := p.valid
    p.ready := r.ready
  }
  nest.io.out.ready := io.out.ready
  io.out.valid := nest.io.out.valid
  io.units     := nest.io.units
  io.links     := nest.io.links
  io.fire      := nest.io.units(0).fire
  io.first     := nest.io.units(0).first
  io.adv       := (if (desc.units.head.latency == 0) true.B else nest.io.units(0).adv)
  io.cls       := nest.io.cls

  // Each unit's T: its fixed one, else the next JOB_BEATS of the start in unit order
  private val words = io.start.bits.jobBeats +: io.start.bits.beats.map(_.toSeq).getOrElse(Seq.empty)
  private val ts    = {
    var next = 0
    desc.units.map { u =>
      u.fixedT.map(_.U(32.W)).getOrElse { next += 1; words(next - 1) }
    }
  }

  private val busy     = nest.io.busy
  private val accepted = io.start.fire
  private val empty    = io.start.bits.jobs === 0.U || ts.map(_ === 0.U).reduce(_ || _)

  io.start.ready           := !busy
  nest.io.config.valid     := accepted && !empty
  nest.io.config.bits.jobs := io.start.bits.jobs
  nest.io.config.bits.ts   := VecInit(ts)

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
