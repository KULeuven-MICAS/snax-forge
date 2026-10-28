package snax.forge.shell

import chisel3._
import chiseltest._

import snax.forge.{Sink, Source}

/** `<tag>_shell_wrapper` under chiseltest, which drives a `Module`: the clock and an active-high reset become `clk_i`
  * and `rst_ni`, and the wrapper's other ports are this harness's `io`, with the same names, the groups' pins
  * (`<group>_busy_o`, `<group>_done_o`) among them, so a spec reads a group's busy without a CSR read.
  */
class ShellHarness(val spec: AcceleratorSpec) extends Module {
  private val wrapper = Module(new ShellWrapper(spec))
  wrapper.clk_i  := clock
  wrapper.rst_ni := !reset.asBool
  val io = IO(chiselTypeOf(wrapper.io))
  io <> wrapper.io

  /** A CSR's offset by name in a group (the first group by default). */
  def csr(name: String, group: Int = 0): Int = spec.groups(group).csr(name).get.offset
}

/** A core and a streamer around a `ShellHarness`, one call per cycle.
  *
  * The core: one CSR request at a time, held until `csr_req_ready_o` (as the Snitch core's write is), the response
  * ready as `rspReady(cycle)` says (always, by default). The streamer: input k is `sources(k)`, output k `sinks(k)`
  * (`Source`, `Sink`), so a spec gives them fixed or random gaps and stalls.
  *
  * Every cycle the streamer's and the core's signals are poked first, then what fires is read, then the clock steps.
  */
class ShellDriver(
  dut:      ShellHarness,
  sources:  Seq[Source],
  sinks:    Seq[Sink],
  rspReady: Int => Boolean = _ => true
) {

  private val io = dut.io

  require(sources.size == io.readers.size && sinks.size == io.writers.size, "one source per input, one sink per output")

  var cycle: Int = 0

  def taken:   Seq[Int]         = sources.map(_.taken)
  def outputs: Seq[Seq[BigInt]] = sinks.map(_.got)

  /** The cycles a request was offered and not taken. */
  var held: Seq[Int] = Seq.empty

  /** The last cycle an output beat fired. */
  var lastOut: Int = -1

  private def b(name: String): Bool = io.bool(name)
  private def u(name: String): UInt = io.uint(name)

  b("csr_req_valid_i").poke(false.B)

  /** One cycle with the request `req` (addr, data, write) offered, or none: whether it was taken, and the response data
    * when one came.
    */
  def tick(req: Option[(Int, BigInt, Boolean)] = None): (Boolean, Option[BigInt]) = {
    assert(cycle < ShellDriver.MaxCycles, s"no end after $cycle cycles")
    for ((src, k) <- sources.zipWithIndex) {
      src.cycle()
      b(s"stream2acc_${k}_valid_i").poke(src.valid.B)
      u(s"stream2acc_${k}_data_i").poke(src.bits.U)
    }
    for ((snk, k) <- sinks.zipWithIndex) b(s"acc2stream_${k}_ready_i").poke(snk.cycle().B)
    val rspTaken = rspReady(cycle)
    b("csr_rsp_ready_i").poke(rspTaken.B)
    req match {
      case Some((addr, d, write)) =>
        b("csr_req_valid_i").poke(true.B)
        u("csr_req_addr_i").poke(addr.U)
        u("csr_req_data_i").poke(d.U)
        b("csr_req_write_i").poke(write.B)
      case None                   =>
        b("csr_req_valid_i").poke(false.B)
    }
    val accepted = req.isDefined && b("csr_req_ready_o").peek().litToBoolean
    if (req.isDefined && !accepted) held :+= cycle
    val rsp =
      if (rspTaken && b("csr_rsp_valid_o").peek().litToBoolean) Some(u("csr_rsp_data_o").peek().litValue) else None
    for ((src, k) <- sources.zipWithIndex) src.fired(b(s"stream2acc_${k}_ready_o").peek().litToBoolean)
    for ((snk, k) <- sinks.zipWithIndex)
      if (snk.fired(b(s"acc2stream_${k}_valid_o").peek().litToBoolean, u(s"acc2stream_${k}_data_o").peek().litValue))
        lastOut = cycle
    dut.clock.step()
    cycle += 1
    (accepted, rsp)
  }

  /** A write, held until taken: the cycle it was taken. */
  def write(addr: Int, value: BigInt): Int = {
    while (!tick(Some((addr, value, true)))._1) {}
    cycle - 1
  }

  /** A read: its value, answered in the cycle it is taken. */
  def read(addr: Int): BigInt = {
    var got: Option[(Boolean, Option[BigInt])] = None
    while (got.forall(!_._1)) got = Some(tick(Some((addr, 0, false))))
    got.get._2.getOrElse(throw new AssertionError(s"cycle ${cycle - 1}: read $addr taken but not answered"))
  }

  /** Cycles with no request. */
  def idle(n: Int): Unit = for (_ <- 0 until n) tick()
}

object ShellDriver {

  /** A guard against a run that never ends. */
  val MaxCycles = 100000

  /** Input k offers beat t of `beats(k)` (`data(k, t)`) from the first cycle `valid(cycle)` holds; each output is ready
    * in the cycles `ready(cycle)` holds.
    */
  def apply(
    dut:   ShellHarness,
    beats: Seq[Int],
    data:  (Int, Int) => BigInt,
    valid: Int => Boolean = _ => true,
    ready: Int => Boolean = _ => true
  ): ShellDriver = {
    var d: ShellDriver = null
    val sources = beats.zipWithIndex.map { case (n, k) => new Source(n, data(k, _), () => valid(d.cycle)) }
    val sinks   = dut.io.writers.map(_ => new Sink(() => ready(d.cycle)))
    d = new ShellDriver(dut, sources, sinks)
    d
  }
}
