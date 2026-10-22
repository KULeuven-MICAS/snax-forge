package snax.forge.shell

import chisel3._
import chiseltest._

/** `<tag>_shell_wrapper` under chiseltest, which drives a `Module`: the clock and an active-high reset become `clk_i`
  * and `rst_ni`, and the wrapper's other ports are this harness's `io`, with the same names.
  */
class ShellHarness(spec: RtlSpec) extends Module {
  private val wrapper = Module(new ShellWrapper(spec))
  wrapper.clk_i  := clock
  wrapper.rst_ni := !reset.asBool
  val io = IO(chiselTypeOf(wrapper.io))
  io <> wrapper.io
}

/** A core and a streamer around a `ShellHarness`, one call per cycle.
  *
  * The core: one CSR request at a time, held until `csr_req_ready_o` (as the Snitch core's write is), the response
  * always taken (`csr_rsp_ready_i` high). The streamer: input k offers beat t of `beats(k)` (`data(k, t)`, valid while
  * `valid(cycle)` and beats are left), each output always ready unless `ready(cycle)` says otherwise, its beats kept in
  * `outputs`.
  *
  * Every cycle the streamer's and the core's signals are poked first, then what fires is read, then the clock steps.
  */
class ShellDriver(
  dut:   ShellHarness,
  beats: Seq[Int],
  data:  (Int, Int) => BigInt,
  valid: Int => Boolean = _ => true,
  ready: Int => Boolean = _ => true
) {

  private val io   = dut.io
  private val ins  = io.readers.size
  private val outs = io.writers.size

  var cycle:   Int                = 0
  val taken:   Array[Int]         = Array.fill(ins)(0)
  val outputs: Array[Seq[BigInt]] = Array.fill(outs)(Seq.empty)

  /** The cycles a request was offered and not taken. */
  var held: Seq[Int] = Seq.empty

  private def b(name: String): Bool = io.bool(name)
  private def u(name: String): UInt = io.uint(name)

  b("csr_rsp_ready_i").poke(true.B)
  b("csr_req_valid_i").poke(false.B)

  /** One cycle with the request `req` (addr, data, write) offered, or none: whether it was taken, and the response data
    * when one came.
    */
  def tick(req: Option[(Int, BigInt, Boolean)] = None): (Boolean, Option[BigInt]) = {
    for (k <- 0 until ins) {
      val v = valid(cycle) && taken(k) < beats(k)
      b(s"stream2acc_${k}_valid_i").poke(v.B)
      u(s"stream2acc_${k}_data_i").poke(if (v) data(k, taken(k)).U else 0.U)
    }
    for (k <- 0 until outs) b(s"acc2stream_${k}_ready_i").poke(ready(cycle).B)
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
    val rsp = if (b("csr_rsp_valid_o").peek().litToBoolean) Some(u("csr_rsp_data_o").peek().litValue) else None
    for (k <- 0 until ins)
      if (b(s"stream2acc_${k}_valid_i").peek().litToBoolean && b(s"stream2acc_${k}_ready_o").peek().litToBoolean)
        taken(k) += 1
    for (k <- 0 until outs)
      if (b(s"acc2stream_${k}_valid_o").peek().litToBoolean && ready(cycle))
        outputs(k) :+= u(s"acc2stream_${k}_data_o").peek().litValue
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
