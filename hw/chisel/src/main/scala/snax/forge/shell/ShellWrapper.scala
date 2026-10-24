package snax.forge.shell

import scala.collection.immutable.SeqMap

import chisel3._
import chisel3.util._

import snax.forge.Blocks

/** The wrapper's ports but the clock and reset, named as `snax_acc_wrapper.sv.tpl` connects `<tag>_shell_wrapper` with
  * `snax_disable_csr_manager` (C§17): the CSR request and response after `snax_csr_mux_demux`, then per input port k
  * `stream2acc_<k>_{data_i,valid_i,ready_o}` and per output port `acc2stream_<k>_{data_o,valid_o,ready_i}`.
  *
  * @param readers
  *   the input ports' data widths, in order
  * @param writers
  *   the output ports' data widths, in order
  */
class ShellPorts(val readers: Seq[Int], val writers: Seq[Int]) extends Record {
  val elements: SeqMap[String, Data] = SeqMap[String, Data](
    "csr_req_addr_i"  -> Input(UInt(32.W)),
    "csr_req_data_i"  -> Input(UInt(32.W)),
    "csr_req_write_i" -> Input(Bool()),
    "csr_req_valid_i" -> Input(Bool()),
    "csr_req_ready_o" -> Output(Bool()),
    "csr_rsp_data_o"  -> Output(UInt(32.W)),
    "csr_rsp_valid_o" -> Output(Bool()),
    "csr_rsp_ready_i" -> Input(Bool())
  ) ++ readers.zipWithIndex.flatMap { case (w, k) =>
    Seq(
      s"stream2acc_${k}_data_i"  -> Input(UInt(w.W)),
      s"stream2acc_${k}_valid_i" -> Input(Bool()),
      s"stream2acc_${k}_ready_o" -> Output(Bool())
    )
  } ++ writers.zipWithIndex.flatMap { case (w, k) =>
    Seq(
      s"acc2stream_${k}_data_o"  -> Output(UInt(w.W)),
      s"acc2stream_${k}_valid_o" -> Output(Bool()),
      s"acc2stream_${k}_ready_i" -> Input(Bool())
    )
  }

  def apply(name: String): Data = elements(name)
  def bool(name:  String): Bool = elements(name).asInstanceOf[Bool]
  def uint(name:  String): UInt = elements(name).asInstanceOf[UInt]
}

/** `<tag>_shell_wrapper`: one generated accelerator as snax_cluster instantiates it (D130, C§17), from `rtl.json`.
  *
  * Inside: `CsrManager` (the fields of `rtl.json`'s `rw`, start, busy and the counter read-only), `CtrlAdapter` (`ctrl`
  * to the block's start and configuration pins, the counter, the ties) and the unit, a catalogue block (`Blocks`) built
  * from its params. Its data ports follow `ports`: the block's input ports, in order, are `stream2acc_0`,
  * `stream2acc_1`, ...; its output ports `acc2stream_0`, ... . A port's data is the block's `bits` as `asUInt`, so lane
  * i of a `Vec` sits at bits `[w(i+1)-1 : wi]`: `[64i+63 : 64i]` for 64-bit lanes. That this is the streamer's packing
  * too is checked on snax_cluster's RTL (D123); here the specs assume it.
  *
  * Clock `clk_i`, reset `rst_ni` active low and asynchronous, as snax_cluster's (`RequireAsyncReset` in ReqRspManager).
  * One unit is built (`rtl.json` with more is refused by the Python side, `gen.instances`).
  */
class ShellWrapper(spec: RtlSpec) extends RawModule {
  require(spec.units.size == 1, s"one unit per accelerator, got ${spec.units.size}")
  private val unit  = spec.units.head
  private val block = Blocks(unit.block)

  override def desiredName: String = s"${spec.tag}_shell_wrapper"

  val clk_i  = IO(Input(Clock()))
  val rst_ni = IO(Input(Bool()))

  private val dirs = block.ports.toMap

  /** The block's port of each BRM port, inputs first then outputs, each in the BRM's order. */
  private val ins  = unit.ports.filter { case (_, b) => dirs(b) == "in" }.map(_._2)
  private val outs = unit.ports.filter { case (_, b) => dirs(b) == "out" }.map(_._2)

  private val built = withClockAndReset(clk_i, (!rst_ni).asAsyncReset) {
    block.build(Blocks.Params(unit.params))
  }
  private def pin(name:  String): Data              = built.io.elements(name)
  private def port(name: String): DecoupledIO[Data] = pin(name).asInstanceOf[DecoupledIO[Data]]

  /** The unit's busy pin, the busy CSR's source: a handle a test harness can tap (`BoringUtils.bore`). A `def`, so it
    * names nothing and adds no port: the emitted RTL is the same with or without it.
    */
  def unitBusy: Bool = pin(block.busy).asInstanceOf[Bool]

  val io = FlatIO(new ShellPorts(ins.map(port(_).bits.getWidth), outs.map(port(_).bits.getWidth)))

  withClockAndReset(clk_i, (!rst_ni).asAsyncReset) {
    val manager = Module(new CsrManager(unit.rw, numRo = 2, spec.tag))
    val ties    = unit.tie.map { case (n, v) => (n, block.ties.find(_.pin == n).get.width, v) }
    val adapter = Module(new CtrlAdapter(unit.rw, ties, spec.tag))

    manager.io.req.bits.addr  := io.uint("csr_req_addr_i")
    manager.io.req.bits.data  := io.uint("csr_req_data_i")
    manager.io.req.bits.write := io.bool("csr_req_write_i")
    manager.io.req.valid      := io.bool("csr_req_valid_i")
    io("csr_req_ready_o")     := manager.io.req.ready
    io("csr_rsp_data_o")      := manager.io.rsp.bits.data
    io("csr_rsp_valid_o")     := manager.io.rsp.valid
    manager.io.rsp.ready      := io.bool("csr_rsp_ready_i")

    adapter.io.ctrl <> manager.io.ctrl
    manager.io.ro := adapter.io.ro

    pin(block.start) := adapter.io.start
    unit.rw.foreach { case (n, _) => pin(n) := adapter.io.config(n) }
    unit.tie.foreach { case (n, _) => pin(n) := adapter.io.ties(n) }
    adapter.io.busy  := pin(block.busy)

    ins.zipWithIndex.foreach { case (b, k) =>
      val p = port(b)
      p.valid                        := io.bool(s"stream2acc_${k}_valid_i")
      p.bits                         := io.uint(s"stream2acc_${k}_data_i").asTypeOf(p.bits)
      io(s"stream2acc_${k}_ready_o") := p.ready
    }
    outs.zipWithIndex.foreach { case (b, k) =>
      val p = port(b)
      io(s"acc2stream_${k}_data_o")  := p.bits.asUInt
      io(s"acc2stream_${k}_valid_o") := p.valid
      p.ready                        := io.bool(s"acc2stream_${k}_ready_i")
    }
  }
}
