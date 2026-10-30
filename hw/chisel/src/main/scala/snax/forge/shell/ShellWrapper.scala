package snax.forge.shell

import scala.collection.immutable.SeqMap

import chisel3._

/** The wrapper's ports but the clock and reset, named as `snax_acc_wrapper.sv.tpl` connects `<tag>_shell_wrapper` with
  * `snax_disable_csr_manager`: the CSR request and response after `snax_csr_mux_demux`, then per input port k
  * `stream2acc_<k>_{data_i,valid_i,ready_o}` and per output port `acc2stream_<k>_{data_o,valid_o,ready_i}`; then per
  * group `<group>_busy_o` and `<group>_done_o` (D146), which snax_cluster leaves unconnected (open item 92).
  *
  * @param readers
  *   the input ports' data widths, in order
  * @param writers
  *   the output ports' data widths, in order
  * @param groups
  *   the groups' names, in order
  */
class ShellPorts(val readers: Seq[Int], val writers: Seq[Int], val groups: Seq[String]) extends Record {
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
  } ++ groups.flatMap(g => Seq(s"${g}_busy_o" -> Output(Bool()), s"${g}_done_o" -> Output(Bool())))

  def apply(name: String): Data = elements(name)
  def bool(name:  String): Bool = elements(name).asInstanceOf[Bool]
  def uint(name:  String): UInt = elements(name).asInstanceOf[UInt]
}

/** `<tag>_shell_wrapper`: one generated accelerator as snax_cluster instantiates it (C§17), from `accelerator.json`.
  *
  * Three modules inside (D147): `<tag>_csr` (`AccCsr`, the CSR manager over the file's CSRs), `<tag>_ctrl` (`AccCtrl`,
  * a control unit per group) and `<tag>_datapath` (`AccDatapath`, the units' datapaths and the links' buffers). Only
  * the outer ports reach the wrapper (a link stays inside the datapath): each, `stream2acc_<k>` for the k-th entry of
  * the file's `inputs` and `acc2stream_<k>` for its `outputs`, is split, its data to its unit's datapath, its valid and
  * ready to its group's control unit. The control unit's enables go to the datapaths' pins (`en`, a plain Mac's output
  * register, is the unit's firing) and the buffers'. A port's data is its lanes side by side, lane i at bits `[w(i+1)-1
  * : wi]` for the port's w-bit dtype (num_slc = 64 / w lanes a word, D142), at least one 64-bit word (a fold's one-lane
  * output); the streamer's packing, checked on snax_cluster's RTL (D123).
  *
  * Clock `clk_i`, reset `rst_ni` active low and asynchronous, as snax_cluster's (`RequireAsyncReset` in ReqRspManager).
  */
class ShellWrapper(spec: AcceleratorSpec) extends RawModule {
  override def desiredName: String = s"${spec.tag}_shell_wrapper"

  val clk_i  = IO(Input(Clock()))
  val rst_ni = IO(Input(Bool()))

  /** A port's data width: its lanes at its dtype's width, at least one 64-bit word. */
  private def width(unit: String, brm: String): Int = {
    val p = spec.unit(unit).port(brm)
    math.max(64, p.lanes * snax.forge.IntType(p.dtype).width)
  }

  val io = FlatIO(
    new ShellPorts(
      spec.inputs.map { case (u, p) => width(u, p) },
      spec.outputs.map { case (u, p) => width(u, p) },
      spec.groups.map(_.name)
    )
  )

  withClockAndReset(clk_i, (!rst_ni).asAsyncReset) {
    val csr  = Module(new AccCsr(spec))
    val ctrl = Module(new AccCtrl(spec))
    val dp   = Module(new AccDatapath(spec))

    csr.io.req.bits.addr  := io.uint("csr_req_addr_i")
    csr.io.req.bits.data  := io.uint("csr_req_data_i")
    csr.io.req.bits.write := io.bool("csr_req_write_i")
    csr.io.req.valid      := io.bool("csr_req_valid_i")
    io("csr_req_ready_o") := csr.io.req.ready
    io("csr_rsp_data_o")  := csr.io.rsp.bits.data
    io("csr_rsp_valid_o") := csr.io.rsp.valid
    csr.io.rsp.ready      := io.bool("csr_rsp_ready_i")

    for ((g, i) <- spec.groups.zipWithIndex) {
      val c = ctrl.io.groups(i)
      c.start <> csr.io.starts(i)
      csr.io.status(i)        := c.status
      csr.io.counter(i)       := c.counter
      io(s"${g.name}_busy_o") := c.busy
      io(s"${g.name}_done_o") := c.done

      // The outer ports, unit by unit in the streamer's order: data to the unit's datapath, handshake to the group's
      // control unit (its inputs flat in unit order, its output its owner's)
      val desc = AccCtrl.desc(spec, g)
      var at   = 0
      for ((u, k) <- g.units.zipWithIndex) {
        val d = dp.io.units(spec.units.indexOf(u))
        for ((j, brm)    <- spec.ins(u.name)) {
          d(brm)                         := io.uint(s"stream2acc_${j}_data_i").asTypeOf(d(brm))
          c.in(at).valid                 := io.bool(s"stream2acc_${j}_valid_i")
          io(s"stream2acc_${j}_ready_o") := c.in(at).ready
          at += 1
        }
        for ((j, brm)    <- spec.outs(u.name)) {
          require(k == desc.output, s"${u.name}.$brm: an outer output of a unit that has an out-edge")
          io(s"acc2stream_${j}_data_o")  := d(brm).asUInt
          io(s"acc2stream_${j}_valid_o") := c.out.valid
          c.out.ready                    := io.bool(s"acc2stream_${j}_ready_i")
        }
        val e = c.units(k)
        for ((name, pin) <- d.elements if !u.ports.exists(_._1 == name))
          pin := (name match {
            case "fire" | "en" => e.fire
            case "first"       => e.first
            case "read"        => e.read
            case "lane"        => e.lane
            case other         => throw new IllegalArgumentException(s"${u.datapath}: unknown pin $other")
          })
      }

      // The buffers' pins, from the group's control unit (its buffers in link order, as GroupDesc.buffers)
      for ((l, n) <- g.links.filter(_.depth > 0).zipWithIndex) {
        val b = dp.io.buffers(dp.buffered(l.name))
        val e = c.links(n)
        for ((name, pin) <- b.elements)
          pin := (name match {
            case "load"  => e.load
            case "wslot" => e.wslot
            case "rslot" => e.rslot
            case other   => throw new IllegalArgumentException(s"${l.name}: unknown pin $other")
          })
      }
    }
  }
}
