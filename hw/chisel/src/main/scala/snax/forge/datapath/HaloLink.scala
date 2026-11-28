package snax.forge.datapath

import chisel3._

import snax.forge.IntType

/** A halo link (the catalogue entry `HaloLink` of CONTRACTS, "Accelerator RTL spec"): one stream's beats as `taps`
  * windows of a row, shifted by 0 … T − 1 elements, for a unit whose T inputs read neighbouring elements that start
  * mid-word (a packed stream starts on a word). Data and enables only.
  *
  * It holds H = ⌈(T − 1) / W⌉ beats. Number the elements of the held beats, oldest first, then the incoming beat: e_0,
  * e_1, …, e_((H+1)W−1). Tap t, lane l is e_(l+t): a wire from a held register or from the input, so the taps read `in`
  * in the cycle it comes (latency 0, as a link is). On `load` the held beats shift by one, the oldest dropped and `in`
  * the newest held. Counting the fill is its group's control unit's: a job's first H beats only fill, and each later
  * beat gives one output, so a job gives beats in − H outputs. SNAX-GVSoC's `SnaxHaloLink` (`snax/snax_datapath.hpp`)
  * is the same block.
  *
  * @param taps
  *   T, the windows it gives, at least 2
  * @param lanes
  *   W, elements per beat, the link's ends'
  * @param dtype
  *   the elements' type, the link's ends'
  */
class HaloLink(val taps: Int = 3, val lanes: Int = 8, val dtype: String = "uint8") extends DatapathModule {
  require(taps >= 2, s"taps must be at least 2 (a window and one shifted by an element), got $taps")
  require(lanes > 0, s"lanes must be at least 1, got $lanes")

  private val t = IntType(dtype)

  /** H, the beats it holds. */
  val held: Int = (taps - 1 + lanes - 1) / lanes

  override def desiredName: String = s"HaloLink_${t.tag}_n${lanes}_t$taps"

  val io = IO(new Bundle {
    val in   = Input(Vec(lanes, t.lane))
    val taps = Output(Vec(HaloLink.this.taps, Vec(lanes, t.lane)))
    val load = Input(Bool())
  })

  // The held beats, oldest first
  private val beats = RegInit(VecInit(Seq.fill(held)(VecInit(Seq.fill(lanes)(0.U(t.width.W))))))

  when(io.load) {
    for (h <- 0 until held - 1) beats(h) := beats(h + 1)
    beats(held - 1) := VecInit(io.in.map(_.asUInt))
  }

  // NB (D206): e_0 … e_(H+1)W-1 are the held beats' elements oldest first, then the input's; tap t lane l is e_(l+t),
  // as SNAX-GVSoC's SnaxHaloLink cuts them
  private val e = beats.flatten ++ io.in.map(_.asUInt)
  for {
    k <- 0 until taps
    l <- 0 until lanes
  } io.taps(k)(l) := IntType.cast(e(l + k), t, t)
}
