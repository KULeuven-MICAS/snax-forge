package snax.forge

import java.nio.file.Path

/** snax-gvsoc's `SnaxUnit` cases, the hand-worked half of the unit cross-check between snax-gvsoc and Chisel.
  *
  * The case files (`tests/snax/unit/cases/`, one `.txt` each, in the format of that folder's README: header lines, then
  * `cycles` and one line per cycle) are worked by hand from `SnaxUnit`'s rule (`snax/snax_unit.hpp`); snax-gvsoc's
  * `tb.cpp` checks the rule against them, and the Chisel specs drive a group controller with its datapath through the
  * same cases. They are read from the folder `SNAX_GVSOC_UNIT_CASES` names (e.g.
  * `gvsoc/snax-gvsoc/tests/snax/unit/cases`); without it a spec cancels (the submodule not checked out). CI's chisel
  * job sets it to the submodule's cases.
  */
object SnaxUnitCases {

  val Env = "SNAX_GVSOC_UNIT_CASES"

  /** A start offered before a cycle's step: jobs, job_beats and the answer the case expects. */
  case class Start(jobs: Int, jobBeats: Int, answer: String)

  /** One cycle line: each input's valid, the output's ready, then the expected fire, push and busy. */
  case class Cycle(valids: Seq[Boolean], ready: Boolean, fire: Boolean, push: Boolean, busy: Boolean)

  /** axpby's slot: its value, and whether it is read from its CSR (the pin) or baked. */
  case class Slot(value: BigInt, csr: Boolean)

  case class Case(
    name:     String,
    latency:  Int,
    ii:       Int,
    inputs:   Int,
    lanes:    Int,
    types:    Seq[String],
    op:       String,
    laneFold: Boolean, // `fold lanes`: a one-input fold whose lanes keep their own sums
    k:        Option[BigInt], // one input's constant: a one-stream SimdLanes (relu's), or threshold's k
    k1:       Option[Slot], // axpby's (op axpby, axpy's): k1 required, k2 optional; axpbypcz's
    k2:       Option[Slot],
    starts:   Map[Int, Start],
    runs:     Seq[RunLine],
    beats:    Map[Int, Seq[BigInt]],
    results:  Map[Int, Seq[BigInt]],
    cycles:   Seq[Cycle],
    kCsr:     Boolean        = false, // k read from its CSR (`k V csr`, threshold's)
    k3:       Option[Slot]   = None,  // axpbypcz's, as k1 and k2
    d:        Option[BigInt] = None,  // axpbypcz's and requant's divisor, a power of two
    lo:       Option[BigInt] = None,  // requant's bounds; hi threshold's too
    hi:       Option[BigInt] = None,
    opType:   Option[String] = None   // `optype`: the op's type, else the inputs' result type
  ) {

    /** One of SimdLanes' ops with slots of their own past axpby: axpbypcz, requant or threshold. */
    def named: Boolean = Seq("axpbypcz", "requant", "threshold").contains(op)
  }

  def parse(path: Path): Case = {
    val (header, rows) = CaseFile.read(path)
    var latency        = 0
    var ii             = 1
    var inputs         = 2
    var lanes          = 1
    var types          = Seq.empty[String]
    var op             = "add"
    var laneFold       = false
    var k              = Option.empty[BigInt]
    var k1             = Option.empty[Slot]
    var k2             = Option.empty[Slot]
    var k3             = Option.empty[Slot]
    var kCsr           = false
    var bounds         = Map.empty[String, BigInt] // d, lo, hi
    var opType         = Option.empty[String]
    var starts         = Map.empty[Int, Start]
    var runs           = Seq.empty[RunLine]
    var beats          = Map.empty[Int, Seq[BigInt]]
    var results        = Map.empty[Int, Seq[BigInt]]
    for (line <- header) line.split("\\s+").toList match {
      case "latency" :: n :: Nil                                   => latency  = n.toInt
      case "ii" :: n :: Nil                                        => ii       = n.toInt
      case "inputs" :: n :: Nil                                    => inputs   = n.toInt
      case "lanes" :: n :: Nil                                     => lanes    = n.toInt
      case "types" :: ts if ts.nonEmpty                            => types    = ts
      case "op" :: o :: Nil                                        => op       = o
      case "fold" :: how :: Nil if how == "tree" || how == "lanes" => laneFold = how == "lanes"
      case "k" :: v :: from                                        =>
        val x = slot(path, line, v, from)
        k    = Some(x.value)
        kCsr = x.csr
      case "k1" :: v :: from                                       => k1       = Some(slot(path, line, v, from))
      case "k2" :: v :: from                                       => k2       = Some(slot(path, line, v, from))
      case "k3" :: v :: from                                       => k3       = Some(slot(path, line, v, from))
      case (b @ ("d" | "lo" | "hi")) :: v :: Nil                   => bounds += b -> CaseFile.number(v)
      case "optype" :: t :: Nil                                    => opType   = Some(t)
      case "start" :: c :: j :: t :: a :: Nil                      =>
        starts += c.toInt -> Start(j.toInt, t.toInt, a)
      case "expect" :: "run" :: RunLine.Fields(i, run)             =>
        require(i == runs.size, s"$path: `$line` is not run ${runs.size}")
        runs :+= run
      case "beat" :: n :: vs if vs.nonEmpty                        => beats += n.toInt -> vs.map(CaseFile.number)
      case "result" :: n :: vs if vs.nonEmpty                      =>
        results += n.toInt -> vs.map(CaseFile.number)
      case _ => throw new IllegalArgumentException(s"$path: `$line` is not a header line")
    }
    require(types.isEmpty || types.size == inputs + 1, s"$path: types gives ${types.size}, not ${inputs + 1}")
    require(k.isEmpty || inputs == 1, s"$path: k is one input's constant; inputs is $inputs")
    require(!laneFold || (inputs == 1 && k.isEmpty), s"$path: fold lanes is a one-input fold, without k")
    require(
      (op != "axpby" || (k1.nonEmpty && inputs == 2)) && (Seq("axpby", "axpbypcz")
        .contains(op) || k1.isEmpty  && k2.isEmpty),
      s"$path: op axpby takes two inputs and k1 (k2 optional); k1 and k2 are axpby's and axpbypcz's"
    )
    require(
      (op == "axpbypcz") == (inputs == 3)  && (op != "axpbypcz" || bounds.keySet == Set("d")) &&
        (k3.isEmpty || op == "axpbypcz"),
      s"$path: op axpbypcz takes three inputs and d (k1, k2, k3 optional), and only it"
    )
    require(
      (op != "requant" || (inputs == 1     && bounds.keySet == Set("d", "lo", "hi")           && k.isEmpty))  &&
        (op != "threshold" || (inputs == 1 && bounds.keySet == Set("hi")                      && k.nonEmpty)) &&
        (Seq("axpbypcz", "requant", "threshold").contains(op) || bounds.isEmpty),
      s"$path: op requant takes one input, d, lo and hi; threshold one input, k and hi"
    )
    val cycles = rows.map { line =>
      val v = line.split("\\s+").map(_ == "1").toSeq
      require(v.size == inputs + 4, s"$path: `$line` has ${v.size} columns, not ${inputs + 4}")
      Cycle(v.take(inputs), v(inputs), v(inputs + 1), v(inputs + 2), v(inputs + 3))
    }
    Case(
      CaseFile.name(path),
      latency,
      ii,
      inputs,
      lanes,
      types,
      op,
      laneFold,
      k,
      k1,
      k2,
      starts,
      runs,
      beats,
      results,
      cycles,
      kCsr,
      k3,
      bounds.get("d"),
      bounds.get("lo"),
      bounds.get("hi"),
      opType
    )
  }

  /** `k1 V [csr]`: the value, baked or read from its CSR. */
  private def slot(path: Path, line: String, v: String, from: List[String]): Slot = {
    require(from == Nil || from == List("csr"), s"$path: `$line`: k1 V [csr]")
    Slot(BigInt(v), from.nonEmpty)
  }

  /** Every case of the folder `Env` names (`CaseFile.all`). */
  def all: Option[Seq[Case]] = CaseFile.all(Env)(parse)
}
