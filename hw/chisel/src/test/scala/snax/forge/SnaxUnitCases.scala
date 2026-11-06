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
    name:    String,
    latency: Int,
    ii:      Int,
    inputs:  Int,
    lanes:   Int,
    types:   Seq[String],
    op:      String,
    k:       Option[BigInt], // one input's constant: a one-stream SimdLanes (relu's)
    k1:      Option[Slot],   // axpby's (op axpby, axpy's): k1 required, k2 optional
    k2:      Option[Slot],
    starts:  Map[Int, Start],
    runs:    Seq[RunLine],
    beats:   Map[Int, Seq[BigInt]],
    results: Map[Int, Seq[BigInt]],
    cycles:  Seq[Cycle]
  )

  def parse(path: Path): Case = {
    val (header, rows) = CaseFile.read(path)
    var latency        = 0
    var ii             = 1
    var inputs         = 2
    var lanes          = 1
    var types          = Seq.empty[String]
    var op             = "add"
    var k              = Option.empty[BigInt]
    var k1             = Option.empty[Slot]
    var k2             = Option.empty[Slot]
    var starts         = Map.empty[Int, Start]
    var runs           = Seq.empty[RunLine]
    var beats          = Map.empty[Int, Seq[BigInt]]
    var results        = Map.empty[Int, Seq[BigInt]]
    for (line <- header) line.split("\\s+").toList match {
      case "latency" :: n :: Nil                       => latency = n.toInt
      case "ii" :: n :: Nil                            => ii      = n.toInt
      case "inputs" :: n :: Nil                        => inputs  = n.toInt
      case "lanes" :: n :: Nil                         => lanes   = n.toInt
      case "types" :: ts if ts.nonEmpty                => types   = ts
      case "op" :: o :: Nil                            => op      = o
      case "k" :: v :: Nil                             => k       = Some(BigInt(v))
      case "k1" :: v :: from                           => k1      = Some(slot(path, line, v, from))
      case "k2" :: v :: from                           => k2      = Some(slot(path, line, v, from))
      case "start" :: c :: j :: t :: a :: Nil          =>
        starts += c.toInt -> Start(j.toInt, t.toInt, a)
      case "expect" :: "run" :: RunLine.Fields(i, run) =>
        require(i == runs.size, s"$path: `$line` is not run ${runs.size}")
        runs :+= run
      case "beat" :: n :: vs if vs.nonEmpty            => beats += n.toInt -> vs.map(CaseFile.number)
      case "result" :: n :: vs if vs.nonEmpty          =>
        results += n.toInt -> vs.map(CaseFile.number)
      case _ => throw new IllegalArgumentException(s"$path: `$line` is not a header line")
    }
    require(types.isEmpty || types.size == inputs + 1, s"$path: types gives ${types.size}, not ${inputs + 1}")
    require(k.isEmpty || inputs == 1, s"$path: k is one input's constant; inputs is $inputs")
    require(
      (op == "axpby") == k1.nonEmpty && (k2.isEmpty || k1.nonEmpty) && (op != "axpby" || inputs == 2),
      s"$path: op axpby takes two inputs and k1 (k2 optional), and only it"
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
      k,
      k1,
      k2,
      starts,
      runs,
      beats,
      results,
      cycles
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
