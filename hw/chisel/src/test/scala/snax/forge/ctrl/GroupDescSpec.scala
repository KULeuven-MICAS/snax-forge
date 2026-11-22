package snax.forge.ctrl

import org.scalatest.flatspec.AnyFlatSpec

/** `GroupDesc` over edges: its topological order, head and owner, two chains on a fork (D189) with their outputs and
  * chains, and the graphs it refuses with SnaxGroupRule's reasons (a fork or a join of links, a cycle, two chains no
  * fork joins, a fork onto a unit a link feeds or of one unit) or snax_group.cpp's (a link on an outer input, two links
  * on one input, an input left to a link that none feeds).
  */
class GroupDescSpec extends AnyFlatSpec {

  private def unit(name: String, inputs: Int, linked: Seq[Int] = Nil) = UnitDesc(name, 1, 1, 1, inputs, Some(1), linked)

  behavior of "GroupDesc"

  it should "order a chain listed out of order, head first, and own the output at the sink" in {
    // u1 -> u2 -> u0
    val d = GroupDesc(
      "g",
      Seq(unit("u0", 0, Seq(0)), unit("u1", 2), unit("u2", 1, Seq(1))),
      Seq(Link(2, 0, 0, 0), Link(1, 2, 1, 1))
    )
    assert(d.order == Seq(1, 2, 0))
    assert(d.head == 1                       && d.output == 0)
    assert(d.ins == Seq(Seq(0), Nil, Seq(1)) && d.outs == Seq(Nil, Seq(1), Seq(0)))
    assert(d.buffers == Seq(1))
  }

  it should "give bicg's one pass two outputs, a chain each, joined by its fork" in {
    // mul -> sum, and mac; mul.a and mac.a forked
    val d =
      GroupDesc("g", Seq(unit("mul", 2), unit("sum", 0, Seq(0)), unit("mac", 2)), Seq(Link(0, 1, 0, 1)), Seq(Seq(0, 2)))
    assert(d.order == Seq(0, 2, 1) && d.head == 0)
    assert(d.outputs == Seq(1, 2)  && d.output == 1)
    assert(d.chain == Seq(0, 0, 1))
  }

  private def refused(units: Seq[UnitDesc], links: Seq[Link], why: String, forks: Seq[Seq[Int]] = Nil): Unit = {
    val e = intercept[IllegalArgumentException](GroupDesc("g", units, links, forks))
    assert(e.getMessage.contains(why), e.getMessage)
  }

  private val three = Seq(unit("a", 1), unit("b", 0, Seq(0)), unit("c", 0, Seq(0)))

  it should "refuse a fork, a join, a cycle and two chains" in {
    refused(
      three,
      Seq(Link(0, 1, 0, 0), Link(0, 2, 0, 0)),
      "unit 0 has 2 out-edges (a fork): a chain has neither"
    )
    refused(
      Seq(unit("a", 1), unit("b", 1), unit("c", 0, Seq(0, 1))),
      Seq(Link(0, 2, 0, 0), Link(1, 2, 1, 0)),
      "unit 2 has 2 in-edges (a join): a chain has neither"
    )
    refused(
      Seq(unit("a", 1, Seq(1)), unit("b", 0, Seq(0))),
      Seq(Link(0, 1, 0, 0), Link(1, 0, 1, 0)),
      "the links form a cycle"
    )
    refused(
      Seq(unit("a", 1), unit("b", 0, Seq(0)), unit("c", 1), unit("d", 0, Seq(0))),
      Seq(Link(0, 1, 0, 0), Link(2, 3, 0, 0)),
      "the chains ending at b and d: no fork joins them; a group's chains share an input"
    )
  }

  it should "refuse a fork onto a unit a link feeds, of one unit or of a unit of span 2" in {
    val two  = Seq(unit("a", 1), unit("b", 1, Seq(1)), unit("c", 1))
    refused(two, Seq(Link(0, 1, 1, 0)), "fork: unit 1 is not a chain's head of span 1", Seq(Seq(2, 1)))
    refused(two, Seq(Link(0, 1, 1, 0)), "a fork gives one input to two or more units", Seq(Seq(0)))
    val wide = Seq(unit("a", 1), unit("b", 0, Seq(0)), UnitDesc("c", 1, 1, 2, 1, Some(1)))
    refused(wide, Seq(Link(0, 1, 0, 0)), "fork: unit 2 is not a chain's head of span 1", Seq(Seq(0, 2)))
  }

  it should "refuse a link to no unit, on an outer input or on an input another link feeds, and an unfed input" in {
    refused(three, Seq(Link(0, 3, 0, 0)), "link 0: no such unit, or a depth below 0")
    refused(three, Seq(Link(0, 1, 0, -1)), "link 0: no such unit, or a depth below 0")
    refused(
      Seq(unit("a", 1), unit("b", 1)),
      Seq(Link(0, 1, 0, 0)),
      "link 0: b's input 0 is not an input left to a link"
    )
    refused(
      three,
      Seq(Link(0, 1, 0, 0), Link(2, 1, 0, 0)),
      "link 1: b's input 0 is not an input left to a link, or another link feeds it"
    )
    refused(Seq(unit("a", 1), unit("b", 0, Seq(0))), Nil, "b's input 0: no link feeds it")
  }
}
