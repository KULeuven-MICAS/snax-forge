package snax.forge.ctrl

import org.scalatest.flatspec.AnyFlatSpec

/** `GroupDesc` over edges: its topological order, head and owner, and the graphs it refuses with SnaxGroupRule's
  * reasons (a fork, a join, a cycle, two chains) or snax_group.cpp's (a link on an outer input, two links on one input,
  * an input left to a link that none feeds).
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

  private def refused(units: Seq[UnitDesc], links: Seq[Link], why: String): Unit = {
    val e = intercept[IllegalArgumentException](GroupDesc("g", units, links))
    assert(e.getMessage.contains(why), e.getMessage)
  }

  private val three = Seq(unit("a", 1), unit("b", 0, Seq(0)), unit("c", 0, Seq(0)))

  it should "refuse a fork, a join, a cycle and two chains" in {
    refused(
      three,
      Seq(Link(0, 1, 0, 0), Link(0, 2, 0, 0)),
      "unit 0 has 2 out-edges (a fork): until tier A a group is a chain"
    )
    refused(
      Seq(unit("a", 1), unit("b", 1), unit("c", 0, Seq(0, 1))),
      Seq(Link(0, 2, 0, 0), Link(1, 2, 1, 0)),
      "unit 2 has 2 in-edges (a join): until tier A a group is a chain"
    )
    refused(
      Seq(unit("a", 1, Seq(1)), unit("b", 0, Seq(0))),
      Seq(Link(0, 1, 0, 0), Link(1, 0, 1, 0)),
      "the links form a cycle"
    )
    refused(
      Seq(unit("a", 1), unit("b", 0, Seq(0)), unit("c", 1), unit("d", 0, Seq(0))),
      Seq(Link(0, 1, 0, 0), Link(2, 3, 0, 0)),
      "2 units have no out-edge: until tier A a group is one chain"
    )
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
