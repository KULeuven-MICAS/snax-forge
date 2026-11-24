"""run/convert.py on snippets and on a synthetic run: names, the window, the trace, the
profile, the invariants."""

import pytest

from snax_forge.run import RunError, parse_line
from snax_forge.run.convert import convert, port_of, source_of

from .helpers import line, synthetic


def parse(lines):
    return [parse_line(x) for x in lines]


@pytest.mark.parametrize(
    "path, text",
    [("snax_acc1/unit", "5 start count=1 rate=1"), ("pe2", "5 state mode=run"),
     ("snax_acc0/streamer_acc", "5 start mover=reader7")],
)  # fmt: skip
def test_a_path_not_of_the_cluster_is_named(cluster, path, text):
    with pytest.raises(RunError) as e:
        source_of(parse_line(line(path, "events", text)), cluster)
    assert e.value.code in ("convert.path", "cluster.mover")


def test_crossbar_inputs_become_ports(cluster):
    assert port_of("snax_acc0.streamer_acc.reader1.lane3", cluster) == "acc_b.lane3"
    assert port_of("snax_acc0.streamer_acc.writer0.lane0", cluster) == "acc_out.lane0"
    assert [port_of(x, cluster) for x in ("core0", "core1", "axi")] == ["core0", "core1", "axi"]
    with pytest.raises(RunError):
        port_of("core5", cluster)


@pytest.mark.parametrize(
    ("edit", "match", "code"),
    [
        (lambda ls: [x for x in ls if "forge_check" not in x], "forge_check", "convert.window"),
        (lambda ls: [x.replace("src=0x80000000", "src=0x10010000") for x in ls], "load_A",
         "convert.tasks"),
        (lambda ls: [x for x in ls if "done mover=writer0" not in x],
         "acc_out: 1 starts and 0 dones", "convert.invariant"),
        (lambda ls: [x.replace("done firings=16", "done firings=15") for x in ls],
         "15 firings in its done events, 16 in its summary", "convert.invariant"),
    ],
    ids=["no forge_check", "copy against its direction", "start without done", "firings"],
)  # fmt: skip
def test_a_broken_run_is_named(cluster, tasks, edit, match, code):
    with pytest.raises(RunError, match=match) as e:
        convert(parse(edit(synthetic(cluster, tasks))), cluster, tasks, 226, "task")
    assert e.value.code == code


@pytest.mark.parametrize("cycles", [225, 227])
def test_forge_cycles_must_be_the_reads_apart(cluster, tasks, cycles):
    """The two mcycle reads are 226 cycles apart: any other forge cycles is a named error."""
    with pytest.raises(RunError, match="226 cycles apart") as e:
        convert(parse(synthetic(cluster, tasks)), cluster, tasks, cycles, "task")
    assert e.value.code == "convert.window"
