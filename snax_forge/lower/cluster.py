"""Design point -> cluster file (LOW1c, D53, D88).

The cluster file SNAX-MODEL builds (CONTRACTS.md section 2) is derived, never
decided: the platform gives the memories, the xbar, the DMA and the
controller; every accelerator gives its entry and, through the platform's
streamer shell, one streamer per port. The components are listed in this
order, which fixes the scheduler's tick order, the xbar's port order and the
trace's source order (D31, D39):

    xbar
    dma                          when the platform has an L2
    per accelerator, in the order the graph first uses it:
        its streamers            in the BRM's port order, <instance>_<port>
        the accelerator          accel kind and params from the BRM
                                 (Instance.accel_entry), attach port -> streamer
    ctl                          the controller

The register map lists every component but the xbar and the controller, in
that order, with the platform's window; a streamer's spatial bounds are
written only when they are not the default ``[n_ports]``.

``Accel`` is one accelerator with its streamers. ``accel_of`` makes one from
a BRM instance and the resolved shell; ``stub`` makes one from a registered
accelerator kind and its params directly, for a cluster whose accelerator
has no BRM (mul1's multi-cycle ``mul`` stub; red4 comes from ``accumulate``
since BRM4).
``cluster_config`` assembles the file; ``cluster_file`` does it for a design
point (snax_forge/design/point.py). scenarios/clusters/clusters.py builds
alu4, red4 and mul1 through these, so the checked-in clusters and the
derived ones cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from snax_forge.brm import Instance
from snax_forge.design.platform import Platform
from snax_forge.design.streamers import Streamer, instances, resolve, streamer_name
from snax_forge.snax_model.scenario import ClusterConfig, ComponentSpec, RegisterMapSpec

if TYPE_CHECKING:
    from snax_forge.design.point import DesignPoint

XBAR, DMA, CTL = "xbar", "dma", "ctl"


@dataclass(frozen=True)
class Accel:
    """One accelerator of the cluster: its entry and its streamers, in port order."""

    name: str
    accel: str
    params: dict[str, Any]
    streamers: tuple[Streamer, ...]

    def attach(self) -> dict[str, str]:
        return {s.port: s.name for s in self.streamers}


def accel_of(name: str, inst: Instance, streamers: Mapping[str, Streamer]) -> Accel:
    """The accelerator ``name`` built from a BRM instance and its resolved streamers."""
    kind, params = inst.accel_entry()
    ports = [p.name for p in inst.brm.interface.ports]
    return Accel(name, kind, params, tuple(streamers[streamer_name(name, p)] for p in ports))


def stub(
    platform: Platform,
    name: str,
    accel: str,
    params: Mapping[str, Any],
    ports: Sequence[tuple[str, bool, int]],
) -> Accel:
    """An accelerator given by a registered kind and params; ``ports`` are
    ``(port, write, lanes)`` in order, their streamers shaped by the platform's shell."""
    streamers = tuple(
        Streamer(s, name, p, write, (lanes,), platform.options(s))
        for p, write, lanes in ports
        for s in [streamer_name(name, p)]
    )
    return Accel(name, accel, dict(params), streamers)


def cluster_config(platform: Platform, accels: Sequence[Accel]) -> ClusterConfig:
    """The cluster file of ``platform`` with ``accels``; see the module doc."""
    comps = [ComponentSpec(XBAR, "xbar", dict(platform.xbar))]
    if platform.l2 is not None:
        comps.append(ComponentSpec(DMA, "dma", platform.dma.to_dict()))
    spatial: dict[str, list[int]] = {}
    for a in accels:
        for s in a.streamers:
            comps.append(ComponentSpec(s.name, "streamer", s.config().to_dict()))
            if list(s.spatial_bounds) != [s.n_ports]:
                spatial[s.name] = list(s.spatial_bounds)
        comps.append(
            ComponentSpec(a.name, "accel", accel=a.accel, params=dict(a.params), attach=a.attach())
        )
    comps.append(ComponentSpec(CTL, "controller", platform.controller.to_dict()))
    blocks = [c.name for c in comps if c.kind not in ("xbar", "controller")]
    return ClusterConfig(
        l1=platform.l1,
        l2=platform.l2,
        components=comps,
        register_map=RegisterMapSpec(
            window=platform.register_window, blocks=blocks, spatial_bounds=spatial
        ),
    )


def cluster_of(platform: Platform, instances: Mapping[str, Instance]) -> ClusterConfig:
    """The cluster of BRM instances on ``platform``, the shell resolved here."""
    shell = resolve(platform, dict(instances))
    return cluster_config(platform, [accel_of(k, inst, shell) for k, inst in instances.items()])


def cluster_file(point: DesignPoint) -> ClusterConfig:
    """The cluster file of a design point, from its platform, instances and streamers."""
    accels = [accel_of(k, inst, point.streamers) for k, inst in instances(point.graph).items()]
    return cluster_config(point.platform, accels)
