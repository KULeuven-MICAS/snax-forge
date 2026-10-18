#
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#

"""alu_probe: snax-gvsoc's snax/snax_alu.py (main, c911569) under another name,
read from snax-forge's gvsoc/targets (GEN3 step 0b): a target outside the
install that starts on the components the install already holds. Kept as a
smoke test. Its accelerator is AluProbeAcc, a copy of snax_alu_acc.py's
SnaxAluAcc on its own copy of the streamer template; below, snax_alu.py's
own text.

The snax_alu target: snax_cluster's cfg/snax_alu_cluster_old.hjson at b2672f8,
unchanged up to the pin, 9fbf35d.

One cluster with a compute core (hart 0, the one that drives the accelerator)
and a DMA core (hart 1), both rv32ima, 128 KiB TCDM in 32 banks and the plain
interconnect. The compute core's accelerator CSRs go to snax_alu_acc.py,
which splits them between the streamer and the ALU. The SNAX ports, the cores',
AXI's and the DMA's TCDM ports go through one crossbar into the banks, the DMA
taking a superbank per beat. The values are written here by hand; everything else is the Snitch
chip of pulp/pulp/chips/snitch/snitch.py, whose own arch tree fixes the core
count at 9.

    gvrun --target alu_probe --work-dir build/work/<name> \
        --param chip/soc/binary=<elf> run
"""

import os

import gvsoc.runner
import gvsoc.systree
import memory.memory
from vp.clock_domain import Clock_domain
from gvrun.attribute import Tree, Area
from pulp.snitch.snitch_cluster.snitch_cluster import ClusterArch
from pulp.chips.snitch.snitch import Snitch
from snax_alu_acc import SnaxAluAcc, SNAX_CSR_FIRST, SNAX_CSR_LAST, TCDM_SIZE, ALU_NUM_PE, \
    ALU_DATA_WIDTH
import interco.router as router
from snax_streamer import SnaxStreamer, nb_csrs, nb_tcdm_ports, tcdm_port_names
from snax_alu_shell import SnaxAluShell


# The probe's own copy of snax_alu_acc.py's STREAMER_TEMPLATE and SnaxAluAcc,
# so the probe can change a property (fifo_depth) without editing snax-gvsoc
PROBE_STREAMER_TEMPLATE = {
    'data_reader_params': {
        'spatial_bounds': [[4], [4]],
        'temporal_dim': [1, 1],
        'num_channel': [4, 4],
        'fifo_depth': [8, 8],
    },
    'data_writer_params': {
        'spatial_bounds': [[4]],
        'temporal_dim': [1],
        'num_channel': [4],
        'fifo_depth': [8],
    },
    'snax_library_name': 'snax-alu',
}
PROBE_ALU_CSR_FIRST = SNAX_CSR_FIRST + nb_csrs(PROBE_STREAMER_TEMPLATE)


class AluProbeAcc(SnaxAluAcc):
    """SnaxAluAcc on PROBE_STREAMER_TEMPLATE: the CSR mux, the streamer, the ALU."""

    nb_tcdm_ports = nb_tcdm_ports(PROBE_STREAMER_TEMPLATE)
    tcdm_port_names = ['streamer.' + name for name in tcdm_port_names(PROBE_STREAMER_TEMPLATE)]

    def __init__(self, parent: gvsoc.systree.Component, name: str):
        gvsoc.systree.Component.__init__(self, parent, name)

        csr_mux = router.Router(self, 'csr_mux')
        streamer = SnaxStreamer(self, 'streamer', template=PROBE_STREAMER_TEMPLATE,
            csr_base=SNAX_CSR_FIRST, tcdm_size=TCDM_SIZE)
        alu = SnaxAluShell(self, 'alu', csr_base=PROBE_ALU_CSR_FIRST, num_pe=ALU_NUM_PE,
            data_width=ALU_DATA_WIDTH)

        streamer_base = 4 * SNAX_CSR_FIRST
        alu_base = 4 * PROBE_ALU_CSR_FIRST
        csr_mux.o_MAP(streamer.i_INPUT(), base=streamer_base,
            size=alu_base - streamer_base)
        csr_mux.o_MAP(alu.i_INPUT(), base=alu_base,
            size=4 * (SNAX_CSR_LAST + 1) - alu_base)

        alu.o_IN_A(streamer.i_READER(0))
        alu.o_IN_B(streamer.i_READER(1))
        alu.o_OUT(streamer.i_WRITER(0))

        self.bind(self, 'input', csr_mux, 'input')

        for port in range(0, self.nb_tcdm_ports):
            streamer.o_TCDM(port, self.i_TCDM_INTERNAL(port))
            streamer.o_TCDM_PRIORITY(port, self.i_TCDM_PRIORITY_INTERNAL(port))

# GVSoC reads a target twice: `make build` lists its components through gapy
# without USE_GVRUN, and gvrun runs it with USE_GVRUN set. Each pulp module
# defines different arch classes for the two, so this target does too: the
# build reuses SnitchArch with the core count changed, the run uses
# SnaxAluAttr below. Only the run needs misa, a value read at run time; both
# need the accelerators and the interconnect, since the build compiles their
# sources and checks the crossbar's configuration.
if os.environ.get('USE_GVRUN') is None:
    from pulp.chips.snitch.snitch import SnitchArch, SnitchArchProperties


# Cores per cluster in the hjson's hive: the compute core, then the DMA core
# (GVSoC's Snitch cluster takes its last core as the DMA core).
NB_CORE = 2

# misa of an rv32ima core: MXL = 1 (32 bit), extensions A (bit 0), I (bit 8)
# and M (bit 12). snax_cluster's start-up code reads it to skip the FP
# registers. Only the CSR value changes: the model still executes FP
# instructions.
MISA_RV32IMA = (1 << 30) | (1 << 12) | (1 << 8) | (1 << 0)

# SNAX accelerators per core index: the compute core (hart 0) drives the ALU.
# Its CSR layout is in snax_alu_acc.py.
SNAX_ACC = {0: AluProbeAcc}

# The DMA and DRAM measured on the RTL in SG2 (docs/notes/NOTES.md, SG2 step 4):
# one 64 B beat per cycle into L1 (dma_data_width 512), 3 transfers queued
# (dma_req_fifo_depth) and 3 bursts in flight (dma_axi_req_fifo_depth). A DMA
# read's first beat comes 5 cycles after its request from L1 and 11 from DRAM:
# the 5 common to both are the DMA's own read path (read_latency), the other 6
# are the DRAM's. From the SG7 round 4 follow-up (docs/plans/SG7.md): a beat
# reaches L1 2 cycles after the DMA sends it, the wide crossbar's cut and
# axi_to_mem_interleaved, and the DMA is done at the send
# (tcdm_write_latency); the DMA's midend, backend and AXI read each take a
# cycle, whatever else the DMA is doing (one_cycle_per_stage).
SNAX_DMA = {'tcdm_width': 64, 'transfer_queue_size': 3, 'burst_queue_size': 3,
    'read_latency': 5, 'tcdm_write_latency': 2, 'one_cycle_per_stage': True}
DRAM_LATENCY = 6
# The hjson's dma_data_width in bits: a DMA beat takes a superbank of
# dma_data_width / data_width banks (data_width 64, the TCDM's 8 B bank), 8,
# in the SNAX crossbar (SG7 round 4)
DMA_DATA_WIDTH = 512

# The cores' loads, measured on the RTL in SG5 (docs/notes/NOTES.md, SG5 step 6)
# in snax-alu's check loop. One integer load in flight, as the hjson's
# num_int_outstanding_loads: 1: a load issues only once the previous load's
# data is back. From one load to the next, 4 cycles from L1 and 14 from DRAM;
# the core's data path gets the cycles GVSoC lacks for that (core_latency), on
# its TCDM port and on its way out of the cluster (narrow_axi). Instruction
# fetches keep their own path. On the TCDM port the SNAX crossbar answers a
# core's read one cycle after its grant plus these cycles; the core goes on
# meanwhile, and the next load or an instruction on the loaded register runs in
# the cycle after: 4 from one L1 load to the next with tcdm 2, as before the
# cores went through the crossbar (SG7 rounds 3 and 3b).
# GVSoC's host reads tohost every 1000 cycles; the RTL's host does not use the
# bus, so the poll reads DRAM through the zero-time debug backdoor (SG7 round 3b).
SNAX_CORE = {'int_loads_in_flight': 1, 'htif_poll_backdoor': True}
CORE_LATENCY = {'tcdm': 2, 'narrow': 6}
DRAM_WIDTH_LOG2 = 6  # 64 B per cycle

# The SNAX ports' interconnect, ClusterArch arguments: the hjson's plain one
# (tcdm.sparse_interconnect false). snax_alu_sparse.py replaces it.
INTERCONNECT = {'snax_interconnect': 'plain'}


class SnaxAluAttr(Tree):
    """The address map and cluster of the target, as SnitchAttr gives them."""

    def __init__(self, parent, name, interconnect):
        super().__init__(parent, name)
        self.chip = SnaxAluAttr.Chip(self, 'chip', interconnect)
        self.hbm = SnaxAluAttr.Hbm(self, 'hbm')

    class Hbm(Tree):

        def __init__(self, parent, name):
            super().__init__(parent, name)
            self.type = 'simple'
            self.size = 0x8000_0000

    class Chip(Tree):

        def __init__(self, parent, name, interconnect):
            super().__init__(parent, name)
            self.soc = SnaxAluAttr.Chip.Soc(self, 'soc', interconnect)

        class Soc(Tree):

            def __init__(self, parent, name, interconnect):
                super().__init__(parent, name)
                self.bootrom = Area(self, 'rom', 0x0000_1000, 0x0001_0000,
                    description='Bootrom range')
                self.hbm = Area(self, 'hbm', 0x8000_0000, 0x8000_0000,
                    description='DRAM range')
                self.cluster = Area(self, 'cluster', 0x1000_0000, 0x0004_0000,
                    description='Cluster range')
                self.hbm_latency = DRAM_LATENCY
                self.floonoc = False
                self.use_spatz = False
                self.nb_cluster = 1
                self.clusters = [
                    ClusterArch(self, 'cluster', self.get_cluster_base(0), 0,
                        nb_core_per_cluster=NB_CORE, misa=MISA_RV32IMA,
                        snax_acc=SNAX_ACC, dma=SNAX_DMA, core=SNAX_CORE,
                        core_latency=CORE_LATENCY, snax_dma_data_width=DMA_DATA_WIDTH,
                        **interconnect)
                ]

            def get_cluster_base(self, id: int):
                return self.cluster.base + id * self.cluster.size

            def get_cluster(self, id: int):
                return self.clusters[id]


class SnaxAluBoard(gvsoc.systree.Component):

    interconnect = INTERCONNECT

    def __init__(self, parent, name: str, parser, options):
        super().__init__(parent, name, options=options)

        clock = Clock_domain(self, 'clock', frequency=10000000)

        if os.environ.get('USE_GVRUN') is None:
            properties = SnitchArchProperties()
            properties.nb_core_per_cluster = NB_CORE
            properties.snax_acc = SNAX_ACC
            properties.dma = SNAX_DMA
            properties.snax_dma_data_width = DMA_DATA_WIDTH
            for key, value in self.interconnect.items():
                setattr(properties, key, value)
            arch = SnitchArch(self, properties=properties)
        else:
            arch = SnaxAluAttr(self, 'snax_alu', self.interconnect)
        # Exposed so tools like the build process can query the memory map.
        self.set_attributes(arch)

        chip = Snitch(self, 'chip', parser, arch.chip, None, [])
        mem = memory.memory.Memory(self, 'mem', size=arch.hbm.size, atomics=True,
            width_log2=DRAM_WIDTH_LOG2)

        self.bind(clock, 'out', chip, 'clock')
        self.bind(clock, 'out', mem, 'clock')
        self.bind(chip, 'hbm', mem, 'input')


class Target(gvsoc.runner.Target):

    gapy_description = "snax_alu under another name, from snax-forge's gvsoc/targets"
    model = SnaxAluBoard
    name = "alu_probe"

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model=SnaxAluBoard, name=name)
