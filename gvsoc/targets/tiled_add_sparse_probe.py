#
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#

"""tiled_add_sparse_probe: snax-gvsoc's snax_tiled_add accelerator
(snax/snax_tiled_add_acc.py) on snax_alu_sparse's sparse interconnect
(snax/snax_alu_sparse.py), read from snax-forge's gvsoc/targets (GEN3 round
1b): a component tree no built target has, made of components the install
holds (snax_tiled_add's and snax_alu_sparse's), so it runs on the JSON
config path. Needs snax-gvsoc from GEN3 step 2 on (SnaxAluBoard.accelerator).

    gvrun --target tiled_add_sparse_probe --work-dir build/work/<name> \
        --param chip/soc/binary=<elf> run
"""

import gvsoc.runner
from snax_alu_sparse import SnaxAluSparseBoard
from snax_tiled_add_acc import SnaxTiledAddAcc


class TiledAddSparseProbeBoard(SnaxAluSparseBoard):

    accelerator = SnaxTiledAddAcc


class Target(gvsoc.runner.Target):

    gapy_description = "snax_alu_sparse's cluster with snax_tiled_add's accelerator"
    model = TiledAddSparseProbeBoard
    name = "tiled_add_sparse_probe"

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model=TiledAddSparseProbeBoard, name=name)
