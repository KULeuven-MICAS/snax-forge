# SNAX-GVSoC from snax-forge (GEN3 step 0a, D128): builds snax-gvsoc, its
# programs, and runs gvrun on a target of the install.
#
# Run it in snax-forge's image, in the plain container shell, not in pixi
# (pixi's environment sets CC and CXX to conda's gcc), from this folder:
#
#     make build                     # GVSoC for TARGETS into out/snax-gvsoc/
#     make sw APPS=snax-alu          # sw/ and the apps, into $(SNAX_GVSOC)/build/sw/
#     make run TARGET=snax_alu ELF=$(SNAX_GVSOC)/build/sw/apps/snax-alu/100/snax-alu.elf
#
# A design point gen-gvsoc wrote to out/gvsoc/<design>/: its program built
# by the generic app, then run on snax_hjson with its cluster file:
#
#     make sw APPS=forge FORGE=../out/gvsoc/vecadd DESIGN=vecadd
#     make run TARGET=snax_hjson DESIGN=vecadd \
#         GVRUN_ARGS="--param hjson=$PWD/../out/gvsoc/vecadd/snax_alu.hjson" \
#         ELF=../out/snax-gvsoc/sw/vecadd/forge.elf
#
# SNAX_GVSOC is the snax-gvsoc checkout: the submodule by default, or a
# sibling checkout (SNAX_GVSOC=../../snax-gvsoc from here).

SNAX_GVSOC ?= $(CURDIR)/snax-gvsoc
SNAX_GVSOC_ABS = $(abspath $(SNAX_GVSOC))

# GVSoC's build/ and install/ (snax-gvsoc's Makefile puts both under it).
# CMake refuses a build folder made from another checkout, so `build` keeps
# the checkout it built in checkout.txt and clears both when it changes.
GVSOC_WORKDIR ?= $(abspath $(CURDIR)/../out/snax-gvsoc)
INSTALL = $(GVSOC_WORKDIR)/install
BUILT_FROM = $(GVSOC_WORKDIR)/checkout.txt

# The targets GVSoC compiles (components and platform trees)
TARGETS ?= snax_alu;snax_alu_sparse;snax_tiled_add;snax_hjson

# Programs of $(SNAX_GVSOC)/sw/apps/ that `sw` builds after sw/ itself. FORGE,
# a folder of gen-gvsoc's form, goes to the forge app as an absolute path, with
# DESIGN, the folder's design; its ELF goes to FORGE_BUILD, never beside the
# submodule's fixtures' ELFs.
APPS ?=
FORGE ?=
DESIGN ?=
FORGE_BUILD = $(GVSOC_WORKDIR)/sw/$(DESIGN)

# `run`: the installed target, the ELF, the work directory (one per design
# when DESIGN is set, else per ELF name), and gvrun options put before the
# command (e.g. --trace=snax_acc0)
TARGET ?=
ELF ?=
WORK_DIR ?= $(GVSOC_WORKDIR)/work/$(TARGET)/$(or $(DESIGN),$(basename $(notdir $(ELF))))
GVRUN_ARGS ?=

.PHONY: build sw run

build:
	@if [ -d $(GVSOC_WORKDIR)/build ] && [ "$$(cat $(BUILT_FROM) 2>/dev/null)" != "$(SNAX_GVSOC_ABS)" ]; then \
		echo "$(GVSOC_WORKDIR) was built from another checkout: clearing it"; \
		rm -rf $(GVSOC_WORKDIR)/build $(GVSOC_WORKDIR)/install; \
	fi
	mkdir -p $(GVSOC_WORKDIR)
	echo "$(SNAX_GVSOC_ABS)" > $(BUILT_FROM)
	$(MAKE) -C $(SNAX_GVSOC_ABS) build GVSOC_WORKDIR=$(GVSOC_WORKDIR) TARGETS="$(TARGETS)"

sw:
	$(MAKE) -C $(SNAX_GVSOC_ABS) sw
	$(foreach app,$(APPS),$(MAKE) -C $(SNAX_GVSOC_ABS)/sw/apps/$(app) $(if $(FORGE),FORGE=$(abspath $(FORGE)) FORGE_BUILD=$(FORGE_BUILD)) &&) true

# The install's gvrun script sets LD_LIBRARY_PATH, PATH, PYTHONPATH,
# USE_GVRUN and USE_GVRUN2 for the install, and passes its targets,
# generators and models directories before these arguments.
run:
	$(if $(TARGET),,$(error set TARGET))
	$(if $(ELF),,$(error set ELF))
	mkdir -p $(WORK_DIR)
	$(INSTALL)/bin/gvrun --target $(TARGET) \
		--work-dir $(WORK_DIR) --param chip/soc/binary=$(abspath $(ELF)) $(GVRUN_ARGS) run
