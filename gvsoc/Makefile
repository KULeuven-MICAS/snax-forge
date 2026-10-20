# SNAX-GVSoC from snax-forge (D128): builds snax-gvsoc, its programs, and
# runs gvrun on a target of the install.
#
# Run it in snax-forge's image, from the plain shell or through pixi (`pixi
# run make -C gvsoc ...`, as the flow does, D136): it names the image's tools
# below, so both build and run the same. From this folder:
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

# The image's tools, whatever environment make runs in: GVSoC's Python (with
# its packages), CMake and the host compilers. pixi's environment sets CC,
# CXX, their flags and CMAKE_PREFIX_PATH to conda's, puts its own python3,
# make, ld and as first on PATH, and CMake clears its cache when CC changes.
# CC and CXX are this file's unless given on the command line (make's own
# default and the environment's do not count). The image's Python and the
# host compilers' folder go first on PATH for the recipes; gvrun runs on
# GVRUN_PYTHON (its last line).
PYTHON ?= /opt/venv/bin/python3
CMAKE ?= /usr/bin/cmake
ifneq ($(origin CC),command line)
CC = /usr/bin/gcc
endif
ifneq ($(origin CXX),command line)
CXX = /usr/bin/g++
endif
GVRUN_PYTHON = $(PYTHON)
export PYTHON CMAKE CC CXX GVRUN_PYTHON
export PATH := $(dir $(PYTHON)):$(dir $(CC)):$(PATH)

# What pixi's environment adds for conda's compilers (the diff of `env`
# between `pixi run` and the plain shell), kept from the sub-makes, CMake
# and the compilers
unexport CPP CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CMAKE_ARGS CMAKE_PREFIX_PATH \
	AR AS LD NM RANLIB STRIP OBJCOPY OBJDUMP READELF ADDR2LINE CXXFILT ELFEDIT \
	GPROF SIZE STRINGS GCC GXX GCC_AR GCC_NM GCC_RANLIB CC_FOR_BUILD CXX_FOR_BUILD \
	CPP_FOR_BUILD DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS LDFLAGS_LD MESON_ARGS \
	CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION CONDA_TOOLCHAIN_BUILD \
	CONDA_TOOLCHAIN_HOST HOST BUILD build_alias host_alias

# GVSoC's build/ and install/ (snax-gvsoc's Makefile puts both under it).
# CMake refuses a build folder made from another checkout, so `build` keeps
# the checkout it built in checkout.txt (its path, then the commit, written
# once the build passed) and clears both when the path changes. The flow
# builds again when the path or the commit differs or the tree is dirty.
GVSOC_WORKDIR ?= $(abspath $(CURDIR)/../out/snax-gvsoc)
INSTALL = $(GVSOC_WORKDIR)/install
BUILT_FROM = $(GVSOC_WORKDIR)/checkout.txt

# The targets GVSoC compiles (components and platform trees): snax-gvsoc's
# SNAX_TARGETS, read from its Makefile
SNAX_TARGETS := $(shell sed -n 's/^SNAX_TARGETS = //p' $(SNAX_GVSOC_ABS)/Makefile)
TARGETS ?= $(SNAX_TARGETS)

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
	@if [ -d $(GVSOC_WORKDIR)/build ] && [ "$$(head -n 1 $(BUILT_FROM) 2>/dev/null)" != "$(SNAX_GVSOC_ABS)" ]; then \
		echo "$(GVSOC_WORKDIR) was built from another checkout: clearing it"; \
		rm -rf $(GVSOC_WORKDIR)/build $(GVSOC_WORKDIR)/install; \
	fi
	mkdir -p $(GVSOC_WORKDIR)
	echo "$(SNAX_GVSOC_ABS)" > $(BUILT_FROM)
	$(MAKE) -C $(SNAX_GVSOC_ABS) build GVSOC_WORKDIR=$(GVSOC_WORKDIR) TARGETS="$(TARGETS)"
	git -c safe.directory='*' -C $(SNAX_GVSOC_ABS) rev-parse HEAD >> $(BUILT_FROM)

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
