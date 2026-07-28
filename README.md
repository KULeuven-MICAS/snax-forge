# :hammer: :fire: SNAX-FORGE :fire: :hammer:

This repository is a work-in-progress (WIP) where we develop a program that uses the DaCe IR to generate and multi-accelerator architecture for the SNAX compute cluster.
This tool would help HW-oriented engineers bring the SW side closer to them rather than the typical otherway around were tools like DaCe make it easier for SW designers and optimization engineers to match the HW-SW combinations.
This work's motivation is to enable a HW-SW co-design but with the perspective on the HW-side.

# Anticipated Features
1. First is to break a program into an IR and use that IR to make accelerator(s) that fit within the SNAX/PULP compute clusters. In here we use the DaCe tool and use it as a model to make our HW accelerators.
2. We will offer some block primitives that enable an efficient yet modular designs that help constrain the design space a bit more unlike classic HLS that maps every operation on every kernel. These accelerators will be generated with Chisel.
3. We offer also a kernel library generation, where for the given designed accelerator we automatically generate the designated library kernels that are light function calls.
4. Finally, we have compute cluster model that simulates the flow of the accelerator for fast investigations rather than relying entirely on RTL simulations. Those can happen afterwards.
