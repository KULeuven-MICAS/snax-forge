# bicg_v4_1 at trace level beat

A recorded SNAX-GVSoC run, so CI converts real events without GVSoC: bicg_v4_1 at M = 40, N = 32, W = 4, R = 4: one group `bicg_types` of units `mul`, `sum` (q = A p, its output packed two to a word, its seed carried across the two strips) and `mac` (s = Aᵀ r in a bank of 4 rows, r held a row), one reader of A forked to `mul` and `mac`, one task; not at level stall, whose conversion refuses it (STATUS, open item 162). Made in snax-forge's image, the submodule checked out, by

    pixi run flow recipes/bicg/bicg_v4_1.json --platform platforms/snax_forge.json --set M=40 --set N=32 --set R=4 --trace beat

and copied from each flow's folder: `run/events.log` as `events_beat.log`, `check.json` (the profile's `total_cycles` and `functional_check`) and `elf_symbols.json` (`run.json`'s), the ELF the same for every level (sha256 `6e153c62ec6aedc4d0dab4f2210213fed4c802310fb7197f784aab7567a48cbe`). The run: 687 cycles, data 0 errors (72 words), counters `q_mult_bicg_types` 327 (320 beats).

The generated files the run was made on are not kept: `tests/run/helpers.py`'s `recorded_files` makes them again from the recipe at these sizes (`FORGE_FIXTURES`, the forge/ folder of the same name) and checks each against its sha256 here; when one differs, record the run again by the commands above and write the new hashes here.

- `accelerator.json`: sha256 `e6812b4416b40073c5ce520090bdc562c150ff362b962a0ac69bf255b75e801e`
- `design_point.json`: sha256 `624ca88411d602054afeb1b421c0cec544653567319e871d001eb9f1e2828e3a`
- `manifest.json`: sha256 `31a951cf76c0e81f1bdb4d0cbf829ab215a54dcf79cbd05bfb2df68744d0492a`
- `snax_forge.hjson`: sha256 `dab52f8d090f0471cc53a7236fd8850a8edb7bf4a1746bbbfc80b536ebbb0733`
- `snax_forge_bicg_v4_1_streamer_bicg_types.hjson`: sha256 `9116b16f890666cf9757a0b8d5aa7316c74a81e5851f63139ac0d5c0f6c96b74`
- `tasks.json`: sha256 `47692ff34efd8ec46ffa63d2f64efaf7bf5449e17860677acaf8ac3ec7d5be53`
