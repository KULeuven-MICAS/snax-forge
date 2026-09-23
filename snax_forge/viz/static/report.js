// The profile report of one run (VIS1): the model's Profile (CONTRACTS.md
// section 7), the run's run.json and the FIFO busy windows (D56), drawn top
// to bottom. Works at every trace level; only the busy-window FIFO numbers
// need a task or beat trace, and they say so when missing.
//
// Nothing is computed here that the profile already has; the report only
// arranges numbers. Components appear in registration order (the cluster
// file's `components`), which is also the tick and trace-source order (D41).

import { dec, h, int, pct, section, table } from "./dom.js";

// Cycle classes per component kind, in CONTRACTS.md section 7 order.
const CLASS_ORDER = {
  accel: ["busy", "stall_out", "stall_in", "idle"],
  streamer: ["busy", "stall_xbar", "stall_fifo", "idle"],
  dma: ["busy", "stall_l1", "stall_mem", "idle"],
  controller: ["command", "wait", "idle"],
};

// Colour group of each class: memory stalls warm, flow stalls cool (VIS1 plan).
const GROUP = {
  busy: "busy", idle: "idle",
  stall_xbar: "mem", stall_l1: "mem",
  stall_fifo: "flow", stall_in: "flow", stall_out: "flow", stall_mem: "flow",
  command: "command", wait: "wait",
};

const LEGEND = [
  ["busy", "Busy"], ["mem", "Memory stall (stall_xbar, stall_l1)"],
  ["flow", "Flow stall (stall_fifo, stall_in, stall_out, stall_mem)"],
  ["idle", "Idle"], ["command", "Controller command (control overhead)"], ["wait", "Controller wait"],
];

/** Cycles per class of a component, from the profile, or null if it has none (the xbar). */
function cyclesOf(profile, spec) {
  if (spec.kind === "accel") return profile.accelerators[spec.name]?.cycles;
  if (spec.kind === "streamer") return profile.streamers[spec.name]?.cycles;
  if (spec.kind === "dma") return profile.dmas[spec.name]?.cycles;
  if (spec.kind === "controller" && profile.controller?.name === spec.name) return profile.controller.cycles;
  return null;
}

function ordered(kind, cycles) {
  const known = CLASS_ORDER[kind] || [];
  return [...known.filter((c) => c in cycles), ...Object.keys(cycles).filter((c) => !known.includes(c))];
}

/** A stacked bar of cycles per class; segment widths are shares of `total`. */
function classBar(kind, cycles, total) {
  return h("div", { class: "stack", role: "img", "aria-label": ordered(kind, cycles).map((c) => `${c} ${cycles[c]}`).join(", ") },
    ordered(kind, cycles).filter((c) => cycles[c] > 0).map((c) =>
      h("span", { class: `seg g-${GROUP[c] || "other"}`, style: { flexGrow: cycles[c] }, title: `${c}: ${int(cycles[c])} cycles (${pct(cycles[c], total)})` })));
}

function classCounts(kind, cycles, total) {
  return h("ul", { class: "counts" },
    ordered(kind, cycles).map((c) =>
      h("li", { class: cycles[c] ? null : "zero" },
        h("i", { class: `key g-${GROUP[c] || "other"}` }), `${c} `, h("b", {}, int(cycles[c])), ` ${pct(cycles[c], total)}`)));
}

// -- 2. Where the cycles go ----------------------------------------------------

function cyclesSection(detail) {
  const { profile, run } = detail;
  const total = profile.total_cycles;
  const rows = run.cluster.components
    .map((spec) => ({ spec, cycles: cyclesOf(profile, spec) }))
    .filter((r) => r.cycles);
  return section("cycles", "Where the cycles go",
    `Every component's ${int(total)} cycles by class, in registration order. Totals only; when they happen is the schedule view.`,
    h("ul", { class: "legend" }, LEGEND.map(([g, label]) => h("li", {}, h("i", { class: `key g-${g}` }), label))),
    h("div", { class: "rows" }, rows.map(({ spec, cycles }) =>
      h("div", { class: "row" },
        h("div", { class: "who" }, h("b", {}, spec.name), h("span", { class: "kind" }, spec.kind === "accel" ? spec.accel : spec.kind)),
        classBar(spec.kind, cycles, total),
        classCounts(spec.kind, cycles, total)))));
}

// -- 3. Accelerators -----------------------------------------------------------

function accelSection(detail) {
  const { profile, run } = detail;
  const specs = Object.fromEntries(run.cluster.components.map((c) => [c.name, c]));
  const names = Object.keys(profile.accelerators);
  if (!names.length) return null;
  return section("accelerators", "Accelerators",
    "Utilisation is busy cycles over the whole run; a busy cycle is a firing. Beats are counted per port.",
    names.map((n) => {
      const a = profile.accelerators[n];
      const spec = specs[n] || {};
      const params = Object.entries(spec.params || {}).map(([k, v]) => `${k} ${v}`).join(", ");
      return h("div", { class: "block" },
        h("h3", {}, n, h("span", { class: "kind" }, `${spec.accel || ""} ${params ? `(${params})` : ""}`)),
        h("div", { class: "util" },
          h("div", { class: "meter" }, h("span", { class: "g-busy", style: { width: `${100 * a.utilisation}%` } })),
          h("b", {}, `${(100 * a.utilisation).toFixed(1)}%`), ` utilisation, ${int(a.cycles.busy)} firings`),
        table(
          [{ key: "port", label: "Port" }, { key: "stream", label: "Streamer" }, { key: "beats", label: "Beats", num: true, fmt: int }],
          Object.entries(a.beats).map(([port, beats]) => ({ port, beats, stream: spec.attach?.[port] ?? "" }))));
    }));
}

// -- 4. Streamers and FIFOs ----------------------------------------------------

/** Tiny histogram: one bar per count 0..depth, heights relative to the largest bucket. */
function histogram(hist, label) {
  const top = Math.max(1, ...hist);
  return h("div", { class: "hist", role: "img", "aria-label": `${label}: ${hist.join(", ")}` },
    hist.map((n, c) => h("span", { title: `${c} held: ${int(n)} cycles` },
      h("i", { style: { height: `${(100 * n) / top}%` } }), h("small", {}, c))));
}

function streamerSection(detail, fifo) {
  const { profile } = detail;
  const names = Object.keys(profile.streamers);
  if (!names.length) return null;
  const note = fifo.available
    ? "Occupancy per lane over the whole run, and over the busy window: the cycles in which the streamer or its accelerator had a task (D56). The histogram shows the window."
    : `Occupancy per lane over the whole run. Busy-window numbers need a task or beat trace (${fifo.reason}).`;
  return section("streamers", "Streamers and FIFOs", note,
    names.map((n) => {
      const s = profile.streamers[n];
      const f = s.fifo;
      const win = fifo.streamers?.[n];
      const winLanes = win && !win.reason ? win.lanes : null;
      const rows = f.hist.map((hist, lane) => ({
        lane, max: f.max[lane], mean: f.mean[lane],
        wmean: winLanes ? winLanes[lane].mean : null,
        hist: winLanes ? winLanes[lane].hist : hist,
      }));
      const winText = !win ? null : win.reason ? `Busy window withheld: ${win.reason}.`
        : `Busy window: ${int(win.window_cycles)} of ${int(profile.total_cycles)} cycles, ${win.window.map(([a, b]) => `[${a}, ${b})`).join(" ")}, owners ${win.owners.join(" and ")}.`;
      return h("div", { class: "block" },
        h("h3", {}, n, h("span", { class: "kind" }, `${s.write ? "writer" : "reader"}, ${s.ports.length} ports, FIFO ${f.name} depth ${f.depth}`)),
        winText ? h("p", { class: "note" }, winText) : null,
        table([
          { key: "lane", label: "Lane", num: true },
          { key: "max", label: "Max", num: true },
          { key: "mean", label: "Mean (run)", num: true, fmt: (v) => dec(v) },
          { key: "wmean", label: "Mean (busy window)", num: true, fmt: (v) => dec(v) },
          { key: "hist", label: winLanes ? "Cycles per count (window)" : "Cycles per count (run)", fmt: (v) => histogram(v, "cycles per count") },
        ], rows));
    }));
}

// -- 5. Memory -----------------------------------------------------------------

function memorySection(detail) {
  const { profile, run } = detail;
  const b = profile.banks;
  if (!b) return null;
  const l1 = run.cluster.l1;
  const group = Math.max(1, (l1.wide_bits || 512) / (l1.width_bits || 64)); // banks per superbank (D33)
  const top = Math.max(1, ...b.grants);
  const hot = b.conflicts.map((c, i) => c > 0 || b.stalls[i] > 0 || b.blocked[i] > 0);
  const nHot = hot.filter(Boolean).length;

  // The strip: one cell per bank, height = grants, superbanks separated.
  const strip = h("div", { class: "strip scroll" },
    b.grants.map((g, i) => h("div", {
      class: `bank${hot[i] ? " hot" : ""}${i % group === 0 && i ? " sb" : ""}`,
      title: `bank ${i}: ${g} grants, ${b.reads[i]} reads, ${b.writes[i]} writes, ${b.conflicts[i]} conflicts, ${b.stalls[i]} stalls, ${b.blocked[i]} blocked by a wider grant`,
    }, h("div", { class: "fill" }, h("i", { style: { height: `${(100 * g) / top}%` } })), h("small", {}, i))));

  const metrics = ["reads", "writes", "grants", "conflicts", "stalls", "blocked"];
  const bankTable = h("div", { class: "scroll" }, h("table", { class: "banks" },
    h("thead", {}, h("tr", {}, h("th", {}, "Bank"), b.grants.map((_, i) => h("th", { class: `num${hot[i] ? " hot" : ""}` }, i)))),
    h("tbody", {}, metrics.map((m) => h("tr", {}, h("th", {}, m),
      b[m].map((v, i) => h("td", { class: `num${hot[i] ? " hot" : ""}${v ? "" : " zero"}` }, int(v))))))));

  const ports = Object.entries(profile.ports).map(([name, p]) => ({ name, ...p }));
  const portTable = table([
    { key: "name", label: "Port" }, { key: "owner", label: "Owner" },
    { key: "width", label: "Width (bits)", num: true },
    { key: "grants", label: "Grants", num: true, fmt: int },
    { key: "stalls", label: "Stalls", num: true, fmt: int },
    { key: "stalls_wider", label: "Stalls by a wider grant", num: true, fmt: int },
  ], ports, (p) => (p.stalls ? "hot" : null));

  const stalled = ports.filter((p) => p.stalls).map((p) => p.name);
  return section("memory", "Memory",
    `${l1.n_banks} banks of ${l1.width_bits} bits, ${group} per superbank. ` +
      (nHot ? `${nHot} banks saw conflicts or stalls (marked).` : "No bank saw a conflict or a stall."),
    h("h3", {}, "Grants per L1 bank"), strip, bankTable,
    h("h3", {}, "Interconnect ports", h("span", { class: "kind" },
      stalled.length ? `stalls on ${stalled.join(", ")}` : "no stalls")),
    portTable);
}

// -- 6. DMA and L2 -------------------------------------------------------------

function dmaSection(detail) {
  const { profile } = detail;
  const dmas = Object.entries(profile.dmas).map(([name, d]) => ({ name, ...d }));
  const l2 = Object.entries(profile.l2).map(([name, m]) => ({ name, ...m }));
  if (!dmas.length && !l2.length) return null;
  return section("dma", "DMA and L2", "Beats are wide (one per cycle at most, D34); bytes follow from the beat width.",
    dmas.length ? table([
      { key: "name", label: "DMA" },
      { key: "beats_read", label: "Beats read", num: true, fmt: int },
      { key: "beats_written", label: "Beats written", num: true, fmt: int },
      { key: "bytes_read", label: "Bytes read", num: true, fmt: int },
      { key: "bytes_written", label: "Bytes written", num: true, fmt: int },
      { key: "max_buffered", label: "Peak buffer (beats)", num: true, fmt: int },
      { key: "busy", label: "Busy cycles", num: true, fmt: (_, r) => int(r.cycles.busy) },
    ], dmas) : null,
    l2.length ? table([
      { key: "name", label: "L2" },
      { key: "reads", label: "Reads (beats)", num: true, fmt: int },
      { key: "writes", label: "Writes (beats)", num: true, fmt: int },
    ], l2) : null);
}

// -- 7. Controller -------------------------------------------------------------

function controllerSection(detail) {
  const { profile, run } = detail;
  const c = profile.controller;
  if (!c) return null;
  const total = profile.total_cycles;
  const waits = c.waits.map((w) => ({ ...w, len: w.last - w.first + 1 }));
  return section("controller", "Controller",
    `${int(c.commands)} commands (${int(c.reads)} reads), ${waits.length} waits, ${int(c.polls)} poll samples. Command cycles are control overhead, kept apart from waiting (D37).`,
    h("div", { class: "row single" }, classBar("controller", c.cycles, total), classCounts("controller", c.cycles, total)),
    h("h3", {}, "Waits"),
    waits.length ? table([
      { key: "pc", label: "pc", num: true },
      { key: "block", label: "Block" }, { key: "mode", label: "Mode" },
      { key: "first", label: "First cycle", num: true, fmt: int },
      { key: "last", label: "Last cycle", num: true, fmt: int },
      { key: "len", label: "Cycles", num: true, fmt: int },
      { key: "done", label: "Block done", num: true, fmt: int },
    ], waits) : h("p", { class: "note" }, "No waits."),
    h("h3", {}, "csr_read values"),
    run.reads.length ? table([
      { key: "cycle", label: "Cycle", num: true, fmt: int },
      { key: "reg", label: "Register" },
      { key: "addr", label: "Address", num: true },
      { key: "value", label: "Value", num: true, fmt: int },
    ], run.reads) : h("p", { class: "note" }, "The program reads no register."));
}

// -- 8. Cluster configuration --------------------------------------------------

function clusterSection(detail) {
  const cl = detail.run.cluster;
  return section("cluster", "Cluster configuration", null,
    h("details", {},
      h("summary", {}, `${detail.run.cluster_file || "inline"}: ${cl.components.length} components, register map with ${Object.keys(detail.run.register_map.blocks || {}).length} blocks`),
      h("pre", {}, JSON.stringify(cl, null, 1))),
    h("details", {},
      h("summary", {}, "Register map"),
      h("pre", {}, JSON.stringify(detail.run.register_map, null, 1))));
}

/** Replace `root`'s content with the report of `detail` (/api/run) and `fifo` (/api/run/fifo). */
export function renderReport(root, detail, fifo) {
  root.replaceChildren(...[
    cyclesSection(detail),
    accelSection(detail),
    streamerSection(detail, fifo),
    memorySection(detail),
    dmaSection(detail),
    controllerSection(detail),
    clusterSection(detail),
  ].filter(Boolean));
}
