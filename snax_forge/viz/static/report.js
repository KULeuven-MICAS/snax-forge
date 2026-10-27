// The profile report of one flow run (D137): the profile (CONTRACTS.md
// section 7), the cluster as the run names it (run/cluster.py) and run.json,
// drawn top to bottom. Works at both trace levels; a core's classes and waits
// need the events (trace task) and say so when missing.
//
// Nothing is computed here that the profile already has; the report only
// arranges numbers. Components appear in the cluster's order: the
// accelerator, its movers, the DMA, the cores.

import { CLASS_GROUP as GROUP, h, int, pct, section, table } from "./dom.js";

// Cycle classes per component kind, in CONTRACTS.md section 7 order.
const CLASS_ORDER = {
  accel: ["busy", "stall_out", "stall_in", "flush", "idle"],
  streamer: ["busy", "stall_xbar", "stall_fifo", "idle"],
  core: ["run", "poll", "barrier", "sleep"],
};

const LEGEND = [
  ["busy", "Busy (a core: run)"], ["mem", "Memory stall (stall_xbar)"],
  ["flow", "Flow stall (stall_fifo, stall_in, stall_out, flush)"],
  ["wait", "A core polling"], ["command", "A core in a barrier"], ["idle", "Idle (a core: sleep)"],
];

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

// -- 1. Where the cycles go ----------------------------------------------------

function cyclesSection(detail) {
  const { profile, trace } = detail;
  const total = profile.total_cycles;
  const rows = [
    ...Object.entries(profile.accelerators).map(([n, a]) => ({ n, kind: "accel", label: "accelerator", cycles: a.cycles, total })),
    ...Object.entries(profile.streamers).map(([n, s]) => ({ n, kind: "streamer", label: s.write ? "writer" : "reader", cycles: s.cycles, total })),
    ...Object.entries(profile.cores).filter(([, c]) => c.cycles).map(([n, c]) => ({ n, kind: "core", label: "core", cycles: c.cycles, total })),
  ];
  const coresNote = trace
    ? ` The cores' too, the same window (boot ${int(trace.boot)} cycles before it).`
    : " The cores' classes need the events (--trace task).";
  return section("cycles", "Where the cycles go",
    `The accelerator's and the movers' ${int(total)} cycles (the program's window) by class.${coresNote} Totals only; when they happen is the Schedule tab.`,
    h("ul", { class: "legend" }, LEGEND.map(([g, label]) => h("li", {}, h("i", { class: `key g-${g}` }), label))),
    h("div", { class: "rows" }, rows.map((r) =>
      h("div", { class: "row" },
        h("div", { class: "who" }, h("b", {}, r.n), h("span", { class: "kind" }, r.label)),
        classBar(r.kind, r.cycles, r.total),
        classCounts(r.kind, r.cycles, r.total)))));
}

// -- 2. The accelerator ----------------------------------------------------------

function accelSection(detail) {
  const { profile, cluster } = detail;
  const names = Object.keys(profile.accelerators);
  if (!names.length) return null;
  const acc = cluster.accelerator;
  const unit = acc.unit;
  const params = ["wait", "latency", "ii", "lanes", "data_width"].filter((k) => k in unit).map((k) => `${k} ${unit[k]}`).join(", ");
  return section("accelerators", "Accelerator",
    "Utilisation is busy cycles over the program's window: a firing and the II gap after it (D59).",
    names.map((n) => {
      const a = profile.accelerators[n];
      return h("div", { class: "block" },
        h("h3", {}, n, h("span", { class: "kind" }, `${unit.unit || ""} on core${acc.core} (${params})`)),
        h("div", { class: "util" },
          h("div", { class: "meter" }, h("span", { class: "g-busy", style: { width: `${100 * a.utilisation}%` } })),
          h("b", {}, `${(100 * a.utilisation).toFixed(1)}%`), ` utilisation, ${int(a.firings)} firings`),
        table(
          [{ key: "port", label: "Port" }, { key: "mover", label: "Mover" }, { key: "dir", label: "Direction" }, { key: "channels", label: "Channels", num: true }],
          acc.movers.map((m) => ({ port: m.port, mover: m.name, dir: m.write ? "write" : "read", channels: m.lanes }))));
    }));
}

// -- 3. Streamers and FIFOs ----------------------------------------------------

function streamerSection(detail) {
  const { profile } = detail;
  const names = Object.keys(profile.streamers);
  if (!names.length) return null;
  return section("streamers", "Streamers and FIFOs",
    "One entry per mover. The FIFO's highest count per lane over the run, against its depth (D137 keeps the maximum only).",
    names.map((n) => {
      const s = profile.streamers[n];
      const f = s.fifo;
      return h("div", { class: "block" },
        h("h3", {}, n, h("span", { class: "kind" }, `${s.write ? "writer" : "reader"}, ${s.ports.length} ports, FIFO ${f.name} depth ${f.depth}`)),
        table([
          { key: "lane", label: "Lane", num: true },
          { key: "port", label: "Port" },
          { key: "max", label: "Highest count", num: true },
          { key: "depth", label: "Depth", num: true },
        ], f.max.map((mx, lane) => ({ lane, port: s.ports[lane], max: mx, depth: f.depth }))));
    }));
}

// -- 4. Cores ------------------------------------------------------------------

function coreSection(detail) {
  const { profile } = detail;
  const cores = Object.entries(profile.cores);
  if (!cores.length) return null;
  return section("cores", "Cores",
    "SNAX CSR accesses over the whole run, and each wait in the window: a poll of a block's CSR (first to last read of one value) or a barrier.",
    cores.map(([n, c]) => h("div", { class: "block" },
      h("h3", {}, n, h("span", { class: "kind" }, `${int(c.csr_writes)} CSR writes, ${int(c.csr_reads)} CSR reads`)),
      c.cycles === null ? h("p", { class: "note" }, "Classes and waits need the events (--trace task).")
        : c.waits.length ? table([
          { key: "block", label: "Waits on" },
          { key: "first", label: "From", num: true, fmt: int },
          { key: "last", label: "To", num: true, fmt: int },
          { key: "cycles", label: "Cycles", num: true, fmt: int },
        ], c.waits.map((w) => ({ ...w, cycles: w.last - w.first + 1 })))
          : h("p", { class: "note" }, "No wait in the window."))));
}

// -- 5. Memory -----------------------------------------------------------------

function memorySection(detail) {
  const { profile, cluster } = detail;
  const b = profile.banks;
  const ports = Object.entries(profile.ports).map(([name, p]) => ({ name, ...p }));
  if (!b && !ports.length) return null;
  const parts = [];
  if (b) {
    const used = b.reads.map((r, i) => r + b.writes[i]);
    const top = Math.max(1, ...used);
    const hot = b.conflicts.map((c) => c > 0);
    parts.push(h("h3", {}, "Accesses per L1 bank (streamers and DMA)"),
      h("div", { class: "strip scroll" },
        used.map((g, i) => h("div", {
          class: `bank${hot[i] ? " hot" : ""}`,
          title: `bank ${i}: ${b.reads[i]} reads, ${b.writes[i]} writes, ${b.conflicts[i]} conflicts`,
        }, h("div", { class: "fill" }, h("i", { style: { height: `${(100 * g) / top}%` } })), h("small", {}, i)))),
      h("div", { class: "scroll" }, h("table", { class: "banks" },
        h("thead", {}, h("tr", {}, h("th", {}, "Bank"), b.reads.map((_, i) => h("th", { class: `num${hot[i] ? " hot" : ""}` }, i)))),
        h("tbody", {}, ["reads", "writes", "conflicts"].map((m) => h("tr", {}, h("th", {}, m),
          b[m].map((v, i) => h("td", { class: `num${hot[i] ? " hot" : ""}${v ? "" : " zero"}` }, int(v)))))))));
  }
  if (ports.length) {
    parts.push(h("h3", {}, "Crossbar ports"),
      table([
        { key: "name", label: "Port" },
        { key: "grants", label: "Grants", num: true, fmt: int },
        { key: "stalls", label: "Stalls", num: true, fmt: int },
        { key: "stalls_dma", label: "Stalls by the DMA", num: true, fmt: int },
        { key: "whole_run", label: "Counted", fmt: (v) => (v ? "whole run, boot included" : "") },
      ], ports, (p) => (p.stalls ? "hot" : null)));
  }
  return section("memory", "Memory",
    `${cluster.banks} banks of ${cluster.bank_bytes} bytes. Banks count the streamers' and the DMA's accesses only; a core's and AXI's port counts from reset.`,
    parts);
}

// -- 6. DMA --------------------------------------------------------------------

function dmaSection(detail) {
  const dmas = Object.entries(detail.profile.dmas).map(([name, d]) => ({ name, ...d }));
  if (!dmas.length) return null;
  return section("dma", "DMA", "Busy cycles are those DMA_BUSY counts; bytes each way by the copies' addresses against the L1.",
    table([
      { key: "name", label: "DMA" },
      { key: "transfers", label: "Transfers", num: true, fmt: int },
      { key: "busy", label: "Busy cycles", num: true, fmt: int },
      { key: "beats_read", label: "Beats read", num: true, fmt: int },
      { key: "beats_written", label: "Beats written", num: true, fmt: int },
      { key: "bytes_l2_to_l1", label: "Bytes L2 → L1", num: true, fmt: int },
      { key: "bytes_l1_to_l2", label: "Bytes L1 → L2", num: true, fmt: int },
    ], dmas));
}

// -- 7. Cluster ----------------------------------------------------------------

function clusterSection(detail) {
  const cl = detail.cluster;
  return section("cluster", "Cluster", null,
    h("details", {},
      h("summary", {}, `${cl.meta?.cluster || "cluster"}: ${cl.accelerator.name} with ${cl.accelerator.movers.length} movers, ${cl.nb_cores} cores`),
      h("pre", {}, JSON.stringify(cl, null, 1))),
    h("details", {},
      h("summary", {}, "run.json"),
      h("pre", {}, JSON.stringify(detail.run, null, 1))));
}

/** Replace `root`'s content with the report of `detail` (/api/run). */
export function renderReport(root, detail) {
  root.replaceChildren(...[
    cyclesSection(detail),
    accelSection(detail),
    streamerSection(detail),
    coreSection(detail),
    memorySection(detail),
    dmaSection(detail),
    clusterSection(detail),
  ].filter(Boolean));
}
