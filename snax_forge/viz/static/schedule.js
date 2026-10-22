// The schedule of one run (D57), with the cluster view under it (cluster.js).
// It draws /api/run/<n>/schedule and nothing else: the lanes, top to bottom
// as the answer orders them, over the segments of the window [0, T), and the
// boot before 0.
//
//   core lane     its class spans (dom.js CLASS_GROUP; sleep left blank), a
//                 thin band of its function spans under them, a tick per CSR
//                 access (a poll's at its first read)
//   DMA lane      one bar per copy, issue to done, with its direction
//   mover, acc    one bar per task, start to done, coloured by its class runs
//                 from beat on (idle left blank), its class totals on hover
//   detail rows   from beat on, inside the open segments in view only, from
//                 /api/run/<n>/schedule/detail: per crossbar port its grants
//                 and refusals (a green strip where read data comes back), the
//                 DMA's beats and bursts, the accelerator's fires and pushes,
//                 each lane's FIFO count (each line holds until the next)
//   tracer rows   the elements traced from the memory tab (hash `tracers`),
//                 straight from /api/run/<n>/journey: a mark per hop (L1 in
//                 the upper half, L2 in the lower) and a diamond per firing
//
// A folded segment is drawn at a fixed width, its length in the ruler and its
// per-core counts on hover; a click opens it (its first cycle joins the hash's
// `open`), and the ruler's ✕ over an opened one folds it again. Open segments
// are drawn at the zoom's pixels per cycle (hash `zoom`, fitted to the width
// by default; the wheel zooms around the pointer, shift + wheel scrolls). A
// click on a cycle selects it (hash `cycle`); the answer opens the fold that
// holds the selected cycle. The arrow keys step to the next or previous event
// cycle, shift + arrow to the next or previous task boundary (/api/run/<n>/step);
// the play button steps as the right arrow does (app.js, player.js). A cycle
// change moves the cursor and updates the cluster view in place.

import { CLASS_GROUP, clampCycle, dec, h, int } from "./dom.js";
import { clusterView, eventList } from "./cluster.js";
import { journeyOf, tracersOf } from "./memory.js";
import { playControl } from "./player.js";
import { DIRECTION_TEXT, eventText, hopText, operandsText } from "./events.js";

const SVG = "http://www.w3.org/2000/svg";
const LANE_H = 28; // px, a lane
const BAND_H = 5; // px, a core's function band
const SUB_H = 16; // px, a detail row
const RULER_H = 24;
const LABEL_W = 170;
const FOLD_W = 30; // px, a folded segment, whatever its length
const MIN_CW = 0.05; // px per cycle, the zoom's limits
const MAX_CW = 64;
const FIT_MAX = 32; // the fitted zoom's largest
const LEVELS = ["off", "task", "beat", "stall", "finegrained"]; // run/trace.py LEVELS
const atLeast = (level, l) => LEVELS.indexOf(level) >= LEVELS.indexOf(l);

/** An SVG element with attributes; `title` becomes a native tooltip. */
function s(tag, attrs = {}, title = null) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  if (title) {
    const t = document.createElementNS(SVG, "title");
    t.textContent = title;
    el.append(t);
  }
  return el;
}

/** Append a text element with `text` to `g`. */
function txt(g, attrs, text) {
  const t = s("text", attrs);
  t.textContent = text;
  g.append(t);
}

const fits = (w, text) => w >= String(text).length * 6 + 6;

// -- data --------------------------------------------------------------------------

// Keyed by the run's detail, as memory.js keys its own: Reload fetches a new
// detail, so a run written again into the same directory is fetched again.
const answers = new WeakMap(); // detail -> Map(query -> Promise)

function cached(detail, key, load) {
  if (!answers.has(detail)) answers.set(detail, new Map());
  const m = answers.get(detail);
  if (!m.has(key)) m.set(key, load());
  return m.get(key);
}

// -- layout: the segments as columns ------------------------------------------------

/** The columns of an answer: the boot, then each segment; a fold FOLD_W wide, an open one cw per cycle. */
function columns(ans, cw) {
  const cols = [];
  let x = 0;
  for (const seg of [ans.boot, ...ans.segments]) {
    const w = seg.folded ? FOLD_W : (seg.to - seg.from) * cw;
    cols.push({ seg, x, w, boot: seg === ans.boot });
    x += w;
  }
  const open = cols.filter((c) => !c.seg.folded);
  /** The open column holding cycle t, or null (t folded or outside the window). */
  const colAt = (t) => {
    let lo = 0;
    let hi = open.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (open[mid].seg.to <= t) lo = mid + 1;
      else hi = mid;
    }
    return lo < open.length && open[lo].seg.from <= t ? open[lo] : null;
  };
  const x0 = (t) => { const c = colAt(t); return c ? c.x + (t - c.seg.from) * cw : null; };
  /** [x1, x2] of each open part of the cycles [a, b). */
  const pieces = (a, b) => open.filter((c) => c.seg.from < b && a < c.seg.to)
    .map((c) => [c.x + (Math.max(a, c.seg.from) - c.seg.from) * cw, c.x + (Math.min(b, c.seg.to) - c.seg.from) * cw]);
  return { cols, open, width: Math.max(x, 1), cw, colAt, x: x0, pieces };
}

// -- rows ---------------------------------------------------------------------------

const sub = (label, title, drawDetail) => ({ label, sub: "", title, height: SUB_H, detail: true, drawDetail });

/** A cell per event at its cycle: `cls(e)` its class, `text(e)` the label drawn when it fits. */
function cells(g, m, evs, cls, text) {
  const cw = m.cw;
  for (const e of evs) {
    const x = m.x(e.t);
    if (x === null) continue;
    g.append(s("rect", { x: x + 0.5, y: 2, width: Math.max(cw - 1, 0.5), height: SUB_H - 7, class: cls(e) }, `cycle ${e.t}: ${eventText(e)}`));
    const label = text(e);
    if (label !== null && fits(cw, label)) txt(g, { x: x + cw / 2, y: SUB_H - 7, class: "cell-text mid on-dark" }, label);
    if (e.op === "r" && e.back !== undefined) { // the read's data back, a strip under its cell
      const xb = m.x(e.back);
      if (xb !== null) g.append(s("rect", { x: xb + 0.5, y: SUB_H - 4, width: Math.max(cw - 1, 0.5), height: 3, class: "s-resp" }, `cycle ${e.back}: read data back (asked in cycle ${e.t})`));
    }
  }
}

/** The rows of the lanes (module doc), in the answer's order, detail rows under their lane. */
function laneRows(detail, ans, showDetail) {
  const T = ans.total_cycles;
  const filter = detail.trace?.filter ?? [];
  const movers = Object.fromEntries(detail.cluster.accelerator.movers.map((m) => [m.name, m]));
  const rows = [];
  for (const lane of ans.lanes) {
    const name = lane.name;
    if (lane.kind === "core") {
      rows.push({ label: name, sub: "core", height: LANE_H, draw(g, m) {
        const top = 3;
        const hgt = LANE_H - BAND_H - 8;
        for (const sp of lane.spans) {
          const group = CLASS_GROUP[sp.class];
          if (!group || group === "idle") continue; // sleep stays blank
          for (const [x1, x2] of m.pieces(sp.from, sp.to)) {
            g.append(s("rect", { x: x1, y: top, width: x2 - x1, height: hgt, class: `s-${group}` }, `${name} ${sp.class}: cycles ${sp.from}–${sp.to - 1}`));
          }
        }
        lane.funcs.forEach((f, i) => {
          for (const [x1, x2] of m.pieces(f.from, f.to)) {
            g.append(s("rect", { x: x1, y: LANE_H - BAND_H - 3, width: x2 - x1, height: BAND_H, class: `fn fn-${i % 2}` }, `${name} in ${f.name ?? "(no function line yet)"}: cycles ${f.from}–${f.to - 1}`));
          }
        });
        for (const e of lane.csr) {
          const x = m.x(e.t);
          if (x === null) continue;
          g.append(s("path", { d: `M${x + m.cw / 2 - 3},0 l6,0 l-3,4 z`, class: `csr-tick${e.k === "poll" ? " poll" : ""}` }, `cycle ${e.t}: ${eventText(e)}`));
        }
      } });
      if (showDetail && filter.includes(name)) {
        rows.push(sub(`${name} port`, `${name}'s crossbar port: grants (teal), refusals (red), read data back (green strip)`,
          (g, d, m) => cells(g, m, d.ports[name] ?? [], (e) => (e.k === "grant" ? "s-req" : "s-mem"), (e) => e.bank)));
      }
    } else if (lane.kind === "dma") {
      rows.push({ label: name, sub: "DMA", height: LANE_H, draw(g, m) {
        for (const b of lane.bars) {
          const to = b.to ?? T - 1;
          const tip = `copy ${b.id} ${b.task}: ${DIRECTION_TEXT[b.dir] ?? b.dir}, ${int(b.bytes)} B; issued in ${b.from}, done in ${b.to ?? "–"}`;
          for (const [x1, x2] of m.pieces(b.from, to + 1)) {
            g.append(s("rect", { x: x1, y: 4, width: x2 - x1, height: LANE_H - 10, class: "s-dma" }, tip));
            const label = DIRECTION_TEXT[b.dir] ?? b.dir;
            if (fits(x2 - x1, label)) txt(g, { x: x1 + 4, y: LANE_H - 10, class: "cell-text on-dark" }, label);
          }
        }
      } });
      if (showDetail) {
        rows.push(sub("dma beats", "the DMA's beats on L1 (its index in the copy), read data back as a green strip",
          (g, d, m) => cells(g, m, d.dma.filter((e) => e.k === "dma"), () => "s-req", (e) => e.i)));
        rows.push(sub("dma bursts", "the DMA's requests to L2 over AXI (r or w), read data back as a green strip",
          (g, d, m) => cells(g, m, d.dma.filter((e) => e.k === "burst"), () => "s-req burst", (e) => e.op)));
      }
    } else {
      rows.push({ label: name, sub: lane.kind === "mover" ? (movers[name]?.write ? "writer" : "reader") : "accelerator", height: LANE_H, draw(g, m) {
        for (const b of lane.bars) {
          const to = b.to ?? T - 1;
          const totals = b.cycles ? Object.entries(b.cycles).map(([k, v]) => `${k} ${int(v)}`).join(", ") : "–";
          const tip = `${b.task}: start in ${b.from}, done in ${b.to ?? "–"}; ${totals}${b.firings !== undefined ? `; ${int(b.firings)} firings` : ""}`;
          for (const r of b.classes ?? []) {
            const group = CLASS_GROUP[r.mode];
            if (!group || group === "idle") continue;
            for (const [x1, x2] of m.pieces(r.from, r.to)) {
              g.append(s("rect", { x: x1, y: 4, width: x2 - x1, height: LANE_H - 10, class: `s-${group}` }, `${b.task} ${r.mode}: cycles ${r.from}–${r.to - 1}\n${tip}`));
            }
          }
          for (const [x1, x2] of m.pieces(b.from, to + 1)) {
            g.append(s("rect", { x: x1, y: 3, width: x2 - x1, height: LANE_H - 8, class: "bar" }, tip));
            if (!b.classes && fits(x2 - x1, b.task)) txt(g, { x: x1 + 4, y: LANE_H - 10, class: "cell-text" }, b.task);
          }
        }
      } });
      if (!showDetail) continue;
      if (lane.kind === "accelerator") {
        rows.push(sub(`${name} fire`, "its firings (n)", (g, d, m) => cells(g, m, d.acc.filter((e) => e.k === "fire"), () => "s-busy", (e) => e.n)));
        rows.push(sub(`${name} push`, "its results handed to the writer (n)", (g, d, m) => cells(g, m, d.acc.filter((e) => e.k === "push"), () => "s-busy push", (e) => e.n)));
      } else {
        const mv = movers[name];
        for (let l = 0; l < (mv?.lanes ?? 0); l++) {
          const port = `${name}.lane${l}`;
          rows.push(sub(port, `${port}: grants (teal, the bank), refusals (red), read data back (green strip)`,
            (g, d, m) => cells(g, m, d.ports[port] ?? [], (e) => (e.k === "grant" ? "s-req" : "s-mem"), (e) => e.bank)));
          rows.push(sub(`${port} FIFO`, `${port}'s FIFO count of ${mv.fifo_depth}, each line holding until the next (stall level)`,
            (g, d, m) => fifoSteps(g, m, d.fifo[port] ?? [], d.to, mv.fifo_depth)));
        }
      }
    }
  }
  return rows;
}

/** A lane's FIFO lines: each count from its cycle to the next line's, or the range's end. */
function fifoSteps(g, m, evs, end, depth) {
  evs.forEach((e, i) => {
    const next = i + 1 < evs.length ? evs[i + 1].t : end;
    if (!e.count) return;
    const hgt = ((SUB_H - 4) * Math.min(e.count, depth)) / depth;
    for (const [x1, x2] of m.pieces(e.t, next)) {
      g.append(s("rect", { x: x1, y: SUB_H - 2 - hgt, width: x2 - x1, height: hgt, class: "s-fifo" }, `cycles ${e.t}–${next - 1}: ${e.src} lane ${e.lane} holds ${e.count} of ${depth}`));
      if (fits(x2 - x1, e.count)) txt(g, { x: x1 + 3, y: SUB_H - 4, class: "cell-text" }, String(e.count));
    }
  });
}

/** One row per traced element (module doc), from its journey. */
function tracerRows(journeys) {
  return journeys.map(({ pick, j }) => {
    const fail = !j || !j.available;
    return {
      label: fail ? pick : j.element, sub: fail ? "not traced" : "tracer", height: LANE_H, tracer: pick,
      title: fail ? (j?.reason ?? "no journey") : `${j.element}: L1 hops above, L2 hops below, firings as diamonds`,
      draw(g, m) {
        if (fail) return;
        const cw = m.cw;
        for (const hop of j.hops) {
          const x = m.x(hop.t);
          if (x === null) continue;
          const top = hop.mem === "l1" ? 2 : LANE_H / 2 + 1;
          const back = hop.act === "data back";
          const cls = back ? "s-resp" : hop.act === "held back" ? "s-mem" : "s-req";
          g.append(s("rect", { x: x + 0.5, y: back ? top + LANE_H / 2 - 6 : top, width: Math.max(cw - 1, 0.5), height: back ? 3 : LANE_H / 2 - 3, class: cls }, `cycle ${hop.t}: ${hopText(hop)}`));
        }
        for (const f of j.firings) {
          const x = m.x(f.t);
          if (x === null) continue;
          const cx = x + cw / 2;
          const r = Math.min(LANE_H / 2 - 2, Math.max(cw / 2, 3));
          g.append(s("path", { d: `M${cx},${LANE_H / 2 - r} l${r},${r} l${-r},${r} l${-r},${-r} z`, class: "s-busy fire-mark" },
            `cycle ${f.t}: firing ${f.n} of ${f.acc} (task ${f.task}) ${f.role} it: ${operandsText(f)}`));
        }
      },
    };
  });
}

// -- drawing -----------------------------------------------------------------------

/** Tick step so that labels are at least 44 px apart. */
function tickStep(cw) {
  for (const k of [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]) if (k * cw >= 44) return k;
  return 100000;
}

/** A fold's tooltip: its cycles, length and per-core counts. */
function foldTip(seg, boot) {
  if (boot) return `boot: ${int(seg.cycles)} cycles before the window (main's first mcycle read); never opens`;
  const cores = Object.entries(seg.cores ?? {}).map(([core, c]) =>
    `${core}: ${Object.entries(c.cycles).map(([k, v]) => `${k} ${int(v)}`).join(", ")}; ${c.csr_writes} CSR writes, ${c.csr_reads} reads; in ${c.funcs.join(", ") || "–"}`);
  return [`cycles ${seg.from}–${seg.to - 1} folded (${int(seg.to - seg.from)} cycles): no task or copy open`, ...cores, "Click to open"].join("\n");
}

/** The chart: ruler, folds, rows; returns the svg, the cursor and each detail row's group. */
function chart(rows, m, ans, onPick, onFold) {
  const height = RULER_H + rows.reduce((a, r) => a + r.height, 0);
  const svg = s("svg", { width: m.width, height, viewBox: `0 0 ${m.width} ${height}`, class: "sched" });
  const defs = s("defs");
  const pat = s("pattern", { id: "fold-hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
  pat.append(s("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "hatch" }));
  defs.append(pat);
  svg.append(defs);

  let y = RULER_H;
  rows.forEach((r, i) => {
    svg.append(s("rect", { x: 0, y, width: m.width, height: r.height, class: `row-bg${r.detail ? " sub" : ""}${i % 2 ? " odd" : ""}` }));
    y += r.height;
  });
  const ruler = s("g", {});
  const step = tickStep(m.cw);
  for (const c of m.cols) {
    const seg = c.seg;
    if (seg.folded) {
      svg.append(s("rect", { x: c.x, y: 0, width: c.w, height, class: `fold${c.boot ? " boot" : ""}` }, foldTip(seg, c.boot)));
      txt(ruler, { x: c.x + c.w / 2, y: RULER_H - 8, class: "ruler-text mid" }, c.boot ? "boot" : int(seg.to - seg.from));
      continue;
    }
    ruler.append(s("line", { x1: c.x, x2: c.x, y1: 0, y2: height, class: "seg-edge" }));
    for (let t = Math.ceil(seg.from / step) * step; t < seg.to; t += step) {
      const x = m.x(t);
      ruler.append(s("line", { x1: x, x2: x, y1: RULER_H - 6, y2: height, class: "tick" }));
      txt(ruler, { x: x + 2, y: RULER_H - 9, class: "ruler-text" }, String(t));
    }
    if (m.cw >= 6) for (let t = seg.from; t < seg.to; t++) if (t % step) ruler.append(s("line", { x1: m.x(t), x2: m.x(t), y1: RULER_H, y2: height, class: "grid" }));
    if (seg.foldable) { // opened: a ✕ in the ruler folds it again
      const b = s("g", { class: "refold" }, `Fold cycles ${seg.from}–${seg.to - 1} again`);
      b.append(s("rect", { x: c.x + 1, y: 1, width: 14, height: 12, rx: 2 }));
      txt(b, { x: c.x + 8, y: 11, class: "ruler-text mid" }, "✕");
      b.addEventListener("click", (ev) => { ev.stopPropagation(); onFold(seg.from, false); });
      ruler.append(b);
    }
  }
  svg.append(ruler);

  y = RULER_H;
  for (const r of rows) {
    const g = s("g", { transform: `translate(0,${y})` });
    if (r.draw) r.draw(g, m);
    r.g = g;
    svg.append(g);
    y += r.height;
  }
  const cursor = s("rect", { x: 0, y: 0, width: Math.max(m.cw, 2), height, class: "cursor", visibility: "hidden" });
  svg.append(cursor);
  svg.addEventListener("click", (ev) => {
    const px = ev.clientX - svg.getBoundingClientRect().left;
    const c = m.cols.find((x) => px >= x.x && px < x.x + x.w);
    if (!c || c.boot) return;
    if (c.seg.folded) onFold(c.seg.from, true);
    else onPick(Math.min(c.seg.from + Math.floor((px - c.x) / m.cw), c.seg.to - 1));
  });
  return { svg, cursor };
}

function labels(rows, setHash, tracers) {
  const tracerLabel = (r) => [
    h("button", { type: "button", class: "lab-link", title: `Open ${r.label} in the memory tab`, onclick: () => setHash({ view: "memory", pick: r.tracer }) }, r.label),
    h("small", {}, r.sub),
    h("button", { type: "button", class: "lab-x", "aria-label": `Stop tracing ${r.label}`, title: "Stop tracing",
      onclick: () => setHash({ tracers: tracers.filter((x) => x !== r.tracer).join(",") || null }) }, "×"),
  ];
  return h("div", { class: "sched-labels", style: { width: `${LABEL_W}px` } },
    h("div", { style: { height: `${RULER_H}px` }, class: "ruler-label" }, "cycle"),
    rows.map((r) => h("div", { class: `lab${r.detail ? " sub" : ""}${r.tracer ? " tracer" : ""}`, style: { height: `${r.height}px` }, title: r.title ?? r.label },
      r.tracer ? tracerLabel(r) : [h("span", {}, r.label), r.sub ? h("small", {}, r.sub) : null])));
}

// -- the schedule on screen --------------------------------------------------------
//
// `current` is what the last full render left on the page, so a cycle change
// or a scroll can update it in place.

let current = null;

/** Fetch and draw the detail rows of each open segment in view, once per render. */
function loadDetail(c) {
  if (!c.detailRows.length) return;
  const left = c.scroller.scrollLeft;
  const right = left + c.scroller.clientWidth - LABEL_W;
  for (const col of c.m.open) {
    if (col.x + col.w < left || col.x > right) continue;
    const key = `${col.seg.from}|${col.seg.to}`;
    if (c.loaded.has(key)) continue;
    c.loaded.add(key);
    cached(c.detail, `detail|${key}`, () => c.api.scheduleDetail(c.detail.name, col.seg.from, col.seg.to)).then((d) => {
      if (current !== c) return;
      if (!d.available) {
        c.detailNote.textContent = d.reason;
        return;
      }
      for (const r of c.detailRows) r.drawDetail(r.g, d, c.m);
    }).catch((e) => { c.detailNote.textContent = String(e.message || e); });
  }
}

/** Move the cursor, and fill the cluster view and the event list for `cycle` (null clears them). */
async function showCycle(cycle, keepInView = true) {
  const c = current;
  c.cycle = cycle;
  const x = cycle === null ? null : c.m?.x(cycle) ?? null;
  if (c.cursor) {
    c.cursor.setAttribute("visibility", x === null ? "hidden" : "visible");
    if (x !== null) {
      c.cursor.setAttribute("x", String(x));
      const sc = c.scroller;
      const plotW = sc.clientWidth - LABEL_W;
      if (keepInView && plotW > 0 && (x < sc.scrollLeft || x + c.m.cw > sc.scrollLeft + plotW)) sc.scrollLeft = Math.max(x - plotW / 2, 0);
    }
  }
  if (cycle === null) {
    c.cs.cluster?.update(null);
    c.cs.title.textContent = "Cluster";
    c.cs.events.replaceChildren(h("p", { class: "note" }, "Select a cycle."));
    return;
  }
  const a = await cached(c.detail, `cluster|${cycle}`, () => c.api.cluster(c.detail.name, cycle));
  if (current !== c || c.cycle !== cycle) return; // a newer cycle or render took over
  c.cs.cluster.update(a);
  c.cs.title.textContent = `Cluster at cycle ${int(cycle)}`;
  c.cs.events.replaceChildren(eventList(a));
}

/**
 * A change of the hash's `cycle` only: update in place if the schedule of
 * `detail` is on screen and the cycle lies in an open segment. Returns false
 * when a full render is needed instead (the cycle is in a fold).
 */
export async function selectCycle(detail, st) {
  if (!current || current.detail !== detail) return false;
  const cycle = cycleOf(st, current.total);
  if (cycle !== null && current.m && current.m.x(cycle) === null) return false;
  await showCycle(cycle);
  return true;
}

/** The cycle /step gives from the shown one, d 1 or -1, by event or boundary; null with none to step to. */
export async function stepSchedule(d, by) {
  const c = current;
  if (!c || !c.available) return null;
  const from = c.cycle ?? (d > 0 ? -1 : c.total);
  const r = await c.api.step(c.detail.name, from, d, by);
  return r.t === from ? null : r.t;
}

// -- wheel zoom (D58) --------------------------------------------------------------
//
// Wheel events come in bursts; they are gathered for a short moment and
// applied as one zoom change, which keeps the cycle under the pointer where
// it was. The anchor survives the redraw in `anchorNext`.

let anchorNext = null; // {t, px}: put cycle t at px from the plot's left edge after the redraw
const wheel = { factor: 1, anchor: null, timer: null };

function onWheel(ev, view) {
  if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) return; // sideways: native scroll
  ev.preventDefault();
  const { scroller, m } = view;
  const dy = ev.deltaY * (ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? 400 : 1);
  wheel.factor *= Math.exp(-dy * 0.002);
  if (!wheel.anchor) {
    const px = Math.max(ev.clientX - scroller.getBoundingClientRect().left - scroller.clientLeft - LABEL_W, 0);
    const x = scroller.scrollLeft + px;
    const c = m.open.find((k) => x >= k.x && x < k.x + k.w);
    wheel.anchor = c ? { t: c.seg.from + (x - c.x) / m.cw, px } : null;
  }
  clearTimeout(wheel.timer);
  wheel.timer = setTimeout(() => {
    const next = Math.min(Math.max(m.cw * wheel.factor, MIN_CW), MAX_CW).toFixed(3);
    const anchor = wheel.anchor;
    wheel.factor = 1;
    wheel.anchor = null;
    if (Math.abs(Number(next) - m.cw) < 1e-3) return; // at a limit: nothing to redraw
    anchorNext = anchor;
    view.setHash({ zoom: next });
  }, 60);
}

// -- the view ----------------------------------------------------------------------

function cycleOf(st, total) {
  if (st.cycle === undefined || st.cycle === "") return null;
  const c = Number.parseInt(st.cycle, 10);
  return Number.isFinite(c) ? clampCycle(c, total) : null;
}

const opensOf = (st) => [...new Set((st.open ?? "").split(",").map((x) => Number.parseInt(x, 10)).filter(Number.isFinite))];

/** The cluster section: heading with the step buttons and play, the cluster view, the cycle's lines. */
function clusterSection(setHash) {
  const go = (d, by) => async () => {
    const n = await stepSchedule(d, by);
    if (n !== null) setHash({ cycle: n });
  };
  const btn = (label, d, by, what) => h("button", { type: "button", onclick: go(d, by), "aria-label": what, title: what }, label);
  const title = h("h2", {}, "Cluster");
  const holder = h("div", {});
  const events = h("div", {});
  const el = h("section", { id: "cluster" },
    h("div", { class: "cycle-head" },
      btn("«", -1, "boundary", "Previous task boundary (shift + ←)"), btn("‹", -1, "event", "Previous event (←)"),
      title,
      btn("›", 1, "event", "Next event (→)"), btn("»", 1, "boundary", "Next task boundary (shift + →)"),
      playControl()),
    holder,
    h("details", { class: "cycle-events", open: true }, h("summary", {}, "Lines in this cycle"), events));
  return { el, title, holder, events, cluster: null };
}

/**
 * Draw the schedule of `detail` and the cluster view under it into `root`.
 * ctx: {api, st (hash state), setHash}.
 */
export async function renderSchedule(root, detail, ctx) {
  const { api, st, setHash } = ctx;
  const total = detail.total_cycles;
  const opened = opensOf(st);
  const cycle = cycleOf(st, total);
  const query = `schedule|${opened.join(",")}|${cycle ?? ""}`;
  const [ans, first] = await Promise.all([
    cached(detail, query, () => api.schedule(detail.name, opened, cycle)),
    cached(detail, `cluster|${cycle ?? 0}`, () => api.cluster(detail.name, cycle ?? 0)),
  ]);
  const cs = clusterSection(setHash);
  cs.cluster = clusterView(detail, first);
  cs.holder.replaceChildren(cs.cluster.el);

  if (!ans.available) {
    root.replaceChildren(h("section", { id: "schedule" }, h("h2", {}, "Schedule"), h("p", { class: "note" }, `No schedule: ${ans.reason}. Run the flow again with --trace task or above.`)), cs.el);
    current = { detail, api, total, available: false, cs, m: null, cursor: null, cycle: null };
    cs.cluster.update(null);
    cs.title.textContent = "Cluster";
    cs.events.replaceChildren(h("p", { class: "note" }, "No lines at trace off."));
    return;
  }

  const beat = atLeast(ans.level, "beat");
  const showDetail = beat && st.detail !== "0";
  const tracers = beat ? tracersOf(st) : [];
  const journeys = await Promise.all(tracers.map(async (pick) => {
    try {
      return { pick, j: await journeyOf(api, detail, pick) };
    } catch (e) {
      return { pick, j: { available: false, reason: String(e.message || e) } };
    }
  }));
  const rows = [...tracerRows(journeys), ...laneRows(detail, ans, showDetail)];

  // The zoom: the hash's, or the open cycles fitted to the width the folds leave.
  const cst = getComputedStyle(root);
  const pad = (Number.parseFloat(cst.paddingLeft) || 0) + (Number.parseFloat(cst.paddingRight) || 0);
  const avail = Math.max((root.clientWidth || 1100) - pad - LABEL_W - 4, 200);
  const openCycles = ans.segments.filter((x) => !x.folded).reduce((a, x) => a + x.to - x.from, 0);
  const folds = 1 + ans.segments.filter((x) => x.folded).length;
  const fit = Math.min(Math.max((avail - folds * FOLD_W) / Math.max(openCycles, 1), MIN_CW), FIT_MAX);
  const cw = st.zoom ? Math.min(Math.max(Number(st.zoom), MIN_CW), MAX_CW) : fit;
  const m = columns(ans, cw);

  const setOpen = (from, open) => setHash({ open: (open ? [...opened, from] : opened.filter((x) => x !== from)).join(",") || null });
  const zoom = (f) => setHash({ zoom: Math.min(Math.max(cw * f, MIN_CW), MAX_CW).toFixed(3) });
  const nFolded = ans.segments.filter((x) => x.folded).length;
  const note = `${int(total)} cycles in the window, after ${int(ans.boot.cycles)} cycles of boot. ${nFolded} quiet stretch${nFolded === 1 ? "" : "es"} of at least ${ans.fold_min} cycles folded: hover for what the cores did, click to open. Click a cycle, or step with the arrow keys (shift: task boundaries), to show it in the cluster view below.`;
  const detailNote = h("p", { class: "note" }, beat ? "" : `Traced at ${ans.level}: run with --trace beat or above for the detail rows.`);
  const tracerNote = !beat ? null : h("p", { class: "note" }, tracers.length
    ? "Tracer rows: L1 hops in the upper half, L2 hops in the lower half; teal a request, green read data back, red held back, a diamond a firing. Click a label to open the element in the Memory tab."
    : "To follow one element through the run, pick a word in the Memory tab and trace it in the schedule.");
  const legend = h("ul", { class: "legend" },
    [["busy", "Busy, run, fire, push, read data back (strip)"], ["req", "Request (accepted), DMA copy"], ["mem", "Memory stall, refused"], ["flow", "Flow stall"], ["command", "Barrier"], ["wait", "Poll"], ["fifo", "FIFO count"]]
      .map(([g, label]) => h("li", {}, h("i", { class: `key g-${g}` }), label)),
    h("li", {}, h("i", { class: "key fold-key" }), "Folded (fixed width)"));
  const controls = h("div", { class: "sched-controls" },
    h("button", { type: "button", onclick: () => zoom(1 / 1.5), "aria-label": "Zoom out" }, "−"),
    h("button", { type: "button", onclick: () => zoom(1.5), "aria-label": "Zoom in" }, "+"),
    h("button", { type: "button", onclick: () => setHash({ zoom: null }) }, "Fit"),
    h("button", { type: "button", disabled: !opened.length, onclick: () => setHash({ open: null }) }, "Fold all again"),
    h("label", {}, h("input", { type: "checkbox", checked: showDetail, disabled: !beat,
      onchange: (e) => setHash({ detail: e.target.checked ? null : 0 }) }), " Detail rows"),
    h("span", { class: "zoom-read" }, `${dec(cw, cw < 1 ? 2 : 1)} px per cycle in the open segments`));

  const keepLeft = root.querySelector(".sched-scroll")?.scrollLeft ?? 0; // survive a redraw
  const keepTop = root.querySelector(".sched-scroll")?.scrollTop ?? 0;
  const { svg, cursor } = chart(rows, m, ans, (t) => setHash({ cycle: t }), setOpen);
  const scroller = h("div", { class: "sched sched-scroll" }, labels(rows, setHash, tracers), svg);
  root.replaceChildren(
    h("section", { id: "schedule" }, h("h2", {}, "Schedule"), h("p", { class: "note" }, note), tracerNote, legend, controls, detailNote, scroller),
    cs.el);
  scroller.addEventListener("wheel", (ev) => onWheel(ev, { scroller, m, setHash }), { passive: false });
  scroller.scrollTop = keepTop;
  const zoomed = !!anchorNext;
  if (anchorNext) { // a wheel zoom: keep the cycle under the pointer in place
    const x = m.x(Math.floor(anchorNext.t));
    scroller.scrollLeft = Math.max((x ?? 0) - anchorNext.px, 0);
    anchorNext = null;
  } else {
    scroller.scrollLeft = keepLeft;
  }
  const c = { detail, api, total, available: true, cs, m, cursor, scroller, cycle: null, loaded: new Set(), detailNote, detailRows: rows.filter((r) => r.drawDetail) };
  current = c;
  let timer = null;
  scroller.addEventListener("scroll", () => { clearTimeout(timer); timer = setTimeout(() => loadDetail(c), 120); });
  await showCycle(cycle, !zoomed);
  loadDetail(c);
}
