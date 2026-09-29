// The schedule of one run (VIS2, D57), HLS-schedule style, with the cluster
// view (VIS3, D61) under it: one row per component in registration order,
// one column per cycle, over a cycle window [from, to). Built from what the
// run already wrote (D39, D40):
//
//   main row      the component's class runs (trace intervals); idle is left
//                 blank, so the rows read like a Gantt chart. Under it a thin
//                 line per task, from its start to its done. The controller's
//                 row shows each command (cmd events) with its register.
//                 A DMA's tasks carry their direction (L2 → L1, L1 → L2, D58).
//   detail rows   beat level only, for components that ran a task: one per
//                 xbar port the component owns, labelled `<port>.target_banks`
//                 (`<owner>.target_banks` for a single port, D58): accepted
//                 requests (teal) and stalls (red) with bank numbers when
//                 there is room, and a green strip under the cell when read
//                 data comes back (resp, D62); its FIFO (largest lane count),
//                 its firings or its DMA requests (with the L2 read data
//                 coming back under the read requests).
//
// The chart scrolls in one box with the row labels pinned on the left, and
// its height is capped so the cluster view fits under it. The mouse wheel
// over the chart zooms around the pointer; shift + wheel and a sideways
// swipe scroll (D58).
//
// Clicking a cycle selects it (hash `cycle`); the arrow keys step it (app.js).
// A cycle change only moves the cursor and updates the cluster view and the
// cycle's event list in place (selectCycle); a selected cycle outside the
// window moves the window to it, keeping its width (D61).
//
// Tracer rows (VIS4b, D97): the elements traced from the memory tab (hash
// `tracers=A:5,B:5`) get a row each at the top, from their journey: a thin
// line over the element's life, L1 hops in the upper half and L2 hops in the
// lower half (request teal, read data back green, held back red), and each
// firing that consumed or produced it as a green diamond. The label opens the
// element in the memory tab; × stops tracing it. Beat-level runs only.
//
// Data: the class intervals and each block's tasks (start paired with done,
// D60) come with the run detail; task events and FIFO counts are fetched once
// per run (events.js); beat events only for the window and for the selected
// cycle. Events come through /api/run/<n>/events with its k filter (D57).

import { dec, h, int } from "./dom.js";
import { clusterView } from "./cluster.js";
import { elementAt, ensureRows, journeyOf, tracersOf } from "./memory.js";
import { BEAT_KINDS, GROUP, beatTraced, byOwner, classAt, describe, fifoCount, fifoIndex, onPort, taskEvents } from "./events.js";

const SVG = "http://www.w3.org/2000/svg";
const MAIN_H = 24; // px, main row
const SUB_H = 17; // px, detail row
const RULER_H = 22;
const LABEL_W = 150;
const BEAT_WINDOW = 400; // default window of a beat-level run
const MAX_CW = 64; // px per cycle, largest zoom
const DIR_TEXT = { l2_to_l1: "L2 → L1", l1_to_l2: "L1 → L2" };


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

// -- data --------------------------------------------------------------------------

const beatsOf = new Map(); // "name|path|from|to" -> beat events of the window

/** Beat events in [from, to). */
async function beatEvents(api, detail, from, to) {
  const key = `${detail.name}|${detail.path}|${from}|${to}`;
  if (!beatsOf.has(key)) beatsOf.set(key, await api.events(detail.name, from, to, { k: BEAT_KINDS }));
  return beatsOf.get(key);
}

// -- rows -------------------------------------------------------------------------

/** The rows to draw, in order: [{name, label, sub, height, draw(g, x, cw)}]. */
function buildRows(detail, win, tasks, beats, fifo, showDetail) {
  const { profile, run, trace } = detail;
  const { from, to } = win;
  const clip = (a, b) => [Math.max(a, from), Math.min(b, to)];
  const intervals = trace.intervals;
  const cmds = tasks.filter((e) => e.k === "cmd");
  const beatBy = byOwner(beats ?? [], profile);

  const rows = [];
  for (const spec of run.cluster.components) {
    const name = spec.name;
    const runs = intervals[name];
    if (!runs) continue; // the xbar has no classes; its work shows on the port rows
    const isCtl = spec.kind === "controller";
    const mySpans = detail.tasks[name] ?? []; // {start, done[, direction]} (D60)
    rows.push({
      name, label: name, sub: spec.kind === "accel" ? spec.accel : spec.kind, height: MAIN_H,
      draw(g, x, cw) {
        for (const [cls, a0, b0] of runs) {
          if (!GROUP[cls]) continue; // idle stays blank
          const [a, b] = clip(a0, b0);
          if (b <= a) continue;
          g.append(s("rect", { x: x(a), y: 3, width: (b - a) * cw, height: MAIN_H - 9, class: `s-${GROUP[cls]}` },
            `${name} ${cls}: cycles ${a0}–${b0 - 1} (${b0 - a0})`));
        }
        if (isCtl) {
          for (const c of cmds) {
            const [a, b] = clip(c.t, c.last + 1);
            if (b <= a) continue;
            g.append(s("rect", { x: x(a), y: 3, width: (b - a) * cw, height: MAIN_H - 9, class: "cmd-edge" }, describe(c)));
            const text = c.op === "wait" ? `wait ${c.block}` : (c.reg ?? "").split(".").pop();
            if ((b - a) * cw >= text.length * 6 + 6) {
              txt(g, { x: x(a) + 3, y: MAIN_H - 10, class: "cell-text on-dark" }, text);
            }
          }
        }
        for (const sp of mySpans) {
          const [a, b] = clip(sp.start, sp.done);
          if (b < a) continue;
          g.append(s("line", { x1: x(a), x2: x(Math.max(b, a)), y1: MAIN_H - 3, y2: MAIN_H - 3, class: "task" },
            `${name} task: start in ${sp.start}, done in ${sp.done}`));
          if (sp.start >= from && sp.start < to) g.append(s("path", { d: `M${x(sp.start)},${MAIN_H - 7} l4,4 l-4,4 z`, class: "task-mark" }, `${name} start in ${sp.start}`));
          if (sp.done >= from && sp.done <= to) g.append(s("rect", { x: x(sp.done) - 1, y: MAIN_H - 8, width: 2, height: 8, class: "task-mark" }, `${name} done in ${sp.done}`));
          if (sp.direction && b > a) { // a DMA task's direction (D58)
            const text = DIR_TEXT[sp.direction] ?? sp.direction;
            if ((b - a) * cw >= text.length * 6 + 8) txt(g, { x: x(a) + 4, y: MAIN_H - 10, class: "cell-text on-dark" }, text);
          }
        }
      },
    });
    // Detail rows only for a component that ran a task (an unused block adds
    // nothing), and for the controller, which never starts but polls.
    if (!showDetail || (!mySpans.length && !isCtl)) continue;
    const mine = beatBy[name] ?? [];

    // One row per xbar port the component owns, named after what it shows (D58).
    const ports = Object.entries(profile.ports).filter(([, p]) => p.owner === name);
    for (const [port, p] of ports) {
      const evs = mine.filter((e) => onPort(e) && e.k !== "resp" && e.port === port);
      const back = mine.filter((e) => e.k === "resp" && e.port === port);
      rows.push({
        name, label: `${ports.length === 1 ? name : port}.target_banks`, sub: `${p.width} bits`,
        title: `port ${port}, ${p.width} bits: the bank of each request and stall; a green strip where read data comes back`, height: SUB_H, detail: true,
        draw(g, x, cw) {
          for (const e of back) { // under the request cells, so both show in a cycle with both
            g.append(s("rect", { x: x(e.t) + 0.5, y: SUB_H - 4, width: Math.max(cw - 1, 0.5), height: 3, class: "s-resp" }, `cycle ${e.t}: ${describe(e)}`));
          }
          for (const e of evs) {
            const cls = e.k === "grant" ? "s-req" : e.wider ? "s-mem wider" : "s-mem";
            g.append(s("rect", { x: x(e.t) + 0.5, y: 2, width: Math.max(cw - 1, 0.5), height: SUB_H - 7, class: cls }, `cycle ${e.t}: ${describe(e)}`));
            const text = e.banks.length > 1 ? `${e.banks[0]}+` : String(e.banks[0]);
            if (cw >= text.length * 6 + 4) txt(g, { x: x(e.t) + cw / 2, y: SUB_H - 7, class: "cell-text mid on-dark" }, text);
          }
        },
      });
    }

    const fifoName = profile.streamers[name]?.fifo.name;
    if (fifoName) {
      const depth = profile.streamers[name].fifo.depth;
      const lanes = profile.streamers[name].fifo.hist.length;
      const counts = Array.from({ length: lanes }, (_, l) => fifoCount(fifo, trace, fifoName, l, from)); // null: unknown
      const changes = mine.filter((e) => e.k === "fifo");
      rows.push({
        name, label: fifoName, sub: `depth ${depth}, largest lane`, height: SUB_H, detail: true,
        draw(g, x, cw) {
          // Step through the window: the count of each lane holds until its next fifo event.
          // A cycle the beat filter dropped (D49) and a lane whose count is unknown are left out.
          let i = 0;
          const cur = [...counts];
          for (let t = from; t < to; t++) {
            while (i < changes.length && changes[i].t <= t) { cur[changes[i].lane] = changes[i].count; i++; }
            if (!beatTraced(trace, fifoName, t)) continue;
            const known = cur.filter((c) => c !== null);
            const top = known.length ? Math.max(...known) : 0;
            if (!top) continue;
            const hgt = ((SUB_H - 4) * Math.min(top, depth)) / depth;
            g.append(s("rect", { x: x(t), y: SUB_H - 2 - hgt, width: cw, height: hgt, class: "s-fifo" },
              `cycle ${t}: ${fifoName} lanes hold ${cur.map((c) => (c === null ? "?" : c)).join(", ")} (depth ${depth})`));
            if (cw >= 10) txt(g, { x: x(t) + cw / 2, y: SUB_H - 5, class: "cell-text mid" }, String(top));
          }
        },
      });
    }

    const fires = mine.filter((e) => e.k === "fire");
    if (spec.kind === "accel") {
      rows.push({
        name, label: `${name} firings`, sub: "", height: SUB_H, detail: true,
        draw(g, x, cw) {
          for (const e of fires) {
            g.append(s("rect", { x: x(e.t) + 0.5, y: 2, width: Math.max(cw - 1, 0.5), height: SUB_H - 4, class: "s-busy" }, `cycle ${e.t}: firing ${e.n}`));
            if (cw >= String(e.n).length * 6 + 4) txt(g, { x: x(e.t) + cw / 2, y: SUB_H - 5, class: "cell-text mid on-dark" }, String(e.n));
          }
        },
      });
    }

    if (spec.kind === "dma") {
      for (const side of ["src", "dst"]) {
        const evs = mine.filter((e) => e.k === "dma_beat" && e.side === side);
        const back = side === "src" ? mine.filter((e) => e.k === "resp" && !e.port) : []; // L2 read data (D62)
        rows.push({
          name, label: `${name} ${side === "src" ? "reads" : "writes"}`, sub: "requests", height: SUB_H, detail: true,
          title: side === "src" ? "read requests by beat; a green strip where L2 read data comes back" : "write requests by beat",
          draw(g, x, cw) {
            for (const e of back) {
              g.append(s("rect", { x: x(e.t) + 0.5, y: SUB_H - 4, width: Math.max(cw - 1, 0.5), height: 3, class: "s-resp" }, `cycle ${e.t}: ${describe(e)}`));
            }
            for (const e of evs) {
              g.append(s("rect", { x: x(e.t) + 0.5, y: 2, width: Math.max(cw - 1, 0.5), height: SUB_H - 7, class: "s-req" }, `cycle ${e.t}: ${describe(e)}`));
              if (cw >= String(e.i).length * 6 + 4) txt(g, { x: x(e.t) + cw / 2, y: SUB_H - 7, class: "cell-text mid on-dark" }, String(e.i));
            }
          },
        });
      }
    }

    if (isCtl) {
      const polls = mine.filter((e) => e.k === "poll");
      if (polls.length) {
        rows.push({
          name, label: `${name} polls`, sub: "", height: SUB_H, detail: true,
          draw(g, x, cw) {
            for (const e of polls) {
              g.append(s("circle", { cx: x(e.t) + cw / 2, cy: SUB_H / 2, r: Math.min(4, Math.max(cw / 2 - 1, 1.5)), class: e.value ? "s-wait" : "s-busy" }, `cycle ${e.t}: ${describe(e)}`));
            }
          },
        });
      }
    }
  }
  return rows;
}

// -- tracer rows ---------------------------------------------------------------------

const HOP_CLASS = { read: "s-req", write: "s-req", "data back": "s-resp", "held back": "s-mem" };
const HOP_TEXT = { read: "R", write: "W", "data back": "", "held back": "✕" };

function hopTip(x) {
  const where = x.mem === "l1" ? `bank ${x.bank}, row ${x.row}` : `beat ${x.beat}`;
  const served = x.served ? `; bank served ${x.served.port} for ${x.served.element ?? `row ${x.served.row}`}` : "";
  return `cycle ${x.t}: ${x.element ?? ""} ${x.mem.toUpperCase()} ${x.act} by ${x.by} (${where})${served}`;
}

function fireTip(f) {
  const ins = Object.entries(f.inputs).map(([p, xs]) => `${p} ${xs.join(", ")}`).join("; ");
  const outs = Object.entries(f.outputs).map(([p, xs]) => `${p} ${xs.join(", ")}`).join("; ");
  return `cycle ${f.t}: firing ${f.n} of ${f.acc} (task ${f.task}) ${f.role} it: ${ins}${outs ? ` → ${outs}` : ""}`;
}

/** One row per traced element (module doc), from its journey. */
function tracerRows(journeys, win) {
  const { from, to } = win;
  return journeys.map(({ pick, j }) => {
    const fail = !j || !j.available;
    return {
      name: pick, label: fail ? pick : j.element, sub: fail ? "not traced" : "tracer", height: MAIN_H, tracer: pick,
      title: fail ? (j?.reason ?? "no journey") : `${j.element}: L1 hops above, L2 hops below, firings as diamonds`,
      draw(g, x, cw) {
        if (fail) return;
        const ts = [...j.hops.map((h) => h.t), ...j.firings.map((f) => f.t)];
        if (!ts.length) return;
        const a = Math.max(Math.min(...ts), from);
        const b = Math.min(Math.max(...ts) + 1, to);
        if (b > a) g.append(s("line", { x1: x(a), x2: x(b), y1: MAIN_H / 2, y2: MAIN_H / 2, class: "task" }, `${j.element}: cycles ${Math.min(...ts)}–${Math.max(...ts)}`));
        for (const hop of j.hops) {
          if (hop.t < from || hop.t >= to) continue;
          const top = hop.mem === "l1" ? 2 : MAIN_H / 2 + 1;
          const hgt = hop.act === "data back" ? 3 : MAIN_H / 2 - 3;
          const y = hop.act === "data back" ? top + MAIN_H / 2 - 6 : top;
          g.append(s("rect", { x: x(hop.t) + 0.5, y, width: Math.max(cw - 1, 0.5), height: hgt, class: HOP_CLASS[hop.act] ?? "s-req" }, hopTip(hop)));
          const text = HOP_TEXT[hop.act];
          if (text && cw >= 10) txt(g, { x: x(hop.t) + cw / 2, y: top + MAIN_H / 2 - 5, class: "cell-text mid on-dark" }, text);
        }
        for (const f of j.firings) {
          if (f.t < from || f.t >= to) continue;
          const cx = x(f.t) + cw / 2;
          const r = Math.min(MAIN_H / 2 - 2, Math.max(cw / 2, 3));
          g.append(s("path", { d: `M${cx},${MAIN_H / 2 - r} l${r},${r} l${-r},${r} l${-r},${-r} z`, class: "s-busy fire-mark" }, fireTip(f)));
        }
      },
    };
  });
}

// -- drawing -----------------------------------------------------------------------

/** Tick step so that labels are at least 44 px apart. */
function tickStep(cw) {
  for (const m of [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]) if (m * cw >= 44) return m;
  return 100000;
}

/** The chart and its cursor (moved in place by selectCycle). */
function chart(rows, win, cw, onPick) {
  const { from, to } = win;
  const n = to - from;
  const width = Math.max(n * cw, 1);
  const height = RULER_H + rows.reduce((a, r) => a + r.height, 0);
  const x = (t) => (t - from) * cw;
  const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, class: "sched" });

  // Row backgrounds, then the cycle grid, then the rows.
  let y = RULER_H;
  rows.forEach((r, i) => {
    svg.append(s("rect", { x: 0, y, width, height: r.height, class: `row-bg${r.detail ? " sub" : ""}${i % 2 ? " odd" : ""}` }));
    y += r.height;
  });
  const step = tickStep(cw);
  const ruler = s("g", {});
  for (let t = Math.ceil(from / step) * step; t < to; t += step) {
    ruler.append(s("line", { x1: x(t), x2: x(t), y1: RULER_H - 6, y2: height, class: "tick" }));
    txt(ruler, { x: x(t) + 2, y: RULER_H - 9, class: "ruler-text" }, String(t));
  }
  if (cw >= 6) for (let t = from; t < to; t++) if (t % step) ruler.append(s("line", { x1: x(t), x2: x(t), y1: RULER_H, y2: height, class: "grid" }));
  svg.append(ruler);

  y = RULER_H;
  for (const r of rows) {
    const g = s("g", { transform: `translate(0,${y})` });
    r.draw(g, x, cw);
    svg.append(g);
    y += r.height;
  }
  const cursor = s("rect", { x: 0, y: 0, width: Math.max(cw, 2), height, class: "cursor", visibility: "hidden" });
  svg.append(cursor);
  svg.addEventListener("click", (ev) => {
    const box = svg.getBoundingClientRect();
    const t = from + Math.floor((ev.clientX - box.left) / cw);
    if (t >= from && t < to) onPick(t);
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

// -- the cycle's event list --------------------------------------------------------

function eventList(detail, cycle, tasks, events) {
  const { profile, run, trace } = detail;
  const running = tasks.filter((e) => e.k === "cmd" && e.t <= cycle && cycle <= e.last);
  const by = byOwner([...running, ...tasks.filter((e) => e.k !== "cmd" && e.t === cycle), ...events], profile);
  const rows = run.cluster.components
    .filter((c) => trace.intervals[c.name] || by[c.name])
    .map((c) => {
      const cls = classAt(trace.intervals[c.name], cycle);
      return h("tr", {},
        h("th", {}, c.name),
        h("td", {}, cls ? h("span", {}, h("i", { class: `key g-${GROUP[cls] || "idle"}` }), cls) : ""),
        h("td", {}, (by[c.name] ?? []).map((e) => h("div", {}, describe(e)))));
    });
  return h("div", { class: "scroll" }, h("table", { class: "cycle-table" },
    h("thead", {}, h("tr", {}, h("th", {}, "Component"), h("th", {}, "Class"), h("th", {}, "Events in this cycle"))),
    h("tbody", {}, rows)));
}

// -- the schedule on screen --------------------------------------------------------
//
// `current` is what the last full render left on the page, so a cycle change
// can update it in place.

let current = null;

/** The shown cycle moved by d and kept inside the run, or null if no schedule is on screen (arrow keys). */
export function stepCycle(d) {
  if (!current || current.cycle === null || !current.detail.trace) return null;
  return Math.min(Math.max(current.cycle + d, 0), current.total - 1);
}

/** The first cycle any block is busy (a start lands in s, busy from s + 1), or 0. */
function firstBusy(detail) {
  let t = Infinity;
  for (const spans of Object.values(detail.tasks ?? {})) for (const sp of spans) t = Math.min(t, sp.start + 1);
  return Number.isFinite(t) ? Math.min(t, detail.run.total_cycles - 1) : 0;
}

/** Move the cursor, and fill the cluster view and the event list for `cycle`. */
async function showCycle(cycle, keepInView = true) {
  const c = current;
  if (!c.detail.trace) return; // level off: the structure only, no cycle
  c.cycle = cycle;
  if (c.cursor) {
    c.cursor.setAttribute("x", String((cycle - c.from) * c.cw));
    c.cursor.setAttribute("visibility", "visible");
    const sc = c.scroller;
    const plotW = sc.clientWidth - LABEL_W;
    const cx = (cycle - c.from) * c.cw;
    if (keepInView && plotW > 0 && (cx < sc.scrollLeft || cx + c.cw > sc.scrollLeft + plotW)) sc.scrollLeft = Math.max(cx - plotW / 2, 0);
  }
  const tr = c.detail.trace;
  const events = tr?.level === "beat" ? await c.api.events(c.detail.name, cycle, cycle + 1, { k: BEAT_KINDS }) : [];
  const names = await elementNames(c, events);
  if (current !== c || c.cycle !== cycle) return; // a newer cycle or render took over
  c.cluster.update(tr ? cycle : null, { tasks: c.tasks, fifo: c.fifo, events, names });
  c.title.textContent = `Cluster at cycle ${int(cycle)}`;
  c.prev.disabled = cycle <= 0;
  c.next.disabled = cycle >= c.total - 1;
  if (tr) c.events.replaceChildren(eventList(c.detail, cycle, c.tasks, events));
}

/**
 * When the run names its regions (D95): a function (bank, row) -> the element
 * there (`A[17]`), after loading the L1 rows the cycle's accesses touch, for
 * the cluster view's tooltips. Null without regions, or if the rows cannot be
 * loaded; the tooltips then say what they said before.
 */
async function elementNames(c, events) {
  if (!c.detail.run.regions?.length) return null;
  const rows = events.filter((e) => e.mem === "l1" && e.row !== undefined).map((e) => e.row);
  try {
    if (rows.length) await ensureRows(c.api, c.detail, "l1", rows);
  } catch {
    return null;
  }
  return (bank, row) => elementAt(c.detail, "l1", bank, row);
}

/**
 * A change of the hash's `cycle` only: update in place if the schedule of
 * `detail` is on screen and the cycle lies in its window. Returns false when
 * a full render is needed instead.
 */
export async function selectCycle(detail, st) {
  if (!current || current.detail !== detail) return false;
  const cycle = st.cycle === undefined ? current.defaultCycle : Math.min(Math.max(intArg(st.cycle, 0), 0), current.total - 1);
  if (current.cursor && (cycle < current.from || cycle >= current.to)) return false;
  await showCycle(cycle);
  return true;
}

// -- wheel zoom (D58) --------------------------------------------------------------
//
// Wheel events come in bursts; they are gathered for a short moment and
// applied as one zoom change, which keeps the cycle under the pointer where
// it was. The anchor survives the redraw in `anchorNext`. Positions are
// measured from the plot's left edge, right of the pinned labels.

let anchorNext = null; // {t, px}: put cycle t at px from the plot's left edge after the redraw
const wheel = { factor: 1, anchor: null, timer: null };

function onWheel(ev, scroller, view) {
  if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) return; // sideways: native scroll
  ev.preventDefault();
  const dy = ev.deltaY * (ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? 400 : 1);
  wheel.factor *= Math.exp(-dy * 0.002);
  if (!wheel.anchor) {
    const px = ev.clientX - scroller.getBoundingClientRect().left - scroller.clientLeft - LABEL_W;
    wheel.anchor = { t: view.from + (scroller.scrollLeft + Math.max(px, 0)) / view.cw, px: Math.max(px, 0) };
  }
  clearTimeout(wheel.timer);
  wheel.timer = setTimeout(() => {
    const next = Math.min(Math.max(view.cw * wheel.factor, view.fit), MAX_CW).toFixed(3);
    const anchor = wheel.anchor;
    wheel.factor = 1;
    wheel.anchor = null;
    if (Math.abs(Number(next) - view.cw) < 1e-3) return; // at a limit: nothing to redraw
    anchorNext = anchor;
    view.setHash({ zoom: next });
  }, 60);
}

// -- the view ----------------------------------------------------------------------

function intArg(v, dflt) {
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? n : dflt;
}

/** The cluster section: heading with cycle stepping, the cluster view, the event list. */
function clusterSection(detail, total, setHash) {
  const go = (d) => setHash({ cycle: Math.min(Math.max((current?.cycle ?? 0) + d, 0), total - 1) });
  const title = h("h2", {}, "Cluster");
  const prev = h("button", { type: "button", onclick: () => go(-1), "aria-label": "Previous cycle" }, "‹");
  const next = h("button", { type: "button", onclick: () => go(1), "aria-label": "Next cycle" }, "›");
  const events = h("div", {});
  const cluster = clusterView(detail);
  const el = h("section", { id: "cluster" },
    h("div", { class: "cycle-head" }, prev, title, next),
    cluster.el,
    detail.trace ? h("details", { class: "cycle-events", open: true }, h("summary", {}, "Events in this cycle"), events) : null);
  return { el, title, prev, next, events, cluster };
}

/**
 * Draw the schedule of `detail` and the cluster view under it into `root`.
 * ctx: {api, st (hash state), setHash}.
 */
export async function renderSchedule(root, detail, ctx) {
  const { api, st, setHash } = ctx;
  const total = detail.run.total_cycles;
  const tr = detail.trace;
  const cs = clusterSection(detail, total, setHash);
  if (!tr) {
    root.replaceChildren(h("section", {}, h("h2", {}, "Schedule"),
      h("p", { class: "note" }, "This run was traced at level off. The schedule needs at least a task trace: run the scenario again with --trace task or --trace beat.")),
    cs.el);
    current = { detail, total, from: 0, to: total, cw: 1, cursor: null, scroller: null, api, tasks: [], fifo: new Map(), defaultCycle: 0, cycle: null, ...cs };
    cs.cluster.update(null);
    cs.title.textContent = "Cluster";
    cs.prev.disabled = cs.next.disabled = true;
    return;
  }
  const beat = tr.level === "beat";
  let from = Math.min(Math.max(intArg(st.from, 0), 0), Math.max(total - 1, 0));
  let to = Math.min(Math.max(intArg(st.to, beat ? Math.min(total, from + BEAT_WINDOW) : total), from + 1), total);
  const picked = st.cycle === undefined ? null : Math.min(Math.max(intArg(st.cycle, 0), 0), total - 1);
  if (picked !== null && (picked < from || picked >= to)) { // stepped out of the window: move it, same width (D61)
    const width = to - from;
    from = Math.min(Math.max(picked - Math.floor(width / 2), 0), Math.max(total - width, 0));
    to = Math.min(from + width, total);
    const q = new URLSearchParams(location.hash.slice(1));
    q.set("from", String(from));
    q.set("to", String(to));
    history.replaceState(null, "", `#${q}`);
  }
  const win = { from, to };
  const showDetail = beat && st.detail !== "0";
  const busy = firstBusy(detail);
  const defaultCycle = busy >= from && busy < to ? busy : from;
  const cycle = picked ?? defaultCycle;

  const [tasks, beats, fifo] = await Promise.all([
    taskEvents(api, detail),
    beat ? beatEvents(api, detail, from, to) : null,
    fifoIndex(api, detail),
  ]);

  const tracers = beat ? tracersOf(st) : [];
  const journeys = await Promise.all(tracers.map(async (pick) => {
    try {
      return { pick, j: await journeyOf(api, detail, pick) };
    } catch (e) {
      return { pick, j: { available: false, reason: String(e.message || e) } };
    }
  }));
  const rows = [...tracerRows(journeys, win), ...buildRows(detail, win, tasks, beats, fifo, showDetail)];
  // Width left for the plot: the main column less its padding, the labels and the borders.
  const cst = getComputedStyle(root);
  const pad = (Number.parseFloat(cst.paddingLeft) || 0) + (Number.parseFloat(cst.paddingRight) || 0);
  const avail = Math.max((root.clientWidth || 1100) - pad - LABEL_W - 4, 200);
  const fit = Math.min(Math.max(avail / (to - from), 0.02), 32); // the window fills the plot
  const cw = st.zoom ? Math.min(Math.max(Number(st.zoom), fit), MAX_CW) : fit;

  const input = (id, value) => h("input", { id, type: "number", min: 0, max: total, value, inputmode: "numeric" });
  const fromIn = input("sched-from", from);
  const toIn = input("sched-to", to);
  const apply = () => { // a new window drops a selected cycle outside it, so it does not pull the window back
    const a = intArg(fromIn.value, from);
    const b = intArg(toIn.value, to);
    setHash({ from: a, to: b, zoom: null, cycle: picked !== null && picked >= a && picked < b ? picked : null });
  };
  const zoom = (f) => setHash({ zoom: Math.min(Math.max(cw * f, fit), MAX_CW).toFixed(3) });

  const filtered = tr.filter_sources || tr.filter_window;
  const note = beat
    ? `Cycles ${int(from)} to ${int(to - 1)} of ${int(total)}. Idle is blank. Under each row, a line runs from each task's start to its done. Click a cycle, or step with the arrow keys, to show it in the cluster view below.` +
      (filtered ? " This beat trace is filtered (see the header), so some detail rows are empty." : "")
    : `Cycles ${int(from)} to ${int(to - 1)} of ${int(total)}. Task-level trace: class runs, tasks and commands; run with --trace beat for requests, read data, FIFO counts and firings.`;

  const tracerNote = !beat ? null : h("p", { class: "note" }, tracers.length
    ? "Tracer rows: L1 hops in the upper half, L2 hops in the lower half; R a read, W a write, a green strip read data back, ✕ held back, a diamond a firing. Click a label to open the element in the Memory tab."
    : "To follow one element through the run, pick a word in the Memory tab and trace it in the schedule.");
  const legend = h("ul", { class: "legend" },
    [["busy", "Busy, firing, read data back (strip)"], ["req", "Request (accepted)"], ["mem", "Memory stall"], ["flow", "Flow stall"], ["command", "Controller command"], ["wait", "Controller wait"], ["fifo", "FIFO count (largest lane)"]]
      .map(([g, label]) => h("li", {}, h("i", { class: `key g-${g}` }), label)));

  const controls = h("form", { class: "sched-controls", onsubmit: (e) => { e.preventDefault(); apply(); } },
    h("label", {}, "From ", fromIn), h("label", {}, "to ", toIn),
    h("button", { type: "submit" }, "Show"),
    h("button", { type: "button", onclick: () => setHash({ from: 0, to: total, zoom: null }) }, "Whole run"),
    h("button", { type: "button", onclick: () => zoom(1 / 1.5), "aria-label": "Zoom out" }, "−"),
    h("button", { type: "button", onclick: () => zoom(1.5), "aria-label": "Zoom in" }, "+"),
    h("button", { type: "button", onclick: () => setHash({ zoom: null }) }, "Fit"),
    h("label", {}, h("input", { type: "checkbox", checked: showDetail, disabled: !beat,
      onchange: (e) => setHash({ detail: e.target.checked ? null : 0 }) }), " Beat detail"),
    h("span", { class: "zoom-read" }, `${dec(cw, cw < 1 ? 2 : 1)} px per cycle`));

  const keepLeft = root.querySelector(".sched-scroll")?.scrollLeft ?? 0; // survive a redraw
  const keepTop = root.querySelector(".sched-scroll")?.scrollTop ?? 0;
  const { svg, cursor } = chart(rows, win, cw, (t) => setHash({ cycle: t }));
  const scroller = h("div", { class: "sched sched-scroll" }, labels(rows, setHash, tracers), svg);
  root.replaceChildren(
    h("section", { id: "schedule" },
      h("h2", {}, "Schedule"), h("p", { class: "note" }, note), tracerNote, legend, controls, scroller),
    cs.el);
  scroller.addEventListener("wheel", (ev) => onWheel(ev, scroller, { from, cw, fit, setHash }), { passive: false });
  scroller.scrollTop = keepTop;
  const zoomed = !!anchorNext;
  if (anchorNext) { // a wheel zoom: keep the cycle under the pointer in place
    scroller.scrollLeft = Math.max((anchorNext.t - from) * cw - anchorNext.px, 0);
    anchorNext = null;
  } else {
    scroller.scrollLeft = keepLeft;
  }
  current = { detail, total, from, to, cw, cursor, scroller, api, tasks, fifo, defaultCycle, cycle: null, ...cs };
  await showCycle(cycle, !zoomed);
}
