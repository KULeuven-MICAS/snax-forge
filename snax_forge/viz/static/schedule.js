// The schedule of one run (VIS2, D57), HLS-schedule style: one row per
// component in registration order, one column per cycle, over a cycle window
// [from, to). Built from what the run already wrote (D39, D40):
//
//   main row      the component's class runs (trace intervals); idle is left
//                 blank, so the rows read like a Gantt chart. Under it a thin
//                 line per task, from its start to its done. The controller's
//                 row shows each command (cmd events) with its register.
//                 A DMA's tasks carry their direction (L2 → L1, L1 → L2, D58).
//   detail rows   beat level only, for components that ran a task: one per
//                 xbar port the component owns, labelled `<port>.target_banks`
//                 (`<owner>.target_banks` for a single port, D58): grants and
//                 stalls, with bank numbers when there is room; its FIFO
//                 (largest lane count), its firings or its DMA beats.
//
// The mouse wheel over the chart zooms around the pointer; shift + wheel and
// a sideways swipe scroll (D58).
//
// Clicking a cycle selects it (hash `cycle`); the panel under the chart lists
// every component's class and every event in that cycle. The selected cycle
// is what the cluster view (VIS3) will follow.
//
// Data: the class intervals come with the run detail; task events (cmd,
// start, done) are fetched once per run for the whole run; beat events only
// for the window, plus the FIFO counts before it (a fifo event is a state
// change, D39). All through /api/run/<name>/events with its k filter (D57).

import { dec, h, int } from "./dom.js";

const SVG = "http://www.w3.org/2000/svg";
const TASK_KINDS = ["cmd", "start", "done"];
const BEAT_KINDS = ["grant", "stall", "fire", "dma_beat", "poll", "fifo"];
const MAIN_H = 24; // px, main row
const SUB_H = 17; // px, detail row
const RULER_H = 22;
const LABEL_W = 150;
const BEAT_WINDOW = 400; // default window of a beat-level run
const MAX_CW = 64; // px per cycle, largest zoom
const DIR_TEXT = { l2_to_l1: "L2 → L1", l1_to_l2: "L1 → L2" };

// Colour group of each cycle class, as in the report (report.js).
const GROUP = {
  busy: "busy", stall_xbar: "mem", stall_l1: "mem",
  stall_fifo: "flow", stall_in: "flow", stall_out: "flow", stall_mem: "flow",
  command: "command", wait: "wait",
};

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

const tasksOf = new WeakMap(); // detail -> task events of the whole run
const beatsOf = new Map(); // "name|from|to" -> {events, fifoStart}

async function taskEvents(api, detail) {
  if (!tasksOf.has(detail)) {
    tasksOf.set(detail, await api.events(detail.name, 0, detail.run.total_cycles + 1, { k: TASK_KINDS }));
  }
  return tasksOf.get(detail);
}

/** Beat events in [from, to) and each FIFO lane's count at `from`. */
async function beatEvents(api, detail, from, to) {
  const key = `${detail.name}|${detail.path}|${from}|${to}`;
  if (!beatsOf.has(key)) {
    const [events, before] = await Promise.all([
      api.events(detail.name, from, to, { k: BEAT_KINDS }),
      from > 0 ? api.events(detail.name, 0, from, { k: ["fifo"] }) : [],
    ]);
    const fifoStart = {}; // "src|lane" -> count
    for (const e of before) fifoStart[`${e.src}|${e.lane}`] = e.count;
    beatsOf.set(key, { events, fifoStart });
  }
  return beatsOf.get(key);
}

/** Pair each start with the next done of the same block: [{src, start, done}]. */
function taskSpans(tasks, total) {
  const open = {};
  const spans = [];
  for (const e of tasks) {
    if (e.k === "start") open[e.src] = e.t;
    else if (e.k === "done" && e.src in open) {
      spans.push({ src: e.src, start: open[e.src], done: e.t });
      delete open[e.src];
    }
  }
  for (const [src, start] of Object.entries(open)) spans.push({ src, start, done: total });
  return spans;
}

/** Who an event belongs to: grants and stalls to the port's owner, a FIFO to its streamer. */
function ownerOf(e, profile, fifoOwner) {
  if (e.k === "grant" || e.k === "stall") return profile.ports[e.port]?.owner ?? e.src;
  if (e.k === "fifo") return fifoOwner[e.src] ?? e.src;
  return e.src;
}

/** One line of text for an event, as the cycle panel lists it. */
export function describe(e) {
  switch (e.k) {
    case "cmd": {
      const what = e.op === "wait" ? `wait ${e.block} (${e.mode})` : `${e.op} ${e.reg ?? ""}${e.value !== undefined ? ` = ${e.value}` : ""}`;
      return `pc ${e.pc}: ${what}, cycles ${e.t}–${e.last}`;
    }
    case "start": return "start landed (busy from the next cycle)";
    case "done": return "done (busy reads 0 from here)";
    case "grant":
    case "stall": {
      const banks = e.banks.length > 1 ? `banks ${e.banks[0]}–${e.banks[e.banks.length - 1]}` : `bank ${e.banks[0]}`;
      const head = e.k === "grant" ? "granted" : e.wider ? "stalled by a wider grant" : "stalled";
      return `${e.port} ${head}: ${e.w ? "write" : "read"} ${e.mem} addr ${e.addr}, ${banks}, row ${e.row}`;
    }
    case "fire": return `firing ${e.n}`;
    case "dma_beat": return `${e.side === "src" ? "read" : "write"} beat ${e.i}, ${e.mem} addr ${e.addr}`;
    case "poll": return `poll ${e.block}: busy = ${e.value}`;
    case "fifo": return `${e.src} lane ${e.lane} holds ${e.count} from here`;
    default: return JSON.stringify(e);
  }
}

// -- rows -------------------------------------------------------------------------

/** The rows to draw, in order: [{name, label, sub, height, draw(g, x, cw)}]. */
function buildRows(detail, win, tasks, beats, showDetail) {
  const { profile, run, trace } = detail;
  const { from, to } = win;
  const clip = (a, b) => [Math.max(a, from), Math.min(b, to)];
  const intervals = trace.intervals;
  const spans = taskSpans(tasks, run.total_cycles);
  const cmds = tasks.filter((e) => e.k === "cmd");
  const fifoOwner = Object.fromEntries(Object.entries(profile.streamers).map(([n, st]) => [st.fifo.name, n]));
  const beatBy = {}; // owner -> events
  for (const e of beats?.events ?? []) (beatBy[ownerOf(e, profile, fifoOwner)] ??= []).push(e);

  const rows = [];
  for (const spec of run.cluster.components) {
    const name = spec.name;
    const runs = intervals[name];
    if (!runs) continue; // the xbar has no classes; its work shows on the port rows
    const isCtl = spec.kind === "controller";
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
        for (const sp of spans) {
          if (sp.src !== name) continue;
          const [a, b] = clip(sp.start, sp.done);
          if (b < a) continue;
          g.append(s("line", { x1: x(a), x2: x(Math.max(b, a)), y1: MAIN_H - 3, y2: MAIN_H - 3, class: "task" },
            `${name} task: start in ${sp.start}, done in ${sp.done}`));
          if (sp.start >= from && sp.start < to) g.append(s("path", { d: `M${x(sp.start)},${MAIN_H - 7} l4,4 l-4,4 z`, class: "task-mark" }, `${name} start in ${sp.start}`));
          if (sp.done >= from && sp.done <= to) g.append(s("rect", { x: x(sp.done) - 1, y: MAIN_H - 8, width: 2, height: 8, class: "task-mark" }, `${name} done in ${sp.done}`));
        }
        for (const tk of detail.dma_tasks?.[name] ?? []) { // D58
          const [a, b] = clip(tk.start, tk.done);
          const text = DIR_TEXT[tk.direction] ?? tk.direction;
          if ((b - a) * cw >= text.length * 6 + 8) txt(g, { x: x(a) + 4, y: MAIN_H - 10, class: "cell-text on-dark" }, text);
        }
      },
    });
    // Detail rows only for a component that ran a task: an unused block adds nothing.
    if (!showDetail || !spans.some((sp) => sp.src === name)) continue;
    const mine = beatBy[name] ?? [];

    // One row per xbar port the component owns, named after what it shows (D58).
    const ports = Object.entries(profile.ports).filter(([, p]) => p.owner === name);
    for (const [port, p] of ports) {
      const evs = mine.filter((e) => (e.k === "grant" || e.k === "stall") && e.port === port);
      rows.push({
        name, label: `${ports.length === 1 ? name : port}.target_banks`, sub: `${p.width} bits`,
        title: `port ${port}, ${p.width} bits: the bank of each grant and stall`, height: SUB_H, detail: true,
        draw(g, x, cw) {
          for (const e of evs) {
            const cls = e.k === "grant" ? "s-busy" : e.wider ? "s-mem wider" : "s-mem";
            g.append(s("rect", { x: x(e.t) + 0.5, y: 2, width: Math.max(cw - 1, 0.5), height: SUB_H - 4, class: cls }, `cycle ${e.t}: ${describe(e)}`));
            const text = e.banks.length > 1 ? `${e.banks[0]}+` : String(e.banks[0]);
            if (cw >= text.length * 6 + 4) txt(g, { x: x(e.t) + cw / 2, y: SUB_H - 5, class: "cell-text mid on-dark" }, text);
          }
        },
      });
    }

    const fifoName = profile.streamers[name]?.fifo.name;
    if (fifoName) {
      const depth = profile.streamers[name].fifo.depth;
      const lanes = profile.streamers[name].fifo.hist.length;
      const counts = Array.from({ length: lanes }, (_, l) => beats.fifoStart[`${fifoName}|${l}`] ?? 0);
      const changes = mine.filter((e) => e.k === "fifo");
      rows.push({
        name, label: fifoName, sub: `depth ${depth}, largest lane`, height: SUB_H, detail: true,
        draw(g, x, cw) {
          // Step through the window: the count of each lane holds until its next fifo event.
          let i = 0;
          const cur = [...counts];
          for (let t = from; t < to; t++) {
            while (i < changes.length && changes[i].t <= t) { cur[changes[i].lane] = changes[i].count; i++; }
            const top = Math.max(...cur);
            if (!top) continue;
            const hgt = ((SUB_H - 4) * top) / depth;
            g.append(s("rect", { x: x(t), y: SUB_H - 2 - hgt, width: cw, height: hgt, class: "s-fifo" },
              `cycle ${t}: ${fifoName} lanes hold ${cur.join(", ")} (depth ${depth})`));
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
        rows.push({
          name, label: `${name} ${side === "src" ? "reads" : "writes"}`, sub: "beats", height: SUB_H, detail: true,
          draw(g, x, cw) {
            for (const e of evs) {
              g.append(s("rect", { x: x(e.t) + 0.5, y: 2, width: Math.max(cw - 1, 0.5), height: SUB_H - 4, class: "s-busy" }, `cycle ${e.t}: ${describe(e)}`));
              if (cw >= String(e.i).length * 6 + 4) txt(g, { x: x(e.t) + cw / 2, y: SUB_H - 5, class: "cell-text mid on-dark" }, String(e.i));
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

// -- drawing -----------------------------------------------------------------------

/** Tick step so that labels are at least 44 px apart. */
function tickStep(cw) {
  for (const m of [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]) if (m * cw >= 44) return m;
  return 100000;
}

function chart(rows, win, cw, cycle, onPick) {
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
  if (cycle !== null && cycle >= from && cycle < to) {
    svg.append(s("rect", { x: x(cycle), y: 0, width: Math.max(cw, 2), height, class: "cursor" }));
  }
  svg.addEventListener("click", (ev) => {
    const box = svg.getBoundingClientRect();
    const t = from + Math.floor((ev.clientX - box.left) / cw);
    if (t >= from && t < to) onPick(t);
  });
  return svg;
}

function labels(rows) {
  return h("div", { class: "sched-labels", style: { width: `${LABEL_W}px` } },
    h("div", { style: { height: `${RULER_H}px` }, class: "ruler-label" }, "cycle"),
    rows.map((r) => h("div", { class: `lab${r.detail ? " sub" : ""}`, style: { height: `${r.height}px` }, title: r.title ?? r.label },
      h("span", {}, r.label), r.sub ? h("small", {}, r.sub) : null)));
}

// -- the cycle panel ---------------------------------------------------------------

function classAt(runs, t) {
  for (const [cls, a, b] of runs) if (a <= t && t < b) return cls;
  return null;
}

function cyclePanel(detail, cycle, tasks, events, setHash) {
  const { profile, run, trace } = detail;
  const total = run.total_cycles;
  const fifoOwner = Object.fromEntries(Object.entries(profile.streamers).map(([n, st]) => [st.fifo.name, n]));
  const running = tasks.filter((e) => e.k === "cmd" && e.t <= cycle && cycle <= e.last);
  const at = [...running, ...tasks.filter((e) => e.k !== "cmd" && e.t === cycle), ...events];
  const by = {};
  for (const e of at) (by[ownerOf(e, profile, fifoOwner)] ??= []).push(e);
  const rows = run.cluster.components
    .filter((c) => trace.intervals[c.name] || by[c.name])
    .map((c) => {
      const cls = trace.intervals[c.name] ? classAt(trace.intervals[c.name], cycle) : null;
      return h("tr", {},
        h("th", {}, c.name),
        h("td", {}, cls ? h("span", {}, h("i", { class: `key g-${GROUP[cls] || "idle"}` }), cls) : ""),
        h("td", {}, (by[c.name] ?? []).map((e) => h("div", {}, describe(e)))));
    });
  const go = (t) => setHash({ cycle: Math.min(Math.max(t, 0), total - 1) });
  return h("div", { class: "cycle-panel" },
    h("div", { class: "cycle-head" },
      h("button", { type: "button", onclick: () => go(cycle - 1), disabled: cycle <= 0, "aria-label": "Previous cycle" }, "‹"),
      h("h3", {}, `Cycle ${int(cycle)}`),
      h("button", { type: "button", onclick: () => go(cycle + 1), disabled: cycle >= total - 1, "aria-label": "Next cycle" }, "›"),
      h("button", { type: "button", onclick: () => setHash({ cycle: null }) }, "Clear")),
    trace.level === "beat" ? null : h("p", { class: "note" }, "Task-level trace: commands, starts and dones only."),
    h("div", { class: "scroll" }, h("table", { class: "cycle-table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Component"), h("th", {}, "Class"), h("th", {}, "Events in this cycle"))),
      h("tbody", {}, rows))));
}

// Left and right arrows step the selected cycle while the schedule is shown.
let keysFor = null; // {total, setHash} of the schedule on screen
document.addEventListener("keydown", (e) => {
  const st = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
  if (!keysFor || st.view !== "schedule" || st.cycle === undefined) return;
  if (e.target instanceof HTMLInputElement) return;
  const d = e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0;
  if (!d) return;
  e.preventDefault();
  keysFor.setHash({ cycle: Math.min(Math.max(Number(st.cycle) + d, 0), keysFor.total - 1) });
});

// -- wheel zoom (D58) --------------------------------------------------------------
//
// Wheel events come in bursts; they are gathered for a short moment and
// applied as one zoom change, which keeps the cycle under the pointer where
// it was. The anchor survives the redraw in `anchorNext`.

let anchorNext = null; // {t, px}: put cycle t at px from the plot's left edge after the redraw
const wheel = { factor: 1, anchor: null, timer: null };

function onWheel(ev, scroller, view) {
  if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) return; // sideways: native scroll
  ev.preventDefault();
  const dy = ev.deltaY * (ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? 400 : 1);
  wheel.factor *= Math.exp(-dy * 0.002);
  if (!wheel.anchor) {
    const px = ev.clientX - scroller.getBoundingClientRect().left;
    wheel.anchor = { t: view.from + (scroller.scrollLeft + px) / view.cw, px };
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

/**
 * Draw the schedule of `detail` into `root`.
 * ctx: {api, st (hash state), setHash}.
 */
export async function renderSchedule(root, detail, ctx) {
  const { api, st, setHash } = ctx;
  const total = detail.run.total_cycles;
  const tr = detail.trace;
  if (!tr) {
    root.replaceChildren(h("section", {}, h("h2", {}, "Schedule"),
      h("p", { class: "note" }, "This run was traced at level off. The schedule needs at least a task trace: run the scenario again with --trace task or --trace beat.")));
    return;
  }
  keysFor = { total, setHash };
  const beat = tr.level === "beat";
  const from = Math.min(Math.max(intArg(st.from, 0), 0), Math.max(total - 1, 0));
  const to = Math.min(Math.max(intArg(st.to, beat ? Math.min(total, from + BEAT_WINDOW) : total), from + 1), total);
  const win = { from, to };
  const showDetail = beat && st.detail !== "0";
  const cycle = st.cycle === undefined ? null : Math.min(Math.max(intArg(st.cycle, 0), 0), total - 1);

  const [tasks, beats] = await Promise.all([
    taskEvents(api, detail),
    beat ? beatEvents(api, detail, from, to) : null,
  ]);
  const atCycle = beat && cycle !== null ? await api.events(detail.name, cycle, cycle + 1, { k: BEAT_KINDS }) : [];

  const rows = buildRows(detail, win, tasks, beats, showDetail);
  // Width left for the plot: the main column less its padding, the labels and the borders.
  const cs = getComputedStyle(root);
  const pad = (Number.parseFloat(cs.paddingLeft) || 0) + (Number.parseFloat(cs.paddingRight) || 0);
  const avail = Math.max((root.clientWidth || 1100) - pad - LABEL_W - 4, 200);
  const fit = Math.min(Math.max(avail / (to - from), 0.02), 32); // the window fills the plot
  const cw = st.zoom ? Math.min(Math.max(Number(st.zoom), fit), MAX_CW) : fit;

  const input = (id, value) => h("input", { id, type: "number", min: 0, max: total, value, inputmode: "numeric" });
  const fromIn = input("sched-from", from);
  const toIn = input("sched-to", to);
  const apply = () => setHash({ from: intArg(fromIn.value, from), to: intArg(toIn.value, to), zoom: null });
  const zoom = (f) => setHash({ zoom: Math.min(Math.max(cw * f, fit), MAX_CW).toFixed(3) });

  const filtered = tr.filter_sources || tr.filter_window;
  const note = beat
    ? `Cycles ${int(from)} to ${int(to - 1)} of ${int(total)}. Idle is blank. Under each row, a line runs from each task's start to its done. Click a cycle to list what happened in it.` +
      (filtered ? " This beat trace is filtered (see the header), so some detail rows are empty." : "")
    : `Cycles ${int(from)} to ${int(to - 1)} of ${int(total)}. Task-level trace: class runs, tasks and commands; run with --trace beat for grants, FIFO counts and firings.`;

  const legend = h("ul", { class: "legend" },
    [["busy", "Busy, grant, firing, beat"], ["mem", "Memory stall"], ["flow", "Flow stall"], ["command", "Controller command"], ["wait", "Controller wait"], ["fifo", "FIFO count (largest lane)"]]
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

  const keepLeft = root.querySelector(".sched-plot")?.scrollLeft ?? 0; // survive a redraw
  const scroller = h("div", { class: "sched-plot" }, chart(rows, win, cw, cycle, (t) => setHash({ cycle: t })));
  root.replaceChildren(h("section", { id: "schedule" },
    h("h2", {}, "Schedule"), h("p", { class: "note" }, note), legend, controls,
    h("div", { class: "sched" }, labels(rows), scroller),
    cycle !== null ? cyclePanel(detail, cycle, tasks, atCycle, setHash)
      : h("p", { class: "note" }, "Select a cycle by clicking in the chart.")));
  scroller.addEventListener("wheel", (ev) => onWheel(ev, scroller, { from, cw, fit, setHash }), { passive: false });
  if (anchorNext) { // a wheel zoom: keep the cycle under the pointer in place
    scroller.scrollLeft = Math.max((anchorNext.t - from) * cw - anchorNext.px, 0);
    anchorNext = null;
    return;
  }
  scroller.scrollLeft = keepLeft;
  if (cycle !== null) { // keep the selected cycle in view
    const cx = (cycle - from) * cw;
    if (cx < scroller.scrollLeft || cx > scroller.scrollLeft + scroller.clientWidth) scroller.scrollLeft = Math.max(cx - scroller.clientWidth / 2, 0);
  }
}
