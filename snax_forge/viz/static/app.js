// Entry point of the viewer (D55, D57): loads the
// run list and the selected run from the API, fills the header and hands the
// page to one view: the profile report (report.js), the schedule with the
// cluster view under it (schedule.js, cluster.js), or the memory layout
// (memory.js).
//
// All view state lives in the URL hash, e.g.
//   #run=vecadd&view=schedule&open=142&cycle=42
// so a browser refresh or a shared link shows the same thing. The selected
// cycle is what the cluster view shows; a change of the cycle alone updates
// the page in place, anything else redraws the view. On the schedule the
// arrow keys step to the next or previous event cycle, shift + arrow to the
// next or previous task boundary (the server's /step); on the memory tab they
// step one cycle. Space plays and pauses (player.js), stepping as the
// right arrow does.
// A run's detail is fetched once and kept until Reload, which asks the server
// to read the directories again (POST /api/reload) and clears the cache.

import { h, int } from "./dom.js";
import { renderMemory, stepMemoryCycle, updateMemory } from "./memory.js";
import { setStepper, stopPlaying, togglePlaying } from "./player.js";
import { renderReport } from "./report.js";
import { renderSchedule, selectCycle, stepSchedule } from "./schedule.js";

const $ = (id) => document.getElementById(id);

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || `${url}: HTTP ${r.status}`);
  return body;
}

const enc = encodeURIComponent;
const api = {
  runs: () => getJSON("/api/runs"),
  run: (name) => getJSON(`/api/run/${enc(name)}`),
  /** The lanes and segments; the folds starting at a cycle of `open` or holding `cycle` opened. */
  schedule: (name, open, cycle) => {
    const q = new URLSearchParams();
    if (open.length) q.set("open", open.join(","));
    if (cycle !== null && cycle !== undefined) q.set("cycle", String(cycle));
    return getJSON(`/api/run/${enc(name)}/schedule${q.size ? `?${q}` : ""}`);
  },
  /** The beat and stall lines of from <= t < to, per lane. */
  scheduleDetail: (name, from, to) => getJSON(`/api/run/${enc(name)}/schedule/detail?${new URLSearchParams({ from: String(from), to: String(to) })}`),
  /** The data side of the cluster at cycle t. */
  cluster: (name, t) => getJSON(`/api/run/${enc(name)}/cluster?t=${enc(String(t))}`),
  /** The next (dir 1) or previous (-1) cycle with an event (by "event") or a task boundary (by "boundary"). */
  step: (name, t, dir, by) => getJSON(`/api/run/${enc(name)}/step?${new URLSearchParams({ t: String(t), dir: String(dir), by })}`),
  /** The memory view; `marks` (conflicts, arrival, use, wait) folds by those marks (D57). */
  memory: (name, marks) => getJSON(`/api/run/${enc(name)}/memory${marks ? `?marks=${enc(marks)}` : ""}`),
  /** Rows from <= r < to of memory `mem` in full (at most 256). */
  memoryRows: (name, mem, from, to, marks) => {
    const q = new URLSearchParams({ from: String(from), to: String(to) });
    if (marks) q.set("marks", marks);
    return getJSON(`/api/run/${enc(name)}/memory/${enc(mem)}/rows?${q}`);
  },
  /** The cycle overlay of cycle t: its accesses (events) and the reads whose data comes back in it (returns). */
  memoryCycle: (name, t) => getJSON(`/api/run/${enc(name)}/memory/cycle?t=${enc(String(t))}`),
  journey: (name, region, index) => getJSON(`/api/run/${enc(name)}/journey?${new URLSearchParams({ region, index: String(index) })}`),
  conflicts: (name, from, to) => getJSON(`/api/run/${enc(name)}/conflicts?${new URLSearchParams({ from: String(from), to: String(to) })}`),
  reload: () => getJSON("/api/reload", { method: "POST" }),
};

// -- view state in the hash -----------------------------------------------------

function hashState() {
  return Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
}

/** Merge `changes` into the hash (null removes a key); triggers hashchange. */
function setHash(changes) {
  const q = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of Object.entries(changes)) {
    if (v === null || v === undefined) q.delete(k);
    else q.set(k, String(v));
  }
  location.hash = q.toString();
}

const VIEWS = { report: "Report", schedule: "Schedule and cluster", memory: "Memory" };

// -- header ------------------------------------------------------------------------

function status(msg, err = false) {
  const s = $("status");
  s.textContent = msg || "";
  s.classList.toggle("error", err);
}

function fillPicker(runs, name) {
  const sel = $("run-select");
  sel.replaceChildren(...runs.map((r) =>
    h("option", { value: r.name, selected: r.name === name }, `${r.name} (${int(r.total_cycles)} cycles, trace ${r.trace_level})`)));
  $("run-pick").hidden = runs.length < 2;
}

function fillTabs(view) {
  $("tabs").replaceChildren(...Object.entries(VIEWS).map(([key, label]) =>
    h("button", { type: "button", role: "tab", "aria-selected": String(key === view),
      onclick: () => setHash({ view: key }) }, label)));
}

function traceText(tr, level) {
  if (!tr) return level;
  return `${tr.level}; ${int(tr.n_events)} events in the window (boot ${int(tr.boot)} cycles)`;
}

function fillHeader(detail) {
  const run = detail.run;
  const check = detail.profile.functional_check;
  $("run-title").textContent = detail.name;
  document.title = `${detail.name}: SNAX-FORGE run viewer`;
  const facts = [
    ["Design point", `${detail.cluster.design} (${run.design_point})`],
    ["Cluster file", (run.hjson || [])[0] || "–"],
    ["Directory", detail.path],
    ["Trace", traceText(detail.trace, run.trace_level)],
    ["Total cycles", `${int(detail.total_cycles)} (main to forge_check)`],
    ["Check", check ? (check.passed ? "passed" : "FAILED") : "–"],
  ];
  $("facts").replaceChildren(...facts.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

// -- showing a run ----------------------------------------------------------------

let cache = {}; // run name -> its detail; cleared by Reload
let shown = null; // "run/view" last drawn, so a cycle change does not refetch
let drawnWith = null; // hash of the last full draw without `cycle`

/** The hash state without `cycle`, as a string to compare draws. */
function withoutCycle(st) {
  const { cycle, ...rest } = st;
  return new URLSearchParams(rest).toString();
}

async function show() {
  const st = hashState();
  try {
    const runs = await api.runs();
    const name = runs.some((r) => r.name === st.run) ? st.run : runs[0].name;
    const view = st.view in VIEWS ? st.view : "report";
    fillPicker(runs, name);
    fillTabs(view);
    if (!cache[name]) {
      status("Loading…");
      cache[name] = await api.run(name);
    }
    const detail = cache[name];
    if (shown !== null && shown !== `${name}/${view}`) stopPlaying(); // another run or view
    fillHeader(detail);
    const root = $("report");
    if (view !== "memory") document.body.classList.remove("drawer-open"); // only the memory tab has the drawer
    if (view === "report") {
      if (shown !== `${name}/report`) renderReport(root, detail);
    } else if (view === "memory") {
      const same = shown === `${name}/memory` && (await updateMemory(detail, st));
      if (!same) await renderMemory(root, detail, { api, status, st, setHash, hashState });
    } else {
      const same = shown === `${name}/schedule` && drawnWith === withoutCycle(st);
      if (!(same && (await selectCycle(detail, st)))) {
        await renderSchedule(root, detail, { api, st, setHash });
        drawnWith = withoutCycle(hashState());
      }
    }
    shown = `${name}/${view}`;
    status("");
  } catch (e) {
    status(String(e.message || e), true);
  }
}

// One draw at a time: a hash change during a draw waits for it, and several
// waiting changes collapse into one draw of the latest hash, so a held arrow
// key or a slow first load never leaves an older draw on screen.
let drawing = null;
let pending = false;
async function showLatest() {
  if (drawing) {
    pending = true;
    return drawing;
  }
  do {
    pending = false;
    drawing = show();
    await drawing;
  } while (pending);
  drawing = null;
}

$("run-select").addEventListener("change", (e) => setHash({ run: e.target.value, open: null, zoom: null, cycle: null, pick: null }));
window.addEventListener("hashchange", showLatest);
$("reload").addEventListener("click", async () => {
  status("Reloading…");
  try {
    stopPlaying();
    await api.reload();
    cache = {};
    shown = null;
    drawnWith = null;
    await showLatest();
  } catch (e) {
    status(String(e.message || e), true);
  }
});

/** The shown view's next cycle (d 1) or previous one (-1), by event or boundary on the schedule; null when it has none to step to. */
async function stepShown(d, by = "event") {
  const view = hashState().view ?? "report";
  if (view === "schedule") return stepSchedule(d, by);
  return view === "memory" ? stepMemoryCycle(d) : null;
}

// The player steps as the right arrow does and stops where it cannot step; a
// tick waits while a draw runs, so the shown cycle is the one it steps from.
setStepper({
  next: async () => {
    const n = await stepShown(1);
    const now = Number.parseInt(hashState().cycle, 10);
    return n === null || n === now ? null : n;
  },
  go: (n) => setHash({ cycle: n }),
  busy: () => drawing !== null,
});

// The page is as wide as the window allows. The schedule fits its window
// to the width it is drawn at, so a change of width draws it again; the other
// views follow the width by themselves (CSS grids).
let drawnAtWidth = window.innerWidth;
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (window.innerWidth === drawnAtWidth) return;
    drawnAtWidth = window.innerWidth;
    if ((hashState().view ?? "report") !== "schedule") return;
    shown = null;
    drawnWith = null;
    showLatest();
  }, 200);
});

// Left and right arrows step the selected cycle while the schedule or the memory
// tab is shown (shift on the schedule: by task boundary); Space plays and
// pauses (a focused button keeps Space for itself).
document.addEventListener("keydown", async (e) => {
  const view = hashState().view ?? "report";
  if (view !== "schedule" && view !== "memory") return;
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (e.key === " ") {
    if (e.target instanceof Element && e.target.closest("button, summary, a")) return;
    e.preventDefault();
    togglePlaying();
    return;
  }
  const d = e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0;
  if (!d) return;
  e.preventDefault();
  const next = await stepShown(d, e.shiftKey ? "boundary" : "event");
  if (next !== null) setHash({ cycle: next });
});

showLatest();
