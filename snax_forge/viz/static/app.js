// Entry point of the viewer (VIS1, VIS2, D55, D57): loads the run list and
// the selected run from the API, fills the header and hands the page to one
// view: the profile report (report.js) or the schedule (schedule.js).
//
// All view state lives in the URL hash, e.g.
//   #run=vecadd&view=schedule&from=0&to=120&cycle=42
// so a browser refresh or a shared link shows the same thing, and the
// schedule's selected cycle is what the cluster view (VIS3) will follow.
// A run's detail is fetched once and kept until Reload, which asks the server
// to read the directories again (POST /api/reload) and clears the cache.

import { h, int } from "./dom.js";
import { renderReport } from "./report.js";
import { renderSchedule } from "./schedule.js";

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
  fifo: (name) => getJSON(`/api/run/${enc(name)}/fifo`),
  /** Events with from <= t < to; opts.src / opts.k are lists of sources / kinds. */
  events: (name, from, to, opts = {}) => {
    const q = new URLSearchParams({ from: String(from), to: String(to) });
    if (opts.src?.length) q.set("src", opts.src.join(","));
    if (opts.k?.length) q.set("k", opts.k.join(","));
    return getJSON(`/api/run/${enc(name)}/events?${q}`).then((r) => r.events);
  },
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

const VIEWS = { report: "Report", schedule: "Schedule" };

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
  const parts = [tr.level];
  if (tr.filter_sources) parts.push(`beat events of ${tr.filter_sources.join(", ")} only`);
  if (tr.filter_window) parts.push(`beat events in cycles [${tr.filter_window[0]}, ${tr.filter_window[1]}) only`);
  return parts.join("; ");
}

function fillHeader(detail) {
  const run = detail.run;
  $("run-title").textContent = detail.name;
  document.title = `${detail.name}: SNAX-FORGE run viewer`;
  const facts = [
    ["Scenario", run.scenario],
    ["Cluster file", run.cluster_file || "inline"],
    ["Directory", detail.path],
    ["Trace", traceText(detail.trace, run.trace_level)],
    ["Total cycles", int(run.total_cycles)],
  ];
  $("facts").replaceChildren(...facts.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

// -- showing a run ----------------------------------------------------------------

let cache = {}; // run name -> {detail, fifo}; cleared by Reload
let shown = null; // "run/view" last drawn, so a cycle change does not refetch

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
      const [detail, fifo] = await Promise.all([api.run(name), api.fifo(name)]);
      cache[name] = { detail, fifo };
    }
    const { detail, fifo } = cache[name];
    fillHeader(detail);
    const root = $("report");
    if (view === "report") {
      if (shown !== `${name}/report`) renderReport(root, detail, fifo);
    } else {
      await renderSchedule(root, detail, { api, st, setHash });
    }
    shown = `${name}/${view}`;
    status("");
  } catch (e) {
    status(String(e.message || e), true);
  }
}

$("run-select").addEventListener("change", (e) => setHash({ run: e.target.value, from: null, to: null, cycle: null }));
window.addEventListener("hashchange", show);
$("reload").addEventListener("click", async () => {
  status("Reloading…");
  try {
    await api.reload();
    cache = {};
    shown = null;
    await show();
  } catch (e) {
    status(String(e.message || e), true);
  }
});

show();
