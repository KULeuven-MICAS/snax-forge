// DFG viewer: one step of a recipe at a time, two with Compare.
//
// The server works out everything shown (viz/dfg/api.py): the step's drawing (its boxes, ports,
// edges and outlines, drawing.py), the Details of every item and the Estimate's figures. graph.js
// lays the drawing out with ELK and draws it; nothing here is computed but where to draw. The
// old step stays on screen until the new one is laid out; then its boxes glide from where they
// were (glide), unless the browser asks for reduced motion.
//
// The URL hash holds the state, as in the run viewer: #step=<k>&sel=<id>&cmp=<j>&t=<transform>&tech=<tech>&open=<blocks>&fold=<loops>&focus=1&zoom=<s,x,y>&zoomc=<s,x,y>
// (step and cmp, view mode: the steps' names, dot4_w8/6_link from several folders, and an old
// hash's bare stem or number still opens its step, /api/find; edit mode: numbers; t, edit mode: that transform's form open for the selection; tech, the technology the cost
// estimate is on, one of /api/mode's techs, passed to every step and compare request; open, the blocks drawn
// as their nodes, comma-separated, passed as ?expand=). A selection is a name:
// everything of that name is outlined (a link node and its container share one) and Details
// shows each, and an edge or a container its whole route. fold, the loop chains drawn as one box
// (by their first loop); focus, the selection's upstream and downstream lit, the rest dimmed; zoom
// and zoomc, the view of the step and of the step compared. In edit mode (edit.js) the steps are
// the session's, the shown one its cursor.

import { dragHandle, h, keep, stored } from "./dom.js";
import * as edit from "./edit.js";
import { Minimap, TASK_COLOURS, Viewport, chainHead, foldChains, layout, lineage, relayout, render, resetLayout, routes } from "./graph.js";

const $ = (id) => document.getElementById(id);

// Every POST is JSON: the server refuses another Content-Type (415)
const POST_JSON = { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" };

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || body.message || `${url}: HTTP ${r.status}`);
  return body;
}

function status(msg, err = false) {
  $("status").textContent = msg || "";
  $("status").classList.toggle("error", err);
}

// -- state ----------------------------------------------------------------------

const state = { mode: "view", steps: [], k: 0, cmp: null, sel: null, sels: [], t: null, session: null, platform: null, tech: null, techs: [], expanded: new Set(),
  folded: new Set(), focus: false, lock: false, wantZoom: {} };

// The selection: ``sels`` every selected item in the order selected (shift-click adds or
// removes one), ``sel`` the last, which Details, the Bindings and the hash's ``t`` follow; the
// hash's ``sel`` holds them all, comma-separated.
const selText = () => (state.sels.length ? state.sels.join(",") : null);
function setSels(text) {
  state.sels = text ? String(text).split(",").filter(Boolean) : [];
  state.sel = state.sels.length ? state.sels[state.sels.length - 1] : null;
}
const views = new Map(); // "step|expanded blocks" -> its view, until Reload
const compares = new Map(); // "j/k" -> the compare answer

const num = (v) => (v === null || v === "" || Number.isNaN(Number(v)) ? null : Number(v));

/** The hash as written; ``k`` and ``cmp`` are still its text until ``resolve``. */
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  const zoom = (key) => { // the view: zoom, x, y
    const z = (q.get(key) || "").split(",").map(Number);
    return z.length === 3 && z.every(Number.isFinite) && z[0] > 0 ? z : null;
  };
  return { k: q.get("step") || null, sel: q.get("sel") || null, cmp: q.get("cmp") || null, t: q.get("t") || null, tech: q.get("tech") || null, open: q.get("open") || null,
    fold: q.get("fold") || null, focus: q.get("focus") === "1", zoom: zoom("zoom"), zoomc: zoom("zoomc") };
}

/** The hash's steps as numbers: edit mode's are; view mode's are names, an old hash's a bare
stem or a number, which the server reads (api.py's find_step); none is null. */
async function resolve(want) {
  const find = async (ref) => {
    if (ref === null || state.mode === "edit") return num(ref);
    const i = state.steps.findIndex((s) => s.name === ref);
    return i >= 0 ? i : (await getJSON(`/api/find?step=${encodeURIComponent(ref)}`)).k;
  };
  return { ...want, k: await find(want.k), cmp: await find(want.cmp) };
}

/** A step as the hash holds it: view mode its name, edit mode its number. */
const ref = (k) => (state.mode === "view" && state.steps[k] ? state.steps[k].name : String(k));

function writeHash() {
  const q = new URLSearchParams();
  q.set("step", ref(state.k));
  if (state.sel) q.set("sel", selText());
  if (state.cmp !== null) q.set("cmp", ref(state.cmp));
  if (state.t && state.sel) q.set("t", state.t);
  if (state.tech) q.set("tech", state.tech);
  if (state.expanded.size) q.set("open", [...state.expanded].sort().join(","));
  if (state.folded.size) q.set("fold", [...state.folded].sort().join(","));
  if (state.focus) q.set("focus", "1");
  for (const p of document.querySelectorAll("#graphs .c-panel")) { // each panel's view
    const z = p._vp;
    if (z && z.L) q.set(p._slot === "cmp" ? "zoomc" : "zoom", [z.s.toFixed(3), Math.round(z.tx), Math.round(z.ty)].join(","));
  }
  history.replaceState(null, "", `#${q}`);
}

const clampStep = (k) => Math.min(Math.max(k, 0), state.steps.length - 1);

const techQuery = () => (state.tech ? `?tech=${encodeURIComponent(state.tech)}` : "");

/** The blocks drawn as their nodes (``state.expanded``) as the step route's query. */
const expandQuery = () => [...state.expanded].sort().map((b) => `&expand=${encodeURIComponent(b)}`).join("");
const viewKey = (k) => `${k}|${[...state.expanded].sort().join(",")}`;

async function view(k) {
  const key = viewKey(k);
  if (!views.has(key)) {
    const q = techQuery() + expandQuery();
    views.set(key, await getJSON(`/api/step/${k}${q ? `?${q.replace(/^[?&]/, "")}` : ""}`));
  }
  return views.get(key);
}

async function compare(j, k) {
  const key = `${j}/${k}`;
  if (!compares.has(key)) compares.set(key, await getJSON(`/api/compare/${j}/${k}${techQuery()}`));
  return compares.get(key);
}

/** The technology dropdown: the server's technologies, the shown one selected. */
function techOptions() {
  $("tech").replaceChildren(...state.techs.map((t) => h("option", { value: t }, t)));
  $("tech").value = state.tech || "";
  // one technology is no choice (nangate45 only): the dropdown hides, the heading names it
  $("tech").closest("label").hidden = state.techs.length < 2;
}

/** A technology from the hash or the dropdown: one of the server's, else the server's own. */
const pickTech = (t, fallback) => (state.techs.includes(t) ? t : fallback);

// -- legend ---------------------------------------------------------------------

function legend() {
  const loops = [["none", "untagged map"], ["tile", "tile"], ["temporal", "temporal"], ["spatial", "spatial"]];
  $("legend").replaceChildren(
    ...loops.map(([k, label]) => h("li", {}, h("i", { class: `key loop-${k}` }), label)),
    h("li", {}, h("i", { class: "key c-tasklet" }), "tasklet"),
    h("li", {}, h("i", { class: "key c-unit" }), "unit"),
    h("li", {}, h("i", { class: "key c-fold" }), "fold"),
    h("li", {}, h("i", { class: "key c-link" }), "link"),
    h("li", {}, h("i", { class: "key c-packer" }), "packer, slicer"),
    h("li", {}, h("i", { class: "key c-copy" }), "copy (transpose)"),
    h("li", {}, h("i", { class: "key c-mux" }), "temporal mux (shared unit)"),
    h("li", {}, h("i", { class: "key c-pill" }), "container"),
    h("li", {}, h("i", { class: "key c-pill transient" }), "transient"),
    h("li", {}, h("i", { class: "key c-pill scalar" }), "scalar"),
  );
}

/** The steps as a dropdown's options; from several folders, an optgroup per folder (the
steps from one folder in a row), in the order given. */
function stepOptionNodes(current) {
  const opt = (s, i) => h("option", { value: i, selected: i === current, class: s.past ? "past" : null }, s.label);
  if (!state.steps.some((s) => s.folder)) return state.steps.map(opt);
  const groups = [];
  state.steps.forEach((s, i) => {
    const last = groups[groups.length - 1];
    if (last && last.folder === s.folder) last.options.push(opt(s, i));
    else groups.push({ folder: s.folder, options: [opt(s, i)] });
  });
  return groups.map((g) => h("optgroup", { label: g.folder || "" }, g.options));
}

function stepSelect(current, onChange) {
  return h("select", { "aria-label": "step", onchange: (ev) => onChange(Number(ev.target.value)) },
    stepOptionNodes(current));
}

/** One panel: a step's drawing (graph.js), its folded loop chains as one box each, laid out
before it is shown; with ``pick``, its own step dropdown (Compare); ``slot`` "main" or "cmp" (the
step compared). After ``share`` the head names the tasks in their colours. The drawing pans and
zooms in the panel (graph.js's Viewport), a minimap in its corner when it does not fit; the head
has the zoom's buttons, clean graph, reset and focus. */
function panel(v, k, pick, seed = null, slot = "main") {
  const tasks = (v.drawing && v.drawing.tasks) || [];
  const zoom = h("span", { class: "c-zoom-read muted" }, "100 %");
  const button = (key, text, title) => h("button", { type: "button", title, "data-zoom": key }, text);
  const tools = h("span", { class: "c-zoom" },
    button("out", "−", "zoom out (-)"), zoom, button("in", "+", "zoom in (+)"),
    button("one", "1:1", "100 % (1)"), button("fit", "fit", "the whole step in the panel (0)"),
    button("clean", "clean graph", "lay the step out again: after a drag in the order the boxes are in now, so every edge is routed round it; else from scratch, every order chosen afresh for the fewest crossings (c)"),
    button("reset", "reset", "back to the first layout, every move by hand undone (r)"),
    h("button", { type: "button", title: "light what the selection reads and feeds, dim the rest (f)", "data-zoom": "focus", "aria-pressed": String(state.focus) }, "focus"));
  const head = h("div", { class: "c-panel-head" },
    v.folder ? h("span", { class: "muted" }, `${v.folder} · `) : null,
    pick ? stepSelect(k, pick) : h("b", {}, v.label),
    h("span", { class: "muted" }, v.graph ? ` ${v.graph}` : ""),
    tasks.map((t, i) => h("span", { class: "c-task", style: { "--t": TASK_COLOURS[i % TASK_COLOURS.length] }, title: `task ${t}: its edges and chips in this colour` }, t)),
    tools);
  if (v.error || !v.drawing) return h("section", { class: "c-panel broken" }, head, h("p", { class: "load-error" }, v.error || "no drawing"));
  const d = foldChains(v.drawing, state.folded);
  const canvas = h("div", { class: "c-canvas" });
  const vp = h("div", { class: "c-scroll c-viewport" }, canvas);
  const el = h("section", { class: "c-panel" }, head, vp);
  Object.assign(el, { _view: v, _d: d, _slot: slot });
  el._vp = new Viewport(vp, canvas, (x) => {
    zoom.textContent = `${Math.round(x.s * 100)} %`;
    if (el._mini) el._mini.draw();
    viewChanged(el);
  });
  el._mini = new Minimap(vp, el._vp);
  tools.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-zoom]");
    if (!b) return;
    ev.stopPropagation();
    const z = el._vp;
    ({ out: () => z.zoom(1 / 1.2), in: () => z.zoom(1.2), one: () => z.set(1, z.tx, z.ty), fit: () => z.fit(),
      clean: () => el._clean(), reset: () => el._reset(), focus: () => toggleFocus() })[b.dataset.zoom]();
  });
  el._ready = layout(d, probe, seed).then((L) => { el._layout = L; }, (e) => { el._error = e; });
  el._draw = (prev) => {
    if (el._error) {
      canvas.replaceChildren(h("p", { class: "load-error" }, `layout: ${el._error.message || el._error}`));
      return;
    }
    el._drawn = render(canvas, d, el._layout, () => el._vp.s);
    const want = state.wantZoom[slot];
    if (want) { // the view the hash asked for, once
      el._vp.L = el._layout;
      el._vp.set(...want);
      state.wantZoom[slot] = null;
    } else el._vp.show(el._layout, prev);
    el._mini.draw(el._drawn);
    if (state.mode === "edit") edit.decorate(v, el);
  };
  const again = async (get) => { // laid out again: the boxes glide to their new places
    if (el._error) return;
    const old = new Map([...el.querySelectorAll(".c-box")].map((b) => [b.dataset.id, b.getBoundingClientRect()]));
    const prev = { L: el._vp.L, s: el._vp.s, tx: el._vp.tx, ty: el._vp.ty };
    el._layout = await get();
    el._draw(prev);
    outline();
    if (!still()) glide(el, old);
  };
  el._clean = () => again(() => relayout(d, probe, el._layout.moves && el._layout.moves.size ? el._drawn.places() : null));
  el._reset = () => again(() => resetLayout(d, probe));
  return el;
}

/** A panel's view moved: the hash keeps it; with the views locked (Compare), the other panel
moves with it. */
let locking = false;
let hashSoon = 0;
function viewChanged(el) {
  cancelAnimationFrame(hashSoon);
  hashSoon = requestAnimationFrame(writeHash);
  if (!state.lock || locking) return;
  locking = true;
  for (const p of document.querySelectorAll("#graphs .c-panel")) {
    if (p !== el && p._vp && p._vp.L) p._vp.set(el._vp.s, el._vp.tx, el._vp.ty);
  }
  locking = false;
}

function toggleFocus() {
  state.focus = !state.focus;
  for (const b of document.querySelectorAll('#graphs [data-zoom="focus"]')) b.setAttribute("aria-pressed", String(state.focus));
  writeHash();
  outline();
}

/** Loop ``id``'s chain folded into one box, or unfolded. */
function toggleChain(el, id) {
  const head = el && el._view && el._view.drawing ? chainHead(el._view.drawing, id) : id;
  if (state.folded.has(head)) state.folded.delete(head);
  else state.folded.add(head);
  draw();
}

/** Find: the boxes whose name or text holds ``text``, in every panel; the next one (after the
last found) centred and selected. */
let found = -1;
function find(text, next = true) {
  const q = text.trim().toLowerCase();
  if (!q) return;
  const panels = [...document.querySelectorAll("#graphs .c-panel")].filter((p) => p._drawn);
  const main = panels[panels.length - 1];
  if (!main) return;
  const hits = main._d.nodes.filter((n) => [n.id, n.sel, n.text, ...(n.chips || []).map(([x]) => x)].some((x) => x && String(x).toLowerCase().includes(q)));
  hits.sort((a, b) => Number(b.sel === q || b.id === q) - Number(a.sel === q || a.id === q));
  if (!hits.length) {
    status(`find: no box holds "${text.trim()}"`, true);
    return;
  }
  status("");
  found = next ? (found + 1) % hits.length : 0;
  const n = hits[found];
  for (const p of panels) {
    const b = p._drawn.places().get(n.id);
    if (b) p._vp.centre(b.x + b.w / 2, b.y + b.h / 2, Math.max(p._vp.s, 0.8));
  }
  select(n.sel || n.id);
}

// boxes are measured here before ELK lays them out: hidden, in the page, so its fonts apply
const probe = h("div", { class: "c-probe", "aria-hidden": "true" });
document.body.append(probe);


/** Block ``name`` opened or closed (its view's ``blocks``): opening opens the blocks holding
it, closing closes those inside it; the blocks sharing its units (``together``) go with it. */
async function toggleBlock(name) {
  const v = await view(state.k);
  const list = v.blocks || [];
  const by = new Map(list.map((b) => [b.name, b]));
  const b = by.get(name);
  if (!b || !b.foldable) return;
  const names = [name, ...(b.together || [])];
  const shown = b.collapsed || b.hidden ? false : true;
  for (const x of names) {
    if (shown) {
      state.expanded.delete(x);
      for (const y of list) { // the blocks inside it close too
        let p = y.parent;
        while (p) {
          if (p === x) state.expanded.delete(y.name);
          p = by.get(p) ? by.get(p).parent : null;
        }
      }
    } else {
      let p = x;
      while (p) {
        state.expanded.add(p);
        p = by.get(p) ? by.get(p).parent : null;
      }
    }
  }
  draw();
}

/** The Blocks section: each block indented under the block holding it, a toggle to open or
close it, its name to select it and its text (``dot · dot/dot_3 · W = 4``). */
async function blocks() {
  const box = $("blocks");
  if (!box) return;
  const v = await view(state.k);
  const list = v.blocks || [];
  $("blocks-panel").hidden = !list.length;
  box.replaceChildren(...list.map((b) => h("div", { class: `dfg-block${state.sels.includes(b.name) ? " sel" : ""}`, style: `padding-left: ${b.depth * 0.9}rem` },
    h("button", { type: "button", class: "dfg-block-toggle", disabled: !b.foldable,
      title: !b.foldable ? "drawn as its nodes: it holds part of a top-level nest only" : b.collapsed || b.hidden ? "expand" : "collapse",
      onclick: () => toggleBlock(b.name) }, b.collapsed || b.hidden ? "▸" : "▾"),
    h("button", { type: "button", class: "dfg-cont-name", onclick: () => select(b.name) }, b.name),
    h("span", { class: "mono muted" }, ` ${b.recipe ? `${b.recipe}${Object.entries(b.params).map(([k, x]) => ` · ${k} = ${x}`).join("")}` : b.kernel ? `${b.kernel} · no recipe` : "by hand"}`))));
}

// -- the move between steps: the boxes glide from their old places --------------------------

const GLIDE_MS = 250;
let drawn = null; // the views last drawn, one per panel

const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Where every box is on screen, per panel: its drawing id -> its rect. */
function places() {
  return [...$("graphs").children].map((p) => new Map([...p.querySelectorAll(".c-box")].map((el) => [el.dataset.id, el.getBoundingClientRect()])));
}

/** A new step's panel: each box that was on screen moves from there to its place, a new one
fades in; the edges show when all settle. */
function glide(panel, old) {
  const runs = [];
  for (const el of panel.querySelectorAll(".c-box")) {
    const was = old.get(el.dataset.id);
    if (!was) {
      runs.push(el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: GLIDE_MS * 1.5, easing: "ease-out" }));
      continue;
    }
    const now = el.getBoundingClientRect();
    const dx = was.left - now.left, dy = was.top - now.top;
    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) continue;
    runs.push(el.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }], { duration: GLIDE_MS, easing: "ease-out" }));
  }
  if (!runs.length) return;
  panel.classList.add("settling");
  Promise.all(runs.map((a) => a.finished.catch(() => null))).then(() => panel.classList.remove("settling"));
}

// -- selection --------------------------------------------------------------------

function outline() {
  for (const el of document.querySelectorAll("#graphs .sel")) {
    el.classList.remove("sel");
    if (el._marker) el.setAttribute("marker-end", `url(#${el._marker})`);
  }
  for (const name of state.sels) { // every selected item highlighted
    for (const el of document.querySelectorAll(`#graphs [data-sel="${CSS.escape(name)}"]`)) {
      el.classList.add("sel");
      if (el._marker) el.setAttribute("marker-end", `url(#${el._marker}-sel)`);
    }
  }
  // a selected unit's instance shared by other nodes (api.py's ``shared``): those too
  for (const panelEl of document.querySelectorAll("#graphs .c-panel")) {
    const shared = (panelEl._view && panelEl._view.shared) || {};
    for (const name of state.sels) {
      for (const other of shared[name] || []) {
        for (const el of panelEl.querySelectorAll(`.c-node[data-sel="${CSS.escape(other)}"]`)) el.classList.add("sel");
      }
    }
  }
  // an edge's or a container's whole route (graph.js's ``routes``); with focus, what the
  // selection reads and feeds lit and the rest dimmed
  for (const p of document.querySelectorAll("#graphs .c-panel")) {
    if (!p._d || !p._drawn) continue;
    for (const id of routes(p._d, state.sels)) {
      for (const el of p.querySelectorAll(`[data-edge="${CSS.escape(id)}"]`)) {
        el.classList.add("sel");
        if (el._marker) el.setAttribute("marker-end", `url(#${el._marker}-sel)`);
      }
    }
    const lit = state.focus ? lineage(p._d, state.sels) : null;
    const canvas = p.querySelector(".c-canvas");
    canvas.classList.toggle("focus", Boolean(lit));
    for (const el of canvas.querySelectorAll(".lit")) el.classList.remove("lit");
    if (lit) {
      for (const id of lit.boxes) { const b = canvas.querySelector(`.c-box[data-id="${CSS.escape(id)}"]`); if (b) b.classList.add("lit"); }
      for (const id of lit.edges) for (const el of canvas.querySelectorAll(`[data-edge="${CSS.escape(id)}"]`)) el.classList.add("lit");
    }
  }
  showPartners();
}

// A selected map's partners (api.py's ``partners``, found by the transforms' own sites): each
// partner map outlined in its transform's colour with a tag naming the transform, and for a
// fuse the marker beside the transient between the two nests. Tags and markers are placed
// absolutely, so the boxes do not move and the edges stay where they are.
const PARTNER_CLASSES = ["p-on", "p-fuse_map", "p-fuse_seed", "p-interchange_map", "p-near"];

function showPartners() {
  for (const el of document.querySelectorAll("#graphs .p-tags, #graphs .p-marker")) el.remove();
  for (const el of document.querySelectorAll("#graphs .p-on")) el.classList.remove(...PARTNER_CLASSES);
  for (const panelEl of document.querySelectorAll("#graphs .c-panel")) {
    const v = panelEl._view;
    if (!v || !v.partners) continue;
    for (const name of state.sels) {
      for (const p of v.partners[name] || []) {
        // a near partner (refused for different ranges only): grey, its tag the refusal
        const cls = p.near ? "p-near" : `p-${p.transform}`;
        const map = panelEl.querySelector(`.c-map[data-sel="${CSS.escape(p.map)}"]`);
        const bar = map && map.querySelector(":scope > .c-bar");
        if (bar) {
          map.classList.add("p-on", cls);
          const tags = bar.querySelector(":scope > .p-tags") || bar.appendChild(h("span", { class: "p-tags" }));
          if (!tags.querySelector(`.${cls}`)) tags.append(h("span", { class: `p-tag ${cls}` }, p.near || p.transform));
        }
        const box = p.marker && p.marker.box && panelEl.querySelector(`[data-box="${CSS.escape(p.marker.box)}"]`);
        if (box && !box.querySelector(`.p-marker.${cls}`)) {
          box.classList.add("p-on", cls);
          box.append(h("span", { class: `p-marker ${cls}` }, p.marker.text));
        }
      }
    }
  }
}

/** Select ``name`` (null clears); with ``add`` (shift-click) add it to the selection, or remove
it when it is in it already. */
function select(name, add = false) {
  const before = selText();
  if (add && name) {
    setSels((state.sels.includes(name) ? state.sels.filter((x) => x !== name) : [...state.sels, name]).join(","));
  } else setSels(name);
  if (selText() !== before) state.t = null;
  writeHash();
  outline();
  details();
  containers();
  bindings();
  blocks();
  if (state.mode === "edit") edit.sidebar();
}

/** Step k shown: in edit mode the cursor moves there (the session's history). */
function goto(k) {
  if (state.mode === "edit") edit.move(k);
  else {
    state.k = k;
    draw();
  }
}

const KINDS = ["node", "block", "container", "group", "accelerator", "edge"];

function valueText(v) {
  if (v === null || v === undefined) return "–";
  if (Array.isArray(v)) return v.map((x) => (typeof x === "object" ? JSON.stringify(x) : String(x))).join(", ") || "–";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** Compare's two columns (api.py's compare_view ``columns``): a label column and the two
steps' texts, the left step first, rows that differ marked; a row's ``under`` opens a section. */
function columnsTable(labels, rows) {
  const body = [];
  let under = null;
  for (const r of rows) {
    if (r.under && r.under !== under) body.push(h("tr", { class: "c-cols-section" }, h("th", { colspan: 3 }, r.under)));
    under = r.under;
    body.push(h("tr", { class: r.differs ? "differs" : null },
      h("th", { scope: "row", class: r.under ? "sub" : null }, r.label),
      h("td", { class: "mono" }, r.a), h("td", { class: "mono" }, r.b)));
  }
  return h("table", { class: "c-cols" },
    h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", { scope: "col" }, labels[0]), h("th", { scope: "col" }, labels[1]))),
    h("tbody", {}, body));
}

/** Details: every item of the selected name, a labelled list; with Compare, two columns. */
async function details() {
  const box = $("details");
  if (!state.sel) {
    box.replaceChildren(h("p", { class: "muted" }, "Click a box, a container, an edge or a group outline; Esc clears."));
    return;
  }
  const out = [];
  if (state.cmp !== null) {
    const c = (await compare(state.cmp, state.k)).columns;
    for (const kind of KINDS) {
      const d = c ? c.details[`${kind}:${state.sel}`] : undefined;
      if (!d) continue;
      out.push(h("div", { class: "c-details" }, h("h3", {}, `${kind} ${state.sel}${d.where ? ` (${d.where})` : ""}`),
        columnsTable(c.labels, d.rows)));
    }
    box.replaceChildren(...(out.length ? out : [h("p", { class: "muted" }, `${state.sel}: in neither step`)]));
    return;
  }
  const b = await view(state.k);
  for (const kind of KINDS) {
    const key = `${kind}:${state.sel}`;
    const item = b.details ? b.details[key] : undefined;
    if (item === undefined) continue;
    const rows = Object.keys(item).filter((f) => f !== "kind").map((f) => (f === "cost"
      ? costRows(item.cost)
      : [h("dt", {}, f), h("dd", { class: "mono" }, valueText(item[f]))]));
    out.push(h("div", { class: "c-details" }, h("h3", {}, `${kind} ${state.sel}`), h("dl", { class: "facts" }, rows)));
  }
  box.replaceChildren(...(out.length ? out : [h("p", { class: "muted" }, `${state.sel}: not in this step`)]));
}

/** Details' cost (api.py's cost_details): a nested list of its texts. */
function costRows(cost) {
  if (!cost) return [h("dt", {}, "cost"), h("dd", { class: "mono" }, "–")];
  const inner = Object.entries(cost).map(([k, v]) => [h("dt", {}, k), h("dd", { class: "mono" }, v)]);
  return [h("dt", {}, "cost"), h("dd", {}, h("dl", { class: "facts c-cost" }, inner))];
}

// -- the Containers (the left column) ----------------------------------------------

/** Every container that a symbol's shape uses, highlighted in the Containers and the graph
(focus or hover on the symbol in the Symbols). */
function hlSymbol(name, on) {
  const v = views.get(viewKey(state.k));
  const users = new Set(((v && v.containers_section) || []).filter((c) => c.symbols.includes(name)).map((c) => c.name));
  for (const el of document.querySelectorAll("#containers [data-cont]")) {
    el.classList.toggle("sym-hl", on && users.has(el.dataset.cont));
  }
  for (const el of document.querySelectorAll("#graphs .c-pill[data-sel]")) {
    el.classList.toggle("sym-hl", on && users.has(el.dataset.sel));
  }
  for (const el of document.querySelectorAll("[data-symbol]")) {
    el.classList.toggle("sym-on", on && el.dataset.symbol === name);
  }
}

const hover = (name) => ({
  onmouseenter: () => hlSymbol(name, true), onmouseleave: () => hlSymbol(name, false),
  onfocusin: () => hlSymbol(name, true), onfocusout: () => hlSymbol(name, false),
});

/** One symbol of the Symbols block: in edit mode an input on its param (``POST /api/params``,
the session's replay and refusal), else its value; focus or hover highlights its containers. */
function symbolEl(x) {
  const value = x.value === null || x.value === undefined ? "?" : String(x.value);
  const input = x.editable
    ? h("input", { value, size: Math.max(3, value.length), "aria-label": `param ${x.param}`,
      onchange: (ev) => edit.setParam(x.param, ev.target) })
    : h("b", { class: "mono" }, value);
  const used = x.containers.length ? `used by ${x.containers.join(", ")}` : "no container uses it";
  return h("label", { class: "dfg-sym", "data-symbol": x.name, tabindex: x.editable ? null : 0,
    title: `${x.name}: ${used}${x.editable ? "; a change replays every step" : ""}`, ...hover(x.name) },
  h("span", { class: "mono" }, x.name), input);
}

/** The Containers section: the Symbols first (each kernel symbol and its value), then a line per
container, ``A: N · L1 · 16 KiB`` (its name selects it; its sizes and dtype are in Details). */
async function containers() {
  const box = $("containers");
  const v = await view(state.k);
  const list = v.containers_section;
  if (!list) {
    box.replaceChildren(h("p", { class: "muted" }, v.error || "no graph"));
    return;
  }
  const symbols = v.symbols_section || [];
  box.replaceChildren(...[
    symbols.length ? h("div", { class: "dfg-symbols" }, h("h3", {}, "Symbols"), symbols.map(symbolEl)) : null,
    ...list.map((c) => h("div", { class: `dfg-cont${state.sel === c.name ? " sel" : ""}`, "data-cont": c.name },
      h("button", { type: "button", class: "dfg-cont-name", onclick: () => select(c.name) }, c.name),
      h("span", { class: "mono" }, `: ${c.line}`))),
  ].filter(Boolean));
}

// -- the Bindings (the left column): each param with what it configures -----------------

/** Every entry naming param ``name`` highlighted (hover or focus on one of its inputs). */
function hlParam(name, on) {
  for (const el of document.querySelectorAll("#bindings [data-param]")) {
    el.classList.toggle("param-hl", on && el.dataset.param === name);
  }
}

/** One param of an entry: ``W = 8``, an input in edit mode (``POST /api/params``, every entry
naming it changes with it), what it is shared with. */
function paramEl(p) {
  const value = p.value === null || p.value === undefined ? "?" : String(p.value);
  const shared = p.shared_with && p.shared_with.length ? `shared with ${p.shared_with.join(", ")}` : "";
  return h("span", { class: "dfg-param", "data-param": p.name, title: shared || `param ${p.name}`,
    onmouseenter: () => hlParam(p.name, true), onmouseleave: () => hlParam(p.name, false),
    onfocusin: () => hlParam(p.name, true), onfocusout: () => hlParam(p.name, false) },
  h("span", { class: "mono" }, `${p.name} = `),
  p.editable
    ? h("input", { value, size: Math.max(3, value.length), "aria-label": `param ${p.name}`, onchange: (ev) => edit.setParam(p.name, ev.target) })
    : h("b", { class: "mono" }, value),
  shared ? h("small", {}, ` · ${shared}`) : null);
}

async function bindings() {
  const box = $("bindings");
  const v = await view(state.k);
  const b = v.bindings_section;
  if (!b) {
    box.replaceChildren();
    return;
  }
  const entry = (e) => h("div", { class: `dfg-binding ${e.kind}${state.sel === e.id ? " sel" : ""}` },
    h("button", { type: "button", class: "dfg-cont-name", onclick: () => select(e.id) }, e.label),
    e.params.map((p) => [" · ", paramEl(p)]),
    e.note ? h("small", { class: "muted" }, ` · ${e.note}`) : null);
  // the tree: accelerator ▸ group ▸ entries, each level a drop-down, open by default;
  // a unit alone, a split not bound yet at the top
  const branch = (n) => (n.kind === "entry" ? entry(b.entries[n.index])
    : h("details", { class: `dfg-bind-${n.kind}`, open: true },
      h("summary", {}, h("span", { class: "kind" }, n.kind), " ", h("b", {}, n.name)),
      h("div", { class: "dfg-bind-kids" }, n.children.map(branch))));
  const rows = b.tree ? b.tree.map(branch) : b.entries.map(entry);
  const unused = b.unused.length ? h("div", { class: "dfg-binding unused" }, h("span", { class: "muted" }, "unused: "),
    b.unused.map((p, i) => [i ? ", " : "", paramEl({ ...p, shared_with: [] })])) : null;
  box.replaceChildren(...(rows.length || unused ? [...rows, unused].filter(Boolean) : [h("p", { class: "muted" }, "Nothing split or bound yet.")]));
}

// -- the Estimate ------------------------------------------------------------------

function hoverGroup(name, on) {
  for (const el of document.querySelectorAll(`#graphs [data-group="${CSS.escape(name)}"]`)) el.classList.toggle("hover", on);
}

// short labels, so the table fits the right panel at 1280 px: the datapath under the block
// in one cell, the numbers right-aligned; the units in each heading's title
const COST_COLUMNS = [["block", "block · datapath", "the unit, buffer link, control unit, streamer or the words through the crossbar and banks, its datapath under it"],
  ["area_text", "µm²", "area, µm² (3 significant figures)"],
  ["energy_text", "nJ", "energy per run, nJ (3 decimals)"], ["share_text", "share", "share of the accelerator's energy"]];

/** One accelerator's cost table (api.py's cost_tables, or compare_cost_tables' before → after
cells): a row per unit, buffer link, control unit and streamer, its words through the crossbar
and the banks, then the total. */
function costTable(t) {
  const num = (key) => key !== "block";
  const l1 = (r) => r.kind === "l1" || String(r.block).startsWith("L1 accesses");
  const cell = (r, key) => {
    if (key !== "block") return r[key];
    const name = l1(r) ? h("span", {}, "L1 accesses") : r.block;
    return [h("div", {}, name), r.datapath ? h("div", { class: "c-cost-dp" }, r.datapath) : null];
  };
  const tr = (r, cls) => h("tr", { class: [cls, r.kind === "streamer" ? "streamer" : null, r.differs ? "differs" : null, r.found === false ? "missing" : null].filter(Boolean).join(" ") || null,
    title: l1(r) ? String(r.block) : null },
  COST_COLUMNS.map(([key]) => h(key === "block" ? "th" : "td", { class: num(key) ? "num" : null, scope: key === "block" ? "row" : null }, cell(r, key))));
  return h("div", { class: "c-cost-wrap" }, h("table", { class: "c-cost-table" },
    h("caption", {}, t.accelerator),
    h("thead", {}, h("tr", {}, COST_COLUMNS.map(([key, label, title]) => h("th", { scope: "col", class: num(key) ? "num" : null, title }, label)))),
    h("tbody", {}, t.rows.map((r) => tr(r, null)), tr(t.total, "total"))));
}

/** One bar of the cost by part, the whole to scale with a segment per part costed, as the
memory tab's scale bar draws a memory (``key`` "area" or "energy"; the shares are api.py's). */
function partBar(parts, key, label) {
  let left = 0;
  const segs = parts.filter((p) => p[`${key}_share`] > 0).map((p) => {
    const w = 100 * p[`${key}_share`];
    const seg = h("div", { class: `c-part-seg p-${p.name}`, style: { left: `${left}%`, width: `${w}%` },
      title: `${p.label}: ${p[`${key}_text`]}, ${p[`${key}_share_text`]} of the ${label}` },
    w >= 9 ? h("span", {}, p.name) : null);
    left += w;
    return seg;
  });
  return h("div", { class: "c-part-bar-wrap" },
    h("div", { class: "c-part-bar-label muted" }, label),
    h("div", { class: "c-part-bar", role: "img", "aria-label": `${label} by part` }, segs));
}

/** The cost by part: its two bars, then, folded, a row per part (area, share, energy, share) and
the totals, with and without the banks. */
function costParts(c) {
  const parts = c.parts || [];
  if (!parts.length) return [];
  const row = (p) => h("tr", {},
    h("th", { scope: "row" }, h("i", { class: `c-part-key p-${p.name}` }), p.label),
    ...["area_text", "area_share_text", "energy_text", "energy_share_text"].map((k) => h("td", { class: "num" }, p[k])));
  return [partBar(parts, "area", "area"), partBar(parts, "energy", "energy per run"),
    h("details", { class: "c-part-details" }, h("summary", {}, "cost by part, as a table"),
      h("div", { class: "c-cost-wrap" }, h("table", { class: "c-cost-table" },
        h("thead", {}, h("tr", {}, ["part", "area", "share", "energy", "share"].map((x, i) => h("th", { scope: "col", class: i ? "num" : null }, x)))),
        h("tbody", {}, parts.map(row),
          h("tr", { class: "total" }, h("th", { scope: "row" }, "total"), h("td", { class: "num" }, c.area_text), h("td", {}), h("td", { class: "num" }, c.energy_text), h("td", {})),
          h("tr", {}, h("th", { scope: "row" }, "without the banks"), h("td", { class: "num" }, c.area_design_text), h("td", {}), h("td", {}), h("td", {}))))))];
}

/** Compare's cost bars (api.py's compare_cost_bars): per area and energy, the two steps' bars one
above the other on one scale, the larger total the whole width. */
function compareBars(bars) {
  if (!bars) return [];
  const bar = (side, what) => {
    let left = 0;
    const segs = side.segments.map((p) => {
      const w = 100 * p.width;
      const seg = h("div", { class: `c-part-seg p-${p.name}`, style: { left: `${left}%`, width: `${w}%` },
        title: `${side.label}, ${p.label}: ${p.text}` }, w >= 9 ? h("span", {}, p.name) : null);
      left += w;
      return seg;
    });
    return h("div", { class: "c-part-row" },
      h("div", { class: "c-part-side" }, h("span", {}, side.label), h("span", { class: "nb" }, side.total_text)),
      h("div", { class: "c-part-bar", role: "img", "aria-label": `${side.label}: ${what} by part` }, segs));
  };
  return [["area", "area"], ["energy", "energy per run"]].map(([key, what]) => h("div", { class: "c-part-bar-wrap" },
    h("div", { class: "c-part-bar-label muted" }, `${what}, on one scale`), ...bars[key].map((side) => bar(side, what))));
}

async function estimate() {
  const box = $("estimate");
  if (state.cmp !== null) { // Compare: the server's two columns, the left step first
    const cmp = await compare(state.cmp, state.k);
    if (!cmp.columns) {
      box.replaceChildren(h("p", { class: "load-error" }, cmp.error || "no estimate"));
      return;
    }
    const tables = cmp.columns.cost_tables || [];
    box.replaceChildren(...[columnsTable(cmp.columns.labels, cmp.columns.estimate),
      cmp.columns.caveat ? h("p", { class: "c-est-line muted" }, cmp.columns.caveat) : null,
      ...compareBars(cmp.columns.cost_bars),
      tables.length ? h("details", { class: "c-part-details" }, h("summary", {}, `cost per block, ${cmp.columns.labels[0]} → ${cmp.columns.labels[1]}`),
        tables.map(costTable)) : null].filter(Boolean));
    return;
  }
  const b = await view(state.k);
  if (b.estimate_error || !b.summary) {
    box.replaceChildren(h("p", { class: "load-error" }, b.estimate_error || b.error || "no estimate"));
    return;
  }
  const s = b.summary;
  const rows = s.groups.map((g) => h("div", {
    class: "c-est-group", onmouseenter: () => hoverGroup(g.name, true), onmouseleave: () => hoverGroup(g.name, false),
  },
  h("b", {}, g.name), " ",
  h("span", { class: "mono" }, g.formula),
  // each figure with its unit on one line (``56 000 µm²`` never breaks); the API's text as is
  h("small", { title: "the expected BUSY_COUNTER, cycles + fill: a guide, not a check" }, " ",
    [`expected BUSY_COUNTER ${g.counter}`, `channels ${g.channels}`, ...(g.area_text ? [g.area_text, g.energy_text] : [])]
      .map((x, i) => [i ? " · " : "", h("span", { class: "nb" }, x)])),
  g.bank_note ? h("p", { class: "c-bank-note", title: "L1's single-port banks serve one word each per cycle: a bound on the streams' L1 time, not on the BUSY_COUNTER" }, `bank bound: ${g.bank_note}`): null));
  const c = s.cost;
  const cost = c ? [
    h("p", { class: "c-est-head" }, h("b", {}, c.heading), ": ", h("b", { class: "nb" }, c.area_text), " · ", h("b", { class: "nb" }, c.energy_text), " per run"),
    h("p", { class: "c-est-line muted" }, c.caveat),
    ...costParts(c),
    (c.tables || []).length ? h("details", { class: "c-part-details" }, h("summary", {}, "cost per block, as a table"), c.tables.map(costTable)) : null,
    h("p", { class: "c-est-line" }, "not characterized: ", c.missing.length ? c.missing.join(", ") : "none"),
    c.not_costed && c.not_costed.length ? h("p", { class: "c-est-line" }, "not costed: ", c.not_costed.join("; ")) : null,
    c.extrapolated && c.extrapolated.length ? h("p", { class: "c-est-line" }, "outside the tables' grid, extrapolated: ", c.extrapolated.join(", ")) : null,
  ].filter(Boolean) : [];
  // edit mode: Save's design checks on this step: the problems first, then the notes,
  // a "before Save" one tagged so (its message's own prefix dropped); the log's counts link here
  const check = (kind, p) => h("li", { class: `c-check ${kind}${p.before_save ? " before-save" : ""}` },
    h("div", { class: "c-check-head" },
      p.before_save ? h("span", { class: "c-check-tag" }, "before Save") : null,
      h("b", { class: "mono" }, `[${p.code}]`), " ", h("span", { class: "mono" }, p.where)),
    h("div", {}, p.before_save ? p.message.replace(/^before Save: /, "") : p.message),
    p.fix ? h("div", { class: "c-check-fix" }, h("span", { class: "muted" }, "fix: "), h("span", { class: "mono" }, p.fix)) : null);
  const problems = b.problems || [], notes = b.notes || [];
  const checks = state.mode === "edit" && b.problems !== undefined ? [h("div", { id: "checks", class: "c-checks" },
    h("p", { class: "c-est-head" }, h("b", {}, "Design checks"), ` (Save's): ${problems.length} problem${problems.length === 1 ? "" : "s"}, ${notes.length} note${notes.length === 1 ? "" : "s"}`),
    problems.length ? h("ul", { id: "checks-problems", class: "c-check-list" }, problems.map((p) => check("problem", p))) : null,
    notes.length ? h("ul", { id: "checks-notes", class: "c-check-list" }, notes.map((p) => check("note", p))) : null)] : [];
  const l1 = s.l1;
  const l1Text = l1.size ? `${l1.footprint_text} of ${l1.size_text} (${l1.share_text})` : l1.footprint_text;
  box.replaceChildren(...[
    h("p", { class: "c-est-head" }, "cycles: ", (s.cycles || []).map((c, i) => [i ? (i === s.cycles.length - 1 ? " = " : " + ") : "",
      h("span", { class: "c-est-cycles", title: c.title }, `${c.label} `, h("b", {}, c.value))])),
    s.unmapped_note ? h("p", { class: "c-est-line muted" }, s.unmapped_note) : null,
    ...rows,
    h("p", { class: "c-est-line" }, "L1 ", l1Text),
    l1.size ? h("div", { class: "c-bar-track" }, h("i", { style: { width: l1.share_css } })) : null,
    h("p", { class: "c-est-line" }, `reads ${s.reads} · writes ${s.writes} words`),
    s.splits.length ? h("p", { class: "c-est-line" }, "lanes ", s.splits.map((x) => `${x.map} ${x.text}`).join("; ")) : null,
    s.streamers ? h("p", { id: "est-streamers", class: "c-est-line", title: "one streamer set per accelerator; each port's streamer is costed in the tables below" },
      s.streamers.text, s.streamers.accelerators.length > 1 ? ": " + s.streamers.accelerators.map((a) => `${a.name}${a.csrs_text ? " " + a.csrs_text : ""}`).join(", ") : "") : null,
    ...checks,
    ...cost,
  ].filter(Boolean));
}

// -- drawing a state ----------------------------------------------------------------

/** The layout of the step shown (the last panel's), the order the steps beside it start from. */
function shownLayout() {
  const ps = [...$("graphs").children];
  const p = ps[ps.length - 1];
  return (p && p._vp && p._vp.L) || null;
}

/** The steps either side of the one shown, fetched and laid out while the page is idle, so ← →
draw at once. A step the server will not give yet (a redo step not replayed) is left. */
function prefetch() {
  const k = state.k;
  const idle = window.requestIdleCallback || ((f) => setTimeout(f, 50));
  idle(async () => {
    for (const j of [k + 1, k - 1]) {
      if (j < 0 || j >= state.steps.length || state.k !== k) continue;
      try {
        const v = await view(j);
        if (v.drawing) await layout(foldChains(v.drawing, state.folded), probe, shownLayout());
      } catch {
        // not there yet: drawn when asked for
      }
    }
  });
}

async function draw() {
  writeHash();
  $("step").value = String(state.k);
  $("compare").checked = state.cmp !== null;
  $("lock").closest("label").hidden = state.cmp === null;
  $("prev").disabled = state.k <= 0;
  $("next").disabled = state.k >= state.steps.length - 1;
  // "Loading…" only for a slow step: the status line takes a row of the header, so showing it
  // on every step would move the page down and back
  const slow = setTimeout(() => status("Loading…"), 300);
  try {
    const shown = await view(state.k);
    const there = state.sels.filter((x) => KINDS.some((kind) => shown.details && shown.details[`${kind}:${x}`]));
    if (there.length !== state.sels.length) {
      setSels(there.join(",")); // kept across steps only where it exists
      state.t = null;
      writeHash();
    }
    // each panel laid out in the order the panel now on screen has (graph.js's ``seeded``)
    const seeds = [...$("graphs").children].map((p) => (p._vp && p._vp.L) || null);
    const seed = (i, n) => (seeds.length === n ? seeds[i] : seeds[seeds.length - 1] || null);
    let panels;
    if (state.cmp === null) {
      panels = [panel(shown, state.k, null, seed(0, 1), "main")];
    } else {
      panels = [
        panel(await view(state.cmp), state.cmp, (j) => { state.cmp = j; draw(); }, seed(0, 2), "cmp"),
        panel(shown, state.k, goto, seed(1, 2), "main"),
      ];
    }
    await Promise.all(panels.map((p) => p._ready));
    const old = places();
    // each panel's view of the step before: the boxes both hold stay where they were
    const prev = [...$("graphs").children].map((p) => p._vp && { L: p._vp.L, s: p._vp.s, tx: p._vp.tx, ty: p._vp.ty });
    $("graphs").replaceChildren(...panels);
    panels.forEach((p, i) => p._draw && p._draw(prev.length === panels.length ? prev[i] : null));
    $("graphs").classList.toggle("two", panels.length === 2);
    const now = panels.map((p) => p._view);
    if (!still() && drawn && old.length === panels.length) {
      panels.forEach((p, i) => p._view !== drawn[i] && glide(p, old[i]));
    }
    drawn = now;
    outline();
    await Promise.all([details(), estimate(), blocks(), containers(), bindings(), state.mode === "edit" ? edit.sidebar() : null]);
    clearTimeout(slow);
    const broken = state.steps.filter((s) => s.error).length;
    status(broken ? `${broken} of ${state.steps.length} files did not load` : "", broken > 0);
    prefetch();
  } catch (e) {
    clearTimeout(slow);
    status(String(e.message || e), true);
  }
}

function stepOptions() {
  $("step").replaceChildren(...stepOptionNodes(null));
}

/** Edit mode: the session as the server has it (its steps, its cursor shown), or the start
dialog before one is open. */
async function show() {
  const s = state.session;
  if (!s) {
    edit.start();
    return;
  }
  state.steps = s.labels.map((label, i) => ({ label, past: i > s.cursor }));
  state.k = s.cursor;
  if (state.cmp !== null) state.cmp = clampStep(state.cmp);
  stepOptions();
  edit.top(s);
  $("title").textContent = "SNAX-FORGE DFG editor";
  await draw();
}

async function load() {
  status("Loading…");
  let want = readHash();
  state.expanded = new Set((want.open || "").split(",").filter(Boolean));
  state.folded = new Set((want.fold || "").split(",").filter(Boolean));
  state.focus = want.focus;
  state.wantZoom = { main: want.zoom, cmp: want.zoomc };
  try {
    const mode = await getJSON("/api/mode");
    state.mode = mode.mode;
    state.platform = mode.platform || null;
    state.techs = mode.techs || [];
    state.tech = pickTech(want.tech, mode.tech || null);
    techOptions();
    if (state.mode === "edit") {
      want = await resolve(want);
      edit.init({ state, status, clear, show, view, writeHash, hover });
      state.session = await edit.session();
      setSels(want.sel);
      state.t = want.t;
      state.cmp = want.cmp;
      if (state.session && want.k !== null && want.k !== state.session.cursor && want.k <= state.session.n) {
        await edit.move(want.k); // a link to another step: the cursor goes there
        return;
      }
      await show();
      return;
    }
    state.steps = await getJSON("/api/steps");
    want = await resolve(want);
  } catch (e) {
    status(String(e.message || e), true);
    return;
  }
  stepOptions();
  state.k = clampStep(want.k === null ? 0 : want.k);
  state.cmp = want.cmp === null ? null : clampStep(want.cmp);
  setSels(want.sel);
  draw();
}

function clear() {
  views.clear();
  compares.clear();
}

// -- the side panels' widths ---------------------------------------------------------------

// Dragged on their inner edge as the log is (dom.js's dragHandle), from MIN_SIDE px to 45% of
// the window, kept in localStorage; a double-click gives back the CSS default (--left, --side).
const MIN_SIDE = { left: 170, right: 230 };

function sidePanels() {
  const layout = document.querySelector(".dfg-layout");
  for (const side of ["left", "right"]) {
    const prop = side === "left" ? "--left" : "--side";
    const key = `snax-forge.dfg.${side}-width`;
    const panel = side === "left" ? $("sidebar") : $("right");
    const set = (px) => layout.style.setProperty(prop, `${Math.round(Math.min(0.45 * innerWidth, Math.max(MIN_SIDE[side], px)))}px`);
    const kept = stored(key);
    if (kept) set(kept);
    const handle = $(`${side}-handle`);
    dragHandle(handle, {
      delta: (dx) => (side === "left" ? dx : -dx), // the left panel grows rightwards, the right leftwards
      measure: () => panel.getBoundingClientRect().width,
      place: set,
      done: () => keep(key, Math.round(panel.getBoundingClientRect().width)),
    });
    handle.addEventListener("dblclick", () => {
      layout.style.removeProperty(prop);
      keep(key, 0);
    });
  }
}

// -- controls ----------------------------------------------------------------------

$("step").addEventListener("change", (ev) => goto(Number(ev.target.value)));
$("prev").addEventListener("click", () => goto(clampStep(state.k - 1)));
$("next").addEventListener("click", () => goto(clampStep(state.k + 1)));
$("tech").addEventListener("change", (ev) => {
  state.tech = pickTech(ev.target.value, state.tech);
  clear(); // every view and comparison was on the old technology
  draw();
});
$("compare").addEventListener("change", (ev) => {
  state.cmp = ev.target.checked ? clampStep(state.k - 1) : null;
  draw();
});
$("lock").addEventListener("change", (ev) => { // Compare's two views pan and zoom together
  state.lock = ev.target.checked;
  const ps = [...document.querySelectorAll("#graphs .c-panel")].filter((p) => p._vp);
  if (state.lock && ps.length === 2) viewChanged(ps[1]);
});
$("find").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") {
    ev.preventDefault();
    find(ev.target.value);
  } else if (ev.key === "Escape") ev.target.blur();
});
$("find").addEventListener("input", (ev) => { found = -1; if (ev.target.value.trim().length > 1) find(ev.target.value, false); });
// A click on an item selects it; on empty space it clears the selection, as Esc does.
$("graphs").addEventListener("click", (ev) => {
  if (ev.target.closest("select")) return;
  const t = ev.target.closest("[data-sel]");
  if (ev.shiftKey && !t) return; // a shift-click on empty space keeps the selection
  select(t ? t.getAttribute("data-sel") : null, ev.shiftKey);
});
// A double-click on a block's box or outline opens or closes it (and the blocks sharing its units).
$("graphs").addEventListener("dblclick", (ev) => {
  const t = ev.target.closest("[data-block]");
  if (t) {
    toggleBlock(t.getAttribute("data-block"));
    return;
  }
  const c = ev.target.closest("[data-chain]"); // a loop: its chain folded or unfolded
  if (c) toggleChain(c.closest(".c-panel"), c.getAttribute("data-chain"));
});
// Keys: ← → the steps; Esc closes an open form, then clears the selection; edit mode: Enter
// applies the form that has the focus, Ctrl+Z (⌘Z) undoes, Ctrl+Shift+Z redoes. In a text field
// only Enter and Esc in a form count; a dialog keeps its own keys.
const MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
document.addEventListener("keydown", (ev) => {
  if (document.querySelector("dialog[open]")) return;
  const typing = ev.target.closest("input, select, textarea, [contenteditable]");
  const form = ev.target.closest(".dfg-form");
  if (ev.key === "Escape" && (!typing || form)) {
    if (state.mode === "edit" && state.t) edit.closeForm();
    else select(null);
  } else if (ev.key === "Enter" && form && !ev.target.closest("button")) {
    ev.preventDefault();
    form.requestSubmit();
  } else if (typing) {
    return;
  } else if (state.mode === "edit" && (MAC ? ev.metaKey : ev.ctrlKey) && ev.key.toLowerCase() === "z") {
    ev.preventDefault();
    if (ev.shiftKey) edit.redo();
    else edit.undo();
  } else if (ev.altKey || ev.ctrlKey || ev.metaKey) {
    return;
  } else if (ev.key === "c" || ev.key === "r") { // clean graph or reset, in every panel
    for (const p of document.querySelectorAll("#graphs .c-panel")) if (p._clean) (ev.key === "c" ? p._clean : p._reset)();
  } else if (ev.key === "f") {
    toggleFocus();
  } else if (ev.key === "/") {
    ev.preventDefault();
    $("find").focus();
  } else if (["0", "1", "+", "=", "-"].includes(ev.key)) { // the zoom, in every panel
    for (const p of document.querySelectorAll("#graphs .c-panel")) {
      const z = p._vp;
      if (!z) continue;
      if (ev.key === "0") z.fit();
      else if (ev.key === "1") z.set(1, z.tx, z.ty);
      else z.zoom(ev.key === "-" ? 1 / 1.2 : 1.2);
    }
  } else if (ev.key === "ArrowLeft" && state.k > 0) goto(state.k - 1);
  else if (ev.key === "ArrowRight" && state.k < state.steps.length - 1) goto(state.k + 1);
});
window.addEventListener("hashchange", async () => {
  let want;
  try {
    want = await resolve(readHash());
  } catch (e) {
    status(String(e.message || e), true);
    return;
  }
  const open = (want.open || "").split(",").filter(Boolean).sort().join(",");
  const fold = (want.fold || "").split(",").filter(Boolean).sort().join(",");
  if (open !== [...state.expanded].sort().join(",") || fold !== [...state.folded].sort().join(",")) { // a block or a chain opened or closed: redraw
    state.expanded = new Set(open.split(",").filter(Boolean));
    state.folded = new Set(fold.split(",").filter(Boolean));
    draw();
    return;
  }
  if (want.focus !== state.focus) toggleFocus();
  const tech = pickTech(want.tech, state.tech);
  if (tech !== state.tech) {
    state.tech = tech;
    techOptions();
    clear();
  } else if (want.k === state.k && (want.sel || null) === selText() && want.cmp === state.cmp && want.t === state.t) return;
  state.cmp = want.cmp === null ? null : clampStep(want.cmp);
  setSels(want.sel);
  state.t = want.t;
  if (want.k !== null && want.k !== state.k) goto(clampStep(want.k));
  else draw();
});
$("reload").addEventListener("click", async () => {
  try {
    await getJSON("/api/reload", POST_JSON);
  } catch (e) {
    status(String(e.message || e), true);
    return;
  }
  clear();
  load();
});

legend();
sidePanels();
load();
