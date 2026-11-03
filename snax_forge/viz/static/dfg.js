// DFG viewer (D55, VIEW2): one step of a recipe at a time, two with Compare.
//
// The server works out everything shown (viz/dfg/api.py): the rows top to
// bottom in execution order, each top-level node's tree with its compact text,
// one edge per memlet, the Details of every item and the Estimate's figures.
// Here the boxes are plain HTML laid out by CSS and the edges one SVG layer
// per panel, measured from the placed boxes and redrawn when a panel changes
// size; nothing is computed but where to draw (D57). Text is never in the SVG
// but a group's name. The old step stays on screen until the new one's view
// arrives; then its boxes glide from where they were (glide), unless the
// browser asks for reduced motion.
//
// The URL hash holds the state, as in the run viewer: #step=<k>&sel=<id>&cmp=<j>&t=<transform>&tech=<tech>
// (step and cmp, view mode: the steps' names, dot4_w8/6_link from several folders, and an old
// hash's bare stem or number still opens its step, /api/find; edit mode: numbers; t, edit mode: that transform's form open for the selection; tech, the technology the cost
// estimate is on, one of /api/mode's techs, passed to every step and compare request). A selection is a name:
// everything of that name is outlined (a link node and its container share one) and Details
// shows each. In edit mode (edit.js) the steps are the session's, the shown one its cursor.

import { dragHandle, h, keep, stored } from "./dom.js";
import * as edit from "./edit.js";

const $ = (id) => document.getElementById(id);

// Every POST is JSON: the server refuses another Content-Type (415, D55)
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

const state = { mode: "view", steps: [], k: 0, cmp: null, sel: null, t: null, session: null, platform: null, tech: null, techs: [] };
const views = new Map(); // step -> its view, until Reload
const compares = new Map(); // "j/k" -> the compare answer

const num = (v) => (v === null || v === "" || Number.isNaN(Number(v)) ? null : Number(v));

/** The hash as written; ``k`` and ``cmp`` are still its text until ``resolve``. */
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  return { k: q.get("step") || null, sel: q.get("sel") || null, cmp: q.get("cmp") || null, t: q.get("t") || null, tech: q.get("tech") || null };
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
  if (state.sel) q.set("sel", state.sel);
  if (state.cmp !== null) q.set("cmp", ref(state.cmp));
  if (state.t && state.sel) q.set("t", state.t);
  if (state.tech) q.set("tech", state.tech);
  history.replaceState(null, "", `#${q}`);
}

const clampStep = (k) => Math.min(Math.max(k, 0), state.steps.length - 1);

const techQuery = () => (state.tech ? `?tech=${encodeURIComponent(state.tech)}` : "");

async function view(k) {
  if (!views.has(k)) views.set(k, await getJSON(`/api/step/${k}${techQuery()}`));
  return views.get(k);
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
    h("li", {}, h("i", { class: "key c-link" }), "link"),
    h("li", {}, h("i", { class: "key c-pill" }), "container"),
    h("li", {}, h("i", { class: "key c-pill transient" }), "transient"),
  );
}

// -- the compact drawing ------------------------------------------------------------

function ports(n, side) {
  const list = side === "in" ? n.inputs : n.outputs;
  if (!list.length) return null;
  return h("div", { class: `c-ports ${side}` },
    list.map((p) => h("i", { class: "c-port", "data-port": `${n.id}.${p.connector}`, title: `${p.connector}: ${p.data}` })));
}

function nodeEl(n) {
  if (n.kind === "map") {
    return h("div", { class: `c-map loop-${n.loop_kind || "none"}`, "data-sel": n.id },
      h("div", { class: "c-bar" }, n.text),
      h("div", { class: "c-body" }, n.body.map(nodeEl)));
  }
  const cls = { tasklet: "c-tasklet", accelerated: "c-unit", link: "c-link" }[n.kind] || "c-tasklet";
  return h("div", { class: `c-node ${cls}`, "data-sel": n.id },
    ports(n, "in"),
    h("div", { class: "c-text" }, n.text),
    n.body && n.body.length ? h("div", { class: "c-body" }, n.body.map(nodeEl)) : null,
    ports(n, "out"));
}

function topEl(n) {
  const el = nodeEl(n);
  el.setAttribute("data-top", n.id);
  return el;
}

function pill(v, key) {
  const b = v.boxes[key];
  const c = v.containers[b.container];
  return h("div", { class: `c-pill${c.transient ? " transient" : ""}${b.again ? " again" : ""}`, "data-box": key, "data-sel": c.name },
    h("b", {}, c.text), c.badge ? h("span", { class: "c-badge" }, c.badge) : null);
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

/** One panel: a step's drawing; with ``pick``, its own step dropdown (Compare). */
function panel(v, k, pick) {
  const head = h("div", { class: "c-panel-head" },
    v.folder ? h("span", { class: "muted" }, `${v.folder} · `) : null,
    pick ? stepSelect(k, pick) : h("b", {}, v.label),
    h("span", { class: "muted" }, v.graph ? ` ${v.graph}` : ""));
  if (v.error) return h("section", { class: "c-panel broken" }, head, h("p", { class: "load-error" }, v.error));
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("class", "dfg-edges");
  const rows = h("div", { class: "c-rows" }, v.rows.map((r) =>
    r.kind === "containers"
      ? h("div", { class: "c-row" }, r.boxes.map((key) => pill(v, key)))
      : h("div", { class: "c-row" }, topEl(r.node))));
  const canvas = h("div", { class: "c-canvas" }, rows, svg);
  const el = h("section", { class: "c-panel" }, head, h("div", { class: "c-scroll" }, canvas));
  el._view = v;
  el._draw = () => {
    if (el.classList.contains("settling")) return; // the boxes are gliding: drawn once they settle
    drawEdges(canvas, svg, v.edges, k);
    drawAccelerators(canvas, svg, v.accelerators || []);
    outline();
  };
  new ResizeObserver(() => requestAnimationFrame(el._draw)).observe(canvas);
  if (state.mode === "edit") edit.decorate(v, el);
  return el;
}

// -- edges ------------------------------------------------------------------------

const SVG = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs) => {
  const e = document.createElementNS(SVG, tag);
  for (const [key, val] of Object.entries(attrs)) e.setAttribute(key, String(val));
  return e;
};

function anchor(canvas, key) {
  return canvas.querySelector(`[data-box="${CSS.escape(key)}"]`) || canvas.querySelector(`[data-port="${CSS.escape(key)}"]`);
}

function drawEdges(canvas, svg, edges, k) {
  const c = canvas.getBoundingClientRect();
  svg.setAttribute("width", canvas.scrollWidth);
  svg.setAttribute("height", canvas.scrollHeight);
  const marker = `arrow-${k}`;
  const defs = svgEl("defs", {});
  for (const [id, cls] of [[marker, "arrow"], [`${marker}-sel`, "arrow-sel"]]) {
    const m = svgEl("marker", {
      id, viewBox: "0 0 10 10", refX: 10, refY: 5, markerWidth: 8, markerHeight: 8,
      markerUnits: "userSpaceOnUse", orient: "auto-start-reverse",
    });
    m.append(svgEl("path", { d: "M0,0 L10,5 L0,10 z", class: cls }));
    defs.append(m);
  }
  const out = [];
  for (const e of edges) {
    const a = anchor(canvas, e.from);
    const b = anchor(canvas, e.to);
    if (!a || !b) continue;
    const ra = a.getBoundingClientRect();
    const rb = b.getBoundingClientRect();
    // From the bottom of one end to the top of the other, a connector inside a node included
    // (the boxes it passes are drawn over it)
    const x1 = ra.left + ra.width / 2 - c.left;
    const x2 = rb.left + rb.width / 2 - c.left;
    const y1 = ra.bottom - c.top;
    const y2 = rb.top - c.top;
    const bend = Math.max(18, Math.abs(y2 - y1) * 0.45);
    const d = `M${x1},${y1} C${x1},${y1 + bend} ${x2},${y2 - bend} ${x2},${y2 - 2}`;
    out.push(svgEl("path", { d, class: "edge-hit", "data-sel": e.id }));
    const p = svgEl("path", { d, class: `edge ${e.dir}`, "data-sel": e.id, "marker-end": `url(#${marker})` });
    p._marker = marker;
    out.push(p);
  }
  svg.replaceChildren(defs, ...out);
}

// -- groups and accelerators (D148): a thin dashed outline each, the name on top ---------

function unionRect(canvas, rects) {
  const c = canvas.getBoundingClientRect();
  const r = rects.filter(Boolean);
  if (!r.length) return null;
  return {
    left: Math.min(...r.map((x) => x.left)) - c.left, top: Math.min(...r.map((x) => x.top)) - c.top,
    right: Math.max(...r.map((x) => x.right)) - c.left, bottom: Math.max(...r.map((x) => x.bottom)) - c.top,
  };
}

const LABEL = 17; // px above an outline's content for its name (style.css: 0.85rem)

/** An outline with its name on top: a group's at the left, an accelerator's at the right. */
function outlineBox(r, pad, cls, name) {
  const g = svgEl("g", { class: cls, "data-sel": name, ...(cls === "group-box" ? { "data-group": name } : {}) });
  g.append(svgEl("rect", { x: r.left - pad, y: r.top - pad - LABEL, width: r.right - r.left + 2 * pad,
    height: r.bottom - r.top + 2 * pad + LABEL, rx: 6 }));
  const right = cls === "acc-box";
  const t = svgEl("text", { x: right ? r.right + pad - 6 : r.left - pad + 6, y: r.top - pad - 4,
    "text-anchor": right ? "end" : "start" });
  t.textContent = name;
  g.append(t);
  return g;
}

function drawAccelerators(canvas, svg, accelerators) {
  const out = [];
  for (const acc of accelerators) {
    const rects = [];
    for (const grp of acc.groups) {
      const tops = grp.tops.map((id) => canvas.querySelector(`[data-top="${CSS.escape(id)}"]`));
      const r = unionRect(canvas, tops.map((t) => t && t.getBoundingClientRect()));
      if (!r) continue;
      rects.push(r);
      out.push(outlineBox(r, 8, "group-box", grp.text));
    }
    // An accelerator that is one group of its name (dot 2 to 6) is drawn once, as the group.
    if (!rects.length || (acc.groups.length === 1 && acc.groups[0].name === acc.name)) continue;
    const all = {
      left: Math.min(...rects.map((x) => x.left)), top: Math.min(...rects.map((x) => x.top)) - LABEL - 4,
      right: Math.max(...rects.map((x) => x.right)), bottom: Math.max(...rects.map((x) => x.bottom)),
    };
    out.push(outlineBox(all, 16, "acc-box", acc.text));
  }
  svg.append(...out);
}

// -- the move between steps: the boxes glide from their old places --------------------------

const HTML = "http://www.w3.org/1999/xhtml";
const GLIDE_MS = 250;
let drawn = null; // the views last drawn, one per panel

const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const boxes = (root) => [...root.querySelectorAll(".c-canvas [data-sel]")].filter((el) => el.namespaceURI === HTML);
const boxKey = (el) => (el.dataset.box ? `box:${el.dataset.box}` : `node:${el.dataset.sel}`);

/** Where every box is on screen, per panel: its key -> its rect. */
function places() {
  return [...$("graphs").children].map((p) => new Map(boxes(p).map((el) => [boxKey(el), el.getBoundingClientRect()])));
}

/** A new step's panel: each box that was on screen moves from there to its place, a box inside
another by what is left of its own move; a new one fades in; the step's added and changed
boxes (decorate's ``diffed``) arrive from a highlight; the edges are drawn when all settle. */
function glide(panel, old) {
  const moved = new Map(); // box -> its whole move, which the boxes inside it share
  const runs = [];
  for (const el of boxes(panel)) {
    const up = el.parentElement.closest(".c-canvas [data-sel]");
    const carried = (up && moved.get(up)) || [0, 0];
    const was = old.get(boxKey(el));
    if (el.classList.contains("diffed")) el.classList.add("arrive");
    if (!was) {
      moved.set(el, carried);
      runs.push(el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: GLIDE_MS * 1.5, easing: "ease-out" }));
      continue;
    }
    const now = el.getBoundingClientRect();
    const whole = [was.left - now.left, was.top - now.top];
    moved.set(el, whole);
    const dx = whole[0] - carried[0];
    const dy = whole[1] - carried[1];
    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) continue;
    runs.push(el.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }],
      { duration: GLIDE_MS, easing: "ease-out" }));
  }
  if (!runs.length) return;
  panel.classList.add("settling");
  Promise.all(runs.map((a) => a.finished.catch(() => null))).then(() => {
    panel.classList.remove("settling");
    panel._draw();
  });
}

// -- selection --------------------------------------------------------------------

function outline() {
  for (const el of document.querySelectorAll("#graphs .sel")) {
    el.classList.remove("sel");
    if (el._marker) el.setAttribute("marker-end", `url(#${el._marker})`);
  }
  if (!state.sel) return;
  for (const el of document.querySelectorAll(`#graphs [data-sel="${CSS.escape(state.sel)}"]`)) {
    el.classList.add("sel");
    if (el._marker) el.setAttribute("marker-end", `url(#${el._marker}-sel)`);
  }
}

function select(name) {
  if (name !== state.sel) state.t = null;
  state.sel = name;
  writeHash();
  outline();
  details();
  containers();
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

const KINDS = ["node", "container", "group", "accelerator", "edge"];

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

// -- the Containers (the left column, C§12) ----------------------------------------------

/** Every container that a symbol's shape uses, highlighted in the Containers and the graph
(focus or hover on the symbol's input, its chip or its text). */
function hlSymbol(name, on) {
  const v = views.get(state.k);
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

/** One symbol of a shape: in edit mode an input on its param (the same as its chip, the same
replay), else its name. */
function symbolEl(name) {
  const params = state.mode === "edit" && state.session ? state.session.params : null;
  if (params && Object.prototype.hasOwnProperty.call(params, name)) {
    const value = String(params[name]);
    return h("label", { class: "dfg-sym", "data-symbol": name, title: `param ${name}: a change replays every step`, ...hover(name) },
      h("input", { value, size: Math.max(2, value.length), "aria-label": `param ${name}`,
        onchange: (ev) => edit.setParam(name, ev.target) }));
  }
  return h("span", { class: "dfg-sym", "data-symbol": name, tabindex: 0, ...hover(name) }, name);
}

/** The Containers section: per container its name (a click selects it), dtype (read only,
D113), placement and bytes, its shape with each symbol and the sizes at the step's symbols. */
async function containers() {
  const box = $("containers");
  const v = await view(state.k);
  const list = v.containers_section;
  if (!list) {
    box.replaceChildren(h("p", { class: "muted" }, v.error || "no graph"));
    return;
  }
  box.replaceChildren(...list.map((c) => h("div", { class: `dfg-cont${state.sel === c.name ? " sel" : ""}`, "data-cont": c.name },
    h("div", { class: "dfg-cont-head" },
      h("button", { type: "button", class: "dfg-cont-name", onclick: () => select(c.name) }, c.name),
      h("small", {}, [c.dtype, c.placement, c.bytes_text].filter(Boolean).join(" · "))),
    h("div", { class: "dfg-cont-shape mono" },
      c.dims.length ? c.dims.map((d, i) => [i ? " × " : "", d.parts.map((p) => (p.symbol ? symbolEl(p.symbol) : p.text))]) : "scalar",
      c.sizes_text && c.sizes_text !== c.shape_text ? h("span", { class: "muted" }, ` = ${c.sizes_text}`) : null))));
}

// -- the Estimate ------------------------------------------------------------------

function hoverGroup(name, on) {
  for (const el of document.querySelectorAll(`#graphs [data-group="${CSS.escape(name)}"]`)) el.classList.toggle("hover", on);
}

const COST_COLUMNS = [["block", "block"], ["datapath", "datapath"], ["area_text", "area (µm²)"],
  ["energy_text", "energy (nJ)"], ["share_text", "share"]];

/** One accelerator's cost table (api.py's cost_tables, or compare_cost_tables' before → after
cells): a row per unit, buffer link and control unit, then the total. */
function costTable(t) {
  const num = (key) => key !== "block" && key !== "datapath";
  const tr = (r, cls) => h("tr", { class: [cls, r.differs ? "differs" : null, r.found === false ? "missing" : null].filter(Boolean).join(" ") || null },
    COST_COLUMNS.map(([key]) => h(key === "block" ? "th" : "td", { class: num(key) ? "num" : null, scope: key === "block" ? "row" : null }, r[key])));
  return h("table", { class: "c-cost-table" },
    h("caption", {}, t.accelerator),
    h("thead", {}, h("tr", {}, COST_COLUMNS.map(([key, label]) => h("th", { scope: "col", class: num(key) ? "num" : null }, label)))),
    h("tbody", {}, t.rows.map((r) => tr(r, null)), tr(t.total, "total")));
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
      tables.length ? h("p", { class: "c-est-line muted" }, `cost per block, ${cmp.columns.labels[0]} → ${cmp.columns.labels[1]}`) : null,
      ...tables.map(costTable)].filter(Boolean));
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
  h("small", {}, ` COUNTER ${g.counter} · channels ${g.channels}` + (g.area_text ? ` · ${g.area_text} · ${g.energy_text}` : ""))));
  const c = s.cost;
  const cost = c ? [
    h("p", { class: "c-est-head" }, h("b", {}, c.heading), ": ", h("b", {}, c.area_text), " · ", h("b", {}, c.energy_text), " per run"),
    h("p", { class: "c-est-line muted" }, c.caveat),
    ...(c.tables || []).map(costTable),
    h("p", { class: "c-est-line" }, "not characterized: ", c.missing.length ? c.missing.join(", ") : "none"),
  ] : [];
  const l1 = s.l1;
  const l1Text = l1.size ? `${l1.footprint_text} of ${l1.size_text} (${l1.share_text})` : l1.footprint_text;
  box.replaceChildren(...[
    h("p", { class: "c-est-head" }, h("b", {}, s.accelerator_cycles), " cycles · ", h("b", {}, s.not_mapped), " not mapped"),
    ...rows,
    h("p", { class: "c-est-line" }, "L1 ", l1Text),
    l1.size ? h("div", { class: "c-bar-track" }, h("i", { style: { width: l1.share_css } })) : null,
    h("p", { class: "c-est-line" }, `reads ${s.reads} · writes ${s.writes} words`),
    s.splits.length ? h("p", { class: "c-est-line" }, "lanes ", s.splits.map((x) => `${x.map} ${x.text}`).join("; ")) : null,
    ...cost,
  ].filter(Boolean));
}

// -- drawing a state ----------------------------------------------------------------

async function draw() {
  writeHash();
  $("step").value = String(state.k);
  $("compare").checked = state.cmp !== null;
  $("prev").disabled = state.k <= 0;
  $("next").disabled = state.k >= state.steps.length - 1;
  status("Loading…");
  try {
    const shown = await view(state.k);
    if (state.sel && !KINDS.some((kind) => shown.details && shown.details[`${kind}:${state.sel}`])) {
      state.sel = null; // kept across steps only where it exists
      state.t = null;
      writeHash();
    }
    let panels;
    if (state.cmp === null) {
      panels = [panel(shown, state.k, null)];
    } else {
      panels = [
        panel(await view(state.cmp), state.cmp, (j) => { state.cmp = j; draw(); }),
        panel(shown, state.k, goto),
      ];
    }
    const old = places();
    $("graphs").replaceChildren(...panels);
    $("graphs").classList.toggle("two", panels.length === 2);
    const now = panels.map((p) => p._view);
    if (!still() && drawn && old.length === panels.length) {
      panels.forEach((p, i) => p._view !== drawn[i] && glide(p, old[i]));
    }
    drawn = now;
    requestAnimationFrame(() => panels.forEach((p) => p._draw && p._draw()));
    await Promise.all([details(), estimate(), containers(), state.mode === "edit" ? edit.sidebar() : null]);
    const broken = state.steps.filter((s) => s.error).length;
    status(broken ? `${broken} of ${state.steps.length} files did not load` : "", broken > 0);
  } catch (e) {
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
      state.sel = want.sel;
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
  state.sel = want.sel;
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
// A click on an item selects it; on empty space it clears the selection, as Esc does.
$("graphs").addEventListener("click", (ev) => {
  if (ev.target.closest("select")) return;
  const t = ev.target.closest("[data-sel]");
  select(t ? t.getAttribute("data-sel") : null);
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
  const tech = pickTech(want.tech, state.tech);
  if (tech !== state.tech) {
    state.tech = tech;
    techOptions();
    clear();
  } else if (want.k === state.k && want.sel === state.sel && want.cmp === state.cmp && want.t === state.t) return;
  state.cmp = want.cmp === null ? null : clampStep(want.cmp);
  state.sel = want.sel;
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
