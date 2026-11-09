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

const state = { mode: "view", steps: [], k: 0, cmp: null, sel: null, sels: [], t: null, session: null, platform: null, tech: null, techs: [], expanded: new Set() };

// The selection (D72): ``sels`` every selected item in the order selected (shift-click adds or
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
  if (state.sel) q.set("sel", selText());
  if (state.cmp !== null) q.set("cmp", ref(state.cmp));
  if (state.t && state.sel) q.set("t", state.t);
  if (state.tech) q.set("tech", state.tech);
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
    h("li", {}, h("i", { class: "key c-pill" }), "container"),
    h("li", {}, h("i", { class: "key c-pill transient" }), "transient"),
    h("li", {}, h("i", { class: "key c-pill scalar" }), "scalar"),
  );
}

// -- the compact drawing ------------------------------------------------------------

function ports(n, side) {
  const list = (side === "in" ? n.inputs : n.outputs).filter((p) => !p.csr);
  if (!list.length) return null;
  return h("div", { class: `c-ports ${side}` },
    list.map((p) => h("i", { class: "c-port", "data-port": `${n.id}.${p.connector}`, title: `${p.connector}: ${p.data}` })));
}

function nodeEl(n) {
  if (n.kind === "block") { // a collapsed block (api.py's ``fold``): one box, double-click opens it
    return h("div", { class: "c-node c-block", "data-sel": n.id, "data-block": n.id, title: "double-click to expand" },
      ports(n, "in"), h("div", { class: "c-text" }, n.text), ports(n, "out"));
  }
  if (n.kind === "map") {
    return h("div", { class: `c-map loop-${n.loop_kind || "none"}`, "data-sel": n.id },
      h("div", { class: "c-bar" }, n.text),
      h("div", { class: "c-body" }, n.body.map(nodeEl)));
  }
  const cls = { tasklet: "c-tasklet", accelerated: "c-unit", link: "c-link" }[n.kind] || "c-tasklet";
  // a fold (api.py's ``fold``): an upside-down trapezoid, its fill a layer of its own under
  // the text (style.css), the output connectors inside its narrow bottom
  return h("div", { class: `c-node ${cls}${n.fold ? " c-fold" : ""}`, "data-sel": n.id },
    n.fold ? h("i", { class: "c-shape" }) : null,
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
  return h("div", { class: `c-pill${c.transient ? " transient" : ""}${c.scalar ? " scalar" : ""}${b.again ? " again" : ""}`, "data-box": key, "data-sel": c.name },
    h("b", {}, c.text), c.badge ? h("span", { class: "c-badge" }, c.badge) : null);
}

/** Rows top to bottom (api.py's ``_layout``): a row of containers, a top-level node, or the
independent chains' columns side by side, each its own rows (a join's node below them). */
function rowEls(v, rows) {
  return rows.map((r) => {
    if (r.kind === "columns") {
      return h("div", { class: "c-cols" }, r.columns.map((c) =>
        h("div", { class: "c-col", "data-column": c.id }, rowEls(v, c.rows))));
    }
    return r.kind === "containers"
      ? h("div", { class: "c-row" }, r.boxes.map((key) => pill(v, key)))
      : h("div", { class: "c-row" }, topEl(r.node));
  });
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
  const rows = h("div", { class: "c-rows" }, rowEls(v, v.rows));
  const canvas = h("div", { class: "c-canvas" }, rows, svg);
  const el = h("section", { class: "c-panel" }, head, h("div", { class: "c-scroll" }, canvas));
  el._view = v;
  el._draw = () => {
    if (el.classList.contains("settling")) return; // the boxes are gliding: drawn once they settle
    drawEdges(canvas, svg, v.edges, k);
    drawOutlines(canvas, svg, v);
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
    const p = svgEl("path", { d, class: `edge ${e.dir}${e.local ? " local" : ""}${e.seed ? " seed" : ""}`, "data-sel": e.id, "marker-end": `url(#${marker})` });
    p._marker = marker;
    out.push(p);
    if (e.slice) { // a collapsed block's port (api.py's fold): the part one firing takes, beside its edge by the container's box
      const sy = e.dir === "read" ? y1 + 14 : y2 - 8; // by the container's box, clear of the outlines' bands
      const t = svgEl("text", { x: curveX(x1, y1, x2, y2, bend, sy) + 8, y: sy, class: "edge-label at-start slice", "data-sel": e.data });
      t.textContent = e.slice;
      out.push(t);
    }
    if (e.label && e.transpose) { // a marked read (mark_transpose, api.py): Aᵀ beside its edge, just above the reader's top-level map, at the container's font size
      const top = (b.closest(".c-row") || b).getBoundingClientRect().top - c.top;
      const ty = Math.max(y1 + 14, top - 8);
      const t = svgEl("text", { x: curveX(x1, y1, x2, y2, bend, ty) + 8, y: ty, class: "edge-label at-start transpose", "data-sel": e.data });
      t.style.fontSize = getComputedStyle(a).fontSize;
      t.textContent = e.label;
      out.push(t);
    } else if (e.label) { // a fused nest's local value (its name, no box), a scalar's broadcast (alpha[0] · broadcast) or a fold's seed (q[m_t]), api.py
      // a local value's right of its edge, halfway down (clear of the line and its arrow); a
      // broadcast's at its start, beside the hexagon, so no map bar it passes hides it
      const [tx, ty] = e.local ? [Math.max(x1, x2) + 10, (y1 + y2) / 2 + 4] : [x1 + 6, y1 + 14];
      const t = svgEl("text", { x: tx, y: ty, class: `edge-label${e.local ? "" : " at-start"}`, "data-sel": e.data });
      t.textContent = e.label;
      out.push(t);
    }
  }
  svg.replaceChildren(defs, ...out);
}

/** The x of an edge's curve (drawEdges's cubic) at height ``y``, by bisection: its y only
grows from the start to the end. */
function curveX(x1, y1, x2, y2, bend, y) {
  const at = (t, p0, p1, p2, p3) => (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3;
  let lo = 0, hi = 1;
  for (let i = 0; i < 30; i++) {
    const mid = (lo + hi) / 2;
    if (at(mid, y1, y1 + bend, y2 - bend, y2 - 2) < y) lo = mid; else hi = mid;
  }
  return at(lo, x1, x1, x2, x2);
}

/** Each row of containers left to right by where its edges land: a box takes the place of the
leftmost connector it feeds (else of the leftmost that wrote it), ties and boxes without edges in
api.py's order, so an edge does not cross the row for want of an order (axpy's fused nest: y,
read by augassign's left connector, first). The connectors sit in node rows, which the order
does not move, so one pass settles it. Then each node's input connectors by where their edges come
from, so two edges into one node do not cross (mvt's tmp0_mult: y1, then the shared A). It runs
before the glide measures where the boxes are. */
function orderRows(panel) {
  const canvas = panel.querySelector(".c-canvas");
  if (!canvas || !panel._view) return;
  const edges = panel._view.edges;
  const xOf = (key) => {
    const a = anchor(canvas, key);
    if (!a) return Infinity;
    const r = a.getBoundingClientRect();
    return r.left + r.width / 2;
  };
  for (const row of canvas.querySelectorAll(".c-rows .c-row")) {
    const pills = [...row.children].filter((el) => el.dataset.box);
    if (pills.length < 2) continue;
    const at = pills.map((el, i) => {
      const key = el.dataset.box;
      const fed = edges.filter((e) => e.from === key).map((e) => xOf(e.to));
      const wrote = edges.filter((e) => e.to === key).map((e) => xOf(e.from));
      return { el, x: Math.min(...(fed.length ? fed : wrote.length ? wrote : [Infinity])), i };
    });
    at.sort((a, b) => (a.x === b.x ? a.i - b.i : a.x - b.x));
    at.forEach(({ el }, k) => { el.style.order = String(k); });
  }
  // then each node's input connectors left to right by where their edges come from (the box
  // above, or the writer's connector), ties in the node's order: mvt's tmp0_mult takes y1 left of
  // the shared A, so their edges do not cross
  for (const row of canvas.querySelectorAll(".c-ports.in")) {
    const at = [...row.children].map((el, i) => {
      const from = edges.filter((e) => e.to === el.dataset.port).map((e) => xOf(e.from));
      return { el, x: from.length ? Math.min(...from) : Infinity, i };
    });
    if (at.length < 2) continue;
    at.sort((a, b) => (a.x === b.x ? a.i - b.i : a.x - b.x));
    at.forEach(({ el }, k) => { el.style.order = String(k); });
  }
}

// -- outlines: groups and accelerators (D148), open blocks (REUSE1) ------------------------------
// One pass for the three kinds, from the inside out: an outline holds the outlines whose boxes
// it holds (a group inside its accelerator, both inside an open block around the same boxes),
// each GAP px outside the ones it holds and its name in a band of its own (a group's on top at
// the left, an accelerator's on top at the right, a block's at the bottom left), so none
// crosses another or its name. The rows that hold an outline's first and last box get the
// margin it needs above and below, so it clears the containers' rows too.

function unionRect(canvas, rects) {
  const c = canvas.getBoundingClientRect();
  const r = rects.filter(Boolean);
  if (!r.length) return null;
  return {
    left: Math.min(...r.map((x) => x.left)) - c.left, top: Math.min(...r.map((x) => x.top)) - c.top,
    right: Math.max(...r.map((x) => x.right)) - c.left, bottom: Math.max(...r.map((x) => x.bottom)) - c.top,
  };
}

const LABEL = 17; // px of an outline's band for its name (style.css: 0.85rem)
const GAP = 8; // px between an outline and what it holds

/** Every outline to draw: ``{kind, name, text, tops, rank}``, ``rank`` ordering outlines
around the same boxes (a group inside its accelerator inside a block, a deeper block inside). */
function outlines(v) {
  const out = [];
  for (const acc of v.accelerators || []) {
    const groups = acc.groups.filter((g) => g.drawn !== false);
    for (const grp of groups) out.push({ kind: "group", name: grp.text, text: grp.text, tops: grp.tops, rank: 0 });
    // an accelerator that is one group of its name (dot 2 to 6) is drawn once, as the group
    if (!groups.length || (acc.groups.length === 1 && acc.groups[0].name === acc.name)) continue;
    out.push({ kind: "acc", name: acc.text, text: acc.text, tops: [...new Set(groups.flatMap((g) => g.tops))], rank: 1 });
  }
  const open = (v.blocks || []).filter((b) => !b.collapsed && !b.hidden && b.foldable);
  const same = new Map(); // blocks open around the same boxes: one outline, ``matvec › dot``
  for (const b of open) {
    const key = b.tops.join(",");
    if (!same.has(key)) same.set(key, []);
    same.get(key).push(b);
  }
  const deepest = Math.max(0, ...open.map((b) => b.depth));
  for (const bs of same.values()) {
    bs.sort((a, b) => a.depth - b.depth);
    const inner = bs[bs.length - 1];
    out.push({ kind: "block", name: inner.name, block: inner.name, tops: bs[0].tops, rank: 2 + deepest - bs[0].depth,
      text: `▾ ${bs.length > 1 ? bs.map((b) => b.name).join(" › ") : inner.text}`,
      tip: `${bs.map((b) => b.text).join("\n")}\ndouble-click to collapse` });
  }
  return out;
}

/** ``a`` holds ``b``: every box of ``b`` is one of ``a``'s, and ``a`` ranks above ``b`` when
they hold the same. */
function holds(a, b) {
  if (a === b || !b.tops.every((t) => a.tops.includes(t))) return false;
  return a.tops.length > b.tops.length || a.rank > b.rank;
}

function drawOutlines(canvas, svg, v) {
  const list = outlines(v);
  const top = (id) => canvas.querySelector(`[data-top="${CSS.escape(id)}"]`);
  for (const o of list) o.els = o.tops.map(top).filter(Boolean);
  const shown = list.filter((o) => o.els.length);
  const boxOf = new Map();
  const box = (o) => { // the outline's box, its name's band included, after those it holds
    if (boxOf.has(o)) return boxOf.get(o);
    const r = unionRect(canvas, o.els.map((el) => el.getBoundingClientRect()));
    for (const inner of shown) {
      if (!holds(o, inner)) continue;
      const b = box(inner);
      r.left = Math.min(r.left, b.left); r.right = Math.max(r.right, b.right);
      r.top = Math.min(r.top, b.top); r.bottom = Math.max(r.bottom, b.bottom);
    }
    const b = { left: r.left - GAP, right: r.right + GAP, top: r.top - GAP - (o.kind === "block" ? 0 : LABEL),
      bottom: r.bottom + GAP + (o.kind === "block" ? LABEL : 0) };
    boxOf.set(o, b);
    return b;
  };
  const out = [];
  const room = new Map(); // a row -> the room its outlines need above and below it
  for (const o of shown) {
    const b = box(o);
    const cls = { group: "group-box", acc: "acc-box", block: "block-box" }[o.kind];
    const g = svgEl("g", { class: cls, "data-sel": o.name, ...(o.kind === "group" ? { "data-group": o.name } : {}),
      ...(o.block ? { "data-block": o.block } : {}) });
    const rect = { x: b.left, y: b.top, width: b.right - b.left, height: b.bottom - b.top, rx: o.kind === "block" ? 10 : 6 };
    g.append(svgEl("rect", { ...rect, class: "outline-hit" }), svgEl("rect", rect));
    const t = o.kind === "acc"
      ? svgEl("text", { x: b.right - 6, y: b.top + LABEL - 5, "text-anchor": "end" })
      : svgEl("text", { x: b.left + 6, y: o.kind === "block" ? b.bottom - 5 : b.top + LABEL - 5 });
    t.textContent = o.text;
    g.append(t);
    if (o.tip) {
      const tip = svgEl("title", {});
      tip.textContent = o.tip;
      g.append(tip);
    }
    out.push(g);
    const rows = o.els.map((el) => el.closest(".c-row")).filter(Boolean);
    const c = canvas.getBoundingClientRect();
    const first = rows.reduce((x, y) => (y.getBoundingClientRect().top < x.getBoundingClientRect().top ? y : x));
    const last = rows.reduce((x, y) => (y.getBoundingClientRect().bottom > x.getBoundingClientRect().bottom ? y : x));
    const up = first.getBoundingClientRect().top - c.top - b.top;
    const down = b.bottom - (last.getBoundingClientRect().bottom - c.top);
    room.set(first, { ...(room.get(first) || { up: 0, down: 0 }), up: Math.max(up, (room.get(first) || { up: 0 }).up) });
    room.set(last, { ...(room.get(last) || { up: 0, down: 0 }), down: Math.max(down, (room.get(last) || { down: 0 }).down) });
  }
  svg.append(...out);
  // the room each row needs: a margin, set only when it changes (the panel's ResizeObserver
  // redraws once with it)
  for (const row of canvas.querySelectorAll(".c-row")) {
    const want = room.get(row) || { up: 0, down: 0 };
    const gap = parseFloat(getComputedStyle(row.parentElement).rowGap) || 0; // what the rows keep already
    const mt = `${Math.max(0, Math.ceil(want.up - gap + 22))}px`; // a pill's height clear of it
    const mb = `${Math.max(0, Math.ceil(want.down - gap + 22))}px`;
    if (row.style.marginTop !== mt) row.style.marginTop = mt;
    if (row.style.marginBottom !== mb) row.style.marginBottom = mb;
  }
}

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
    h("span", { class: "mono muted" }, ` ${b.recipe ? `${b.recipe}${Object.entries(b.params).map(([k, x]) => ` · ${k} = ${x}`).join("")}` : `${b.kernel} · no recipe`}`))));
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
another by what is left of its own move; a new one fades in; the edges are drawn when all
settle. */
function glide(panel, old) {
  const moved = new Map(); // box -> its whole move, which the boxes inside it share
  const runs = [];
  for (const el of boxes(panel)) {
    const up = el.parentElement.closest(".c-canvas [data-sel]");
    const carried = (up && moved.get(up)) || [0, 0];
    const was = old.get(boxKey(el));
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
  for (const name of state.sels) { // every selected item highlighted
    for (const el of document.querySelectorAll(`#graphs [data-sel="${CSS.escape(name)}"]`)) {
      el.classList.add("sel");
      if (el._marker) el.setAttribute("marker-end", `url(#${el._marker}-sel)`);
    }
  }
  // a selected unit's instance shared by other nodes (api.py's ``shared``, D165): those too
  for (const panelEl of document.querySelectorAll("#graphs .c-panel")) {
    const shared = (panelEl._view && panelEl._view.shared) || {};
    for (const name of state.sels) {
      for (const other of shared[name] || []) {
        for (const el of panelEl.querySelectorAll(`.c-node[data-sel="${CSS.escape(other)}"]`)) el.classList.add("sel");
      }
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

// -- the Containers (the left column, C§12) ----------------------------------------------

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

// -- the Bindings (the left column, C§12): each param with what it configures -----------------

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
  // the tree (D156): accelerator ▸ group ▸ entries, each level a drop-down, open by default;
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

// short labels, so the table fits the right panel at 1280 px (R3): the datapath under the block
// in one cell, the numbers right-aligned; the units in each heading's title
const COST_COLUMNS = [["block", "block · datapath", "the unit, buffer link, control unit or L1 accesses, its datapath under it"],
  ["area_text", "µm²", "area, µm² (3 significant figures)"],
  ["energy_text", "nJ", "energy per run, nJ (3 decimals)"], ["share_text", "share", "share of the accelerator's energy"]];

/** One accelerator's cost table (api.py's cost_tables, or compare_cost_tables' before → after
cells): a row per unit, buffer link and control unit, the L1 accesses, then the total. */
function costTable(t) {
  const num = (key) => key !== "block";
  // the L1 accesses' row is a placeholder (D154): muted, a short label, the full one as its title
  const l1 = (r) => r.kind === "l1" || String(r.block).startsWith("L1 accesses");
  const cell = (r, key) => {
    if (key !== "block") return r[key];
    const name = l1(r) ? [h("span", {}, "L1 accesses"), " ", h("span", { class: "c-cost-tag" }, "placeholder")] : r.block;
    return [h("div", {}, name), r.datapath ? h("div", { class: "c-cost-dp" }, r.datapath) : null];
  };
  const tr = (r, cls) => h("tr", { class: [cls, r.differs ? "differs" : null, r.found === false ? "missing" : null, l1(r) ? "placeholder" : null].filter(Boolean).join(" ") || null,
    title: l1(r) ? String(r.block) : null },
  COST_COLUMNS.map(([key]) => h(key === "block" ? "th" : "td", { class: num(key) ? "num" : null, scope: key === "block" ? "row" : null }, cell(r, key))));
  return h("div", { class: "c-cost-wrap" }, h("table", { class: "c-cost-table" },
    h("caption", {}, t.accelerator),
    h("thead", {}, h("tr", {}, COST_COLUMNS.map(([key, label, title]) => h("th", { scope: "col", class: num(key) ? "num" : null, title }, label)))),
    h("tbody", {}, t.rows.map((r) => tr(r, null)), tr(t.total, "total"))));
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
  // each figure with its unit on one line (``56 000 µm²`` never breaks); the API's text as is
  h("small", { title: "the expected COUNTER, cycles + fill: a guide, not a check (D155)" }, " ",
    [`expected COUNTER ${g.counter}`, `channels ${g.channels}`, ...(g.area_text ? [g.area_text, g.energy_text] : [])]
      .map((x, i) => [i ? " · " : "", h("span", { class: "nb" }, x)])),
  g.bank_note ? h("p", { class: "c-bank-note", title: "L1's single-port banks serve one word each per cycle: a bound on the streams' L1 time, not on the COUNTER (D155)" }, `bank bound: ${g.bank_note}`) : null));
  const c = s.cost;
  const cost = c ? [
    h("p", { class: "c-est-head" }, h("b", {}, c.heading), ": ", h("b", { class: "nb" }, c.area_text), " · ", h("b", { class: "nb" }, c.energy_text), " per run"),
    h("p", { class: "c-est-line muted" }, c.caveat),
    ...(c.tables || []).map(costTable),
    h("p", { class: "c-est-line" }, "not characterized: ", c.missing.length ? c.missing.join(", ") : "none"),
  ] : [];
  // edit mode: Save's design checks on this step (D156): the problems first, then the notes,
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
    s.streamers ? h("p", { id: "est-streamers", class: "c-est-line", title: "one streamer set per accelerator (D157); the streamers are not characterized yet (open item 110)" },
      s.streamers.text, s.streamers.accelerators.length > 1 ? ": " + s.streamers.accelerators.map((a) => `${a.name}${a.csrs_text ? " " + a.csrs_text : ""}`).join(", ") : "") : null,
    ...checks,
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
    const there = state.sels.filter((x) => KINDS.some((kind) => shown.details && shown.details[`${kind}:${x}`]));
    if (there.length !== state.sels.length) {
      setSels(there.join(",")); // kept across steps only where it exists
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
    panels.forEach(orderRows);
    $("graphs").classList.toggle("two", panels.length === 2);
    const now = panels.map((p) => p._view);
    if (!still() && drawn && old.length === panels.length) {
      panels.forEach((p, i) => p._view !== drawn[i] && glide(p, old[i]));
    }
    drawn = now;
    requestAnimationFrame(() => panels.forEach((p) => p._draw && p._draw()));
    await Promise.all([details(), estimate(), blocks(), containers(), bindings(), state.mode === "edit" ? edit.sidebar() : null]);
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
  if (t) toggleBlock(t.getAttribute("data-block"));
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
