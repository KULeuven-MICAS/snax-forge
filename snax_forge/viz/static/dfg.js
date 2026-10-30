// DFG viewer (D55, VIEW2): one step of a recipe at a time, two with Compare.
//
// The server works out everything shown (viz/dfg/api.py): the rows top to
// bottom in execution order, each top-level node's tree with its compact text,
// one edge per memlet, the Details of every item and the Estimate's figures.
// Here the boxes are plain HTML laid out by CSS and the edges one SVG layer
// per panel, measured from the placed boxes and redrawn when a panel changes
// size; nothing is computed but where to draw (D57). Text is never in the SVG
// but a group's name.
//
// The URL hash holds the state, as in the run viewer: #step=<k>&sel=<id>&cmp=<j>.
// A selection is a name: everything of that name is outlined (a link node and
// its container share one) and Details shows each.

import { h } from "./dom.js";

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

const state = { steps: [], k: 0, cmp: null, sel: null };
const views = new Map(); // step -> its view, until Reload
const compares = new Map(); // "j/k" -> the compare answer

function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  const num = (v) => (v === null || v === "" || Number.isNaN(Number(v)) ? null : Number(v));
  return { k: num(q.get("step")), sel: q.get("sel") || null, cmp: num(q.get("cmp")) };
}

function writeHash() {
  const q = new URLSearchParams();
  q.set("step", String(state.k));
  if (state.sel) q.set("sel", state.sel);
  if (state.cmp !== null) q.set("cmp", String(state.cmp));
  history.replaceState(null, "", `#${q}`);
}

const clampStep = (k) => Math.min(Math.max(k, 0), state.steps.length - 1);

async function view(k) {
  if (!views.has(k)) views.set(k, await getJSON(`/api/step/${k}`));
  return views.get(k);
}

async function compare(j, k) {
  const key = `${j}/${k}`;
  if (!compares.has(key)) compares.set(key, await getJSON(`/api/compare/${j}/${k}`));
  return compares.get(key);
}

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

function stepSelect(current, onChange) {
  return h("select", { "aria-label": "step", onchange: (ev) => onChange(Number(ev.target.value)) },
    state.steps.map((s, i) => h("option", { value: i, selected: i === current }, s.label)));
}

/** One panel: a step's drawing; with ``pick``, its own step dropdown (Compare). */
function panel(v, k, pick) {
  const head = h("div", { class: "c-panel-head" },
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
  el._draw = () => {
    drawEdges(canvas, svg, v.edges, k);
    drawAccelerators(canvas, svg, v.accelerators || []);
    outline();
  };
  new ResizeObserver(() => requestAnimationFrame(el._draw)).observe(canvas);
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
    const x1 = ra.left + ra.width / 2 - c.left;
    const x2 = rb.left + rb.width / 2 - c.left;
    let y1 = ra.bottom - c.top;
    let y2 = rb.top - c.top;
    // An edge to or from a connector inside a top-level node stops at that node's box.
    const top = e.stop ? canvas.querySelector(`[data-top="${CSS.escape(e.stop)}"]`) : null;
    if (top) {
      const rt = top.getBoundingClientRect();
      if (e.dir === "read") y2 = rt.top - c.top;
      else y1 = rt.bottom - c.top;
    }
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

/** An outline with its name on top: a group's at the left, an accelerator's at the right. */
function outlineBox(r, pad, cls, name) {
  const g = svgEl("g", { class: cls, "data-sel": name, ...(cls === "group-box" ? { "data-group": name } : {}) });
  g.append(svgEl("rect", { x: r.left - pad, y: r.top - pad - 14, width: r.right - r.left + 2 * pad,
    height: r.bottom - r.top + 2 * pad + 14, rx: 6 }));
  const right = cls === "acc-box";
  const t = svgEl("text", { x: right ? r.right + pad - 6 : r.left - pad + 6, y: r.top - pad - 3,
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
      left: Math.min(...rects.map((x) => x.left)), top: Math.min(...rects.map((x) => x.top)) - 18,
      right: Math.max(...rects.map((x) => x.right)), bottom: Math.max(...rects.map((x) => x.bottom)),
    };
    out.push(outlineBox(all, 16, "acc-box", acc.text));
  }
  svg.append(...out);
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
  state.sel = name;
  writeHash();
  outline();
  details();
}

const KINDS = ["node", "container", "group", "accelerator", "edge"];

function valueText(v) {
  if (v === null || v === undefined) return "–";
  if (Array.isArray(v)) return v.map((x) => (typeof x === "object" ? JSON.stringify(x) : String(x))).join(", ") || "–";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** Details: every item of the selected name, a labelled list; with Compare, before → after. */
async function details() {
  const box = $("details");
  if (!state.sel) {
    box.replaceChildren(h("p", { class: "muted" }, "Click a box, a container, an edge or a group outline; Esc clears."));
    return;
  }
  const b = await view(state.k);
  const a = state.cmp !== null ? await view(state.cmp) : null;
  const changed = a ? (await compare(state.cmp, state.k)).changed : {};
  const out = [];
  for (const kind of KINDS) {
    const key = `${kind}:${state.sel}`;
    const before = a && a.details ? a.details[key] : undefined;
    const after = b.details ? b.details[key] : undefined;
    if (before === undefined && after === undefined) continue;
    const fields = [...new Set([...Object.keys(before || {}), ...Object.keys(after || {})])].filter((f) => f !== "kind");
    const moved = new Set(changed[key] || []);
    const rows = fields.map((f) => {
      const one = valueText((after || before)[f]);
      const text = a && moved.has(f)
        ? `${valueText(before ? before[f] : undefined)} → ${valueText(after ? after[f] : undefined)}`
        : one;
      return [h("dt", { class: moved.has(f) ? "changed" : null }, f), h("dd", { class: moved.has(f) ? "changed mono" : "mono" }, text)];
    });
    const where = a ? (before === undefined ? " (added)" : after === undefined ? " (removed)" : "") : "";
    out.push(h("div", { class: "c-details" }, h("h3", {}, `${kind} ${state.sel}${where}`), h("dl", { class: "facts" }, rows)));
  }
  box.replaceChildren(...(out.length ? out : [h("p", { class: "muted" }, `${state.sel}: not in this step`)]));
}

// -- the Estimate ------------------------------------------------------------------

const pair = (a, b) => (a === null ? String(b) : String(a) === String(b) ? String(b) : `${a} → ${b}`);

function hoverGroup(name, on) {
  for (const el of document.querySelectorAll(`#graphs [data-group="${CSS.escape(name)}"]`)) el.classList.toggle("hover", on);
}

async function estimate() {
  const box = $("estimate");
  const b = await view(state.k);
  const a = state.cmp !== null ? await view(state.cmp) : null;
  if (b.estimate_error || !b.summary) {
    box.replaceChildren(h("p", { class: "load-error" }, b.estimate_error || b.error || "no estimate"));
    return;
  }
  const s = b.summary;
  const p = a && a.summary ? a.summary : null;
  const was = (f) => (p ? f(p) : null);
  const groupWas = (name, field) => {
    if (!p) return null;
    const g = p.groups.find((x) => x.name === name);
    return g ? g[field] : "–";
  };
  const rows = s.groups.map((g) => h("div", {
    class: "c-est-group", onmouseenter: () => hoverGroup(g.name, true), onmouseleave: () => hoverGroup(g.name, false),
  },
  h("b", {}, g.name), " ",
  h("span", { class: "mono" }, pair(groupWas(g.name, "formula"), g.formula)),
  h("small", {}, ` COUNTER ${pair(groupWas(g.name, "counter"), g.counter)} · channels ${pair(groupWas(g.name, "channels"), g.channels)}`)));
  const l1 = s.l1;
  const l1Text = l1.size
    ? `${pair(was((x) => x.l1.footprint_text), l1.footprint_text)} of ${l1.size_text} (${pair(was((x) => x.l1.share_text), l1.share_text)})`
    : pair(was((x) => x.l1.footprint_text), l1.footprint_text);
  box.replaceChildren(...[
    h("p", { class: "c-est-head" },
      h("b", {}, pair(was((x) => x.accelerator_cycles), s.accelerator_cycles)), " cycles · ",
      h("b", {}, pair(was((x) => x.not_mapped), s.not_mapped)), " not mapped"),
    ...rows,
    h("p", { class: "c-est-line" }, "L1 ", l1Text),
    l1.size ? h("div", { class: "c-bar-track" }, h("i", { style: { width: l1.share_css } })) : null,
    h("p", { class: "c-est-line" }, `reads ${pair(was((x) => x.reads), s.reads)} · writes ${pair(was((x) => x.writes), s.writes)} words`),
    s.splits.length ? h("p", { class: "c-est-line" }, "lanes ", s.splits.map((x) => `${x.map} ${x.text}`).join("; ")) : null,
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
    let panels;
    if (state.cmp === null) {
      panels = [panel(await view(state.k), state.k, null)];
    } else {
      panels = [
        panel(await view(state.cmp), state.cmp, (j) => { state.cmp = j; draw(); }),
        panel(await view(state.k), state.k, (k) => { state.k = k; draw(); }),
      ];
    }
    $("graphs").replaceChildren(...panels);
    $("graphs").classList.toggle("two", panels.length === 2);
    requestAnimationFrame(() => panels.forEach((p) => p._draw && p._draw()));
    await Promise.all([details(), estimate()]);
    const broken = state.steps.filter((s) => s.error).length;
    status(broken ? `${broken} of ${state.steps.length} files did not load` : "", broken > 0);
  } catch (e) {
    status(String(e.message || e), true);
  }
}

async function load() {
  status("Loading…");
  try {
    state.steps = await getJSON("/api/steps");
  } catch (e) {
    status(String(e.message || e), true);
    return;
  }
  $("step").replaceChildren(...state.steps.map((s, i) => h("option", { value: i }, s.label)));
  const want = readHash();
  state.k = clampStep(want.k === null ? 0 : want.k);
  state.cmp = want.cmp === null ? null : clampStep(want.cmp);
  state.sel = want.sel;
  draw();
}

// -- controls ----------------------------------------------------------------------

$("step").addEventListener("change", (ev) => { state.k = Number(ev.target.value); draw(); });
$("prev").addEventListener("click", () => { state.k = clampStep(state.k - 1); draw(); });
$("next").addEventListener("click", () => { state.k = clampStep(state.k + 1); draw(); });
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
document.addEventListener("keydown", (ev) => {
  if (ev.target.closest("select, input")) return;
  if (ev.key === "Escape") select(null);
  else if (ev.key === "ArrowLeft" && state.k > 0) { state.k -= 1; draw(); }
  else if (ev.key === "ArrowRight" && state.k < state.steps.length - 1) { state.k += 1; draw(); }
});
window.addEventListener("hashchange", () => {
  const want = readHash();
  if (want.k === state.k && want.sel === state.sel && want.cmp === state.cmp) return;
  state.k = clampStep(want.k === null ? 0 : want.k);
  state.cmp = want.cmp === null ? null : clampStep(want.cmp);
  state.sel = want.sel;
  draw();
});
$("reload").addEventListener("click", async () => {
  try {
    await getJSON("/api/reload", POST_JSON);
  } catch (e) {
    status(String(e.message || e), true);
    return;
  }
  views.clear();
  compares.clear();
  load();
});

legend();
load();
