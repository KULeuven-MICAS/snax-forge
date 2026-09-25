// DFG viewer (VIS5, D76, D81): one panel per .snaxdfg file, side by side.
//
// The server works out what goes where (api.py): the rows top to bottom in
// execution order, each top-level node's tree, and one edge per memlet from a
// container box or a writer's connector to a connector or a container box.
// Here the boxes are plain HTML laid out by CSS (flexbox does the nesting),
// and the edges are one SVG layer over each panel: measured from the placed
// boxes and redrawn when the panel changes size. Text is never in the SVG.
//
// Hovering a node, a connector or a container highlights everything with
// the same name in every panel, so one node can be followed through the
// steps of a recipe (ids survive split_map and bind).

import { h } from "./dom.js";

const $ = (id) => document.getElementById(id);
const enc = encodeURIComponent;

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || `${url}: HTTP ${r.status}`);
  return body;
}

function status(msg, err = false) {
  $("status").textContent = msg || "";
  $("status").classList.toggle("error", err);
}

const LOOP_KINDS = [
  ["none", "map, not mapped yet"],
  ["tile", "tile"],
  ["temporal", "temporal"],
  ["spatial", "spatial"],
];

function legend() {
  $("legend").replaceChildren(
    h("li", {}, h("i", { class: "key data" }), "data container"),
    h("li", {}, h("i", { class: "key data transient" }), "transient data container"),
    ...LOOP_KINDS.map(([k, label]) => h("li", {}, h("i", { class: `key loop-${k}` }), label)),
    h("li", {}, h("i", { class: "key op-accelerated" }), "accelerated node"),
  );
}

// -- boxes ----------------------------------------------------------------------

const tip = (obj) => JSON.stringify(obj, null, 1);

function containerBox(view, key) {
  const b = view.boxes[key];
  const name = b.container;
  const c = view.containers[name];
  const shape = c.shape.map((s, i) => (c.sizes && String(s) !== String(c.sizes[i]) ? `${s} = ${c.sizes[i]}` : s));
  return h("div", {
    class: `box${c.transient ? " transient" : ""}${b.again ? " again" : ""}`, "data-box": key, "data-id": name,
    title: tip({ container: name, ...c, version: b.version, written_by: b.written_by, shown_again: b.again }),
  },
  h("span", { class: "box-kind" }, c.transient ? "transient data" : "data container"),
  h("b", {}, name),
  h("small", {}, `${c.dtype}[${shape.join(", ")}]`),
  b.written_by.length ? h("small", { class: "ver" }, `written by ${b.written_by.join(", ")}`) : null);
}

function port(n, p) {
  return h("div", { class: "port", "data-port": `${n.id}.${p.connector}`, "data-id": p.data, title: p.text },
    h("b", {}, p.connector), " ", h("span", {}, p.text));
}

function nodeEl(n) {
  if (n.kind === "map") {
    const kind = n.loop_kind || "none";
    return h("div", { class: `scope loop-${kind}`, "data-id": n.id, title: tip({ id: n.id, attrs: n.attrs }) },
      h("div", { class: "scope-head" },
        h("b", {}, n.id),
        h("span", { class: "mono" }, n.title),
        n.iterations === null ? null : h("span", { class: "count" }, `${n.iterations}×`),
        h("span", { class: "tag" }, n.loop_kind || "map")),
      h("div", { class: "scope-body" }, n.body.map(nodeEl)));
  }
  return h("div", { class: `op op-${n.kind}`, "data-id": n.id, title: tip({ id: n.id, kind: n.kind, attrs: n.attrs }) },
    n.inputs.length ? h("div", { class: "ports in" }, n.inputs.map((p) => port(n, p))) : null,
    h("div", { class: "op-head" },
      h("b", {}, n.id), h("span", { class: "tag" }, n.kind),
      n.kind === "tasklet" ? null : h("div", { class: "op-title" }, n.title),
      (n.heading || []).map((l) => h("div", { class: "op-title" }, l))),
    n.lines.length ? h("div", { class: "op-lines mono" }, n.lines.map((l) => h("div", {}, l))) : null,
    n.replaced ? h("div", { class: "op-replaced" }, `replaces ${n.replaced}`) : null,
    n.body && n.body.length ? h("div", { class: "scope-body" }, n.body.map(nodeEl)) : null,
    n.outputs.length ? h("div", { class: "ports out" }, n.outputs.map((p) => port(n, p))) : null);
}

/** A top-level node: the box an edge from or to a container stops at (D82). */
function topEl(n) {
  const el = nodeEl(n);
  el.setAttribute("data-top", n.id);
  return el;
}

function symbolsText(symbols) {
  const s = Object.entries(symbols);
  if (!s.length) return "no symbols";
  return s.map(([k, v]) => (v === null ? `${k} unbound` : `${k} = ${v}`)).join(", ");
}

function panel(view, index) {
  const head = h("div", { class: "panel-head" },
    h("h2", {}, view.name),
    h("p", { class: "note" }, view.graph ? `${view.graph} · ${symbolsText(view.symbols || {})}` : "not loaded"),
    h("p", { class: "path" }, view.path));
  if (view.error) {
    return h("section", { class: "dfg-panel broken" }, head, h("p", { class: "load-error" }, view.error));
  }
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "dfg-edges");
  const rows = h("div", { class: "dfg-rows" }, view.rows.map((r) =>
    r.kind === "containers"
      ? h("div", { class: "dfg-row" }, r.boxes.map((k) => containerBox(view, k)))
      : h("div", { class: "dfg-row" }, topEl(r.node))));
  const canvas = h("div", { class: "dfg-canvas" }, rows, svg);
  const el = h("section", { class: "dfg-panel" }, head, h("div", { class: "dfg-scroll" }, canvas));
  el._draw = () => drawEdges(canvas, svg, view.edges, index);
  new ResizeObserver(() => requestAnimationFrame(el._draw)).observe(canvas);
  return el;
}

// -- edges ------------------------------------------------------------------------

const SVG = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs) => {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
};

function anchor(canvas, key) {
  return canvas.querySelector(`[data-box="${CSS.escape(key)}"]`) || canvas.querySelector(`[data-port="${CSS.escape(key)}"]`);
}

function drawEdges(canvas, svg, edges, index) {
  const c = canvas.getBoundingClientRect();
  svg.setAttribute("width", canvas.scrollWidth);
  svg.setAttribute("height", canvas.scrollHeight);
  const marker = `arrow-${index}`;
  const defs = svgEl("defs", {});
  // Two markers, so a highlighted edge gets a highlighted head; fixed size, not scaled by the stroke.
  for (const [id, cls] of [[marker, "arrow"], [`${marker}-hl`, "arrow-hl"]]) {
    const m = svgEl("marker", {
      id, viewBox: "0 0 10 10", refX: 10, refY: 5, markerWidth: 9, markerHeight: 9,
      markerUnits: "userSpaceOnUse", orient: "auto-start-reverse",
    });
    m.append(svgEl("path", { d: "M0,0 L10,5 L0,10 z", class: cls }));
    defs.append(m);
  }
  const paths = [];
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
    // An edge to or from a connector inside a top-level node stops at that node's box,
    // straight above or below the connector, so it never runs over the text inside.
    const top = e.stop ? canvas.querySelector(`[data-top="${CSS.escape(e.stop)}"]`) : null;
    if (top) {
      const rt = top.getBoundingClientRect();
      if (e.dir === "read") y2 = rt.top - c.top;
      else y1 = rt.bottom - c.top;
    }
    const k = Math.max(18, Math.abs(y2 - y1) * 0.45);
    const ids = [e.data, e.from.split(/[.@]/)[0], e.to.split(/[.@]/)[0]].join(" "); // container, nodes
    const p = svgEl("path", {
      d: `M${x1},${y1} C${x1},${y1 + k} ${x2},${y2 - k} ${x2},${y2 - 2}`,
      class: `edge ${e.dir}`, "data-ids": ids, "marker-end": `url(#${marker})`,
    });
    p._marker = marker;
    paths.push(p);
  }
  svg.replaceChildren(defs, ...paths);
  if (hovered) highlight(hovered);
}

// -- hover ------------------------------------------------------------------------

let hovered = null;

function highlight(id) {
  for (const el of document.querySelectorAll(".hl")) {
    el.classList.remove("hl");
    if (el._marker) el.setAttribute("marker-end", `url(#${el._marker})`);
  }
  hovered = id;
  if (!id) return;
  const q = CSS.escape(id);
  for (const el of document.querySelectorAll(`[data-id="${q}"]`)) el.classList.add("hl");
  for (const el of document.querySelectorAll(`path[data-ids~="${q}"]`)) {
    el.classList.add("hl");
    el.setAttribute("marker-end", `url(#${el._marker}-hl)`);
  }
}

$("graphs").addEventListener("mouseover", (ev) => {
  const t = ev.target.closest("[data-id]");
  const id = t ? t.getAttribute("data-id") : null;
  if (id !== hovered) highlight(id);
});
$("graphs").addEventListener("mouseleave", () => highlight(null));

// -- loading ----------------------------------------------------------------------

async function load() {
  status("Loading…");
  try {
    const list = await getJSON("/api/graphs");
    const views = await Promise.all(list.map((e) => getJSON(`/api/graph/${enc(e.name)}`)));
    const panels = views.map(panel);
    $("graphs").replaceChildren(...panels);
    requestAnimationFrame(() => panels.forEach((p) => p._draw && p._draw()));
    const broken = views.filter((v) => v.error).length;
    status(broken ? `${broken} of ${views.length} files did not load` : "", broken > 0);
  } catch (e) {
    status(String(e.message || e), true);
  }
}

$("reload").addEventListener("click", async () => {
  try {
    await getJSON("/api/reload", { method: "POST" });
  } catch (e) {
    status(String(e.message || e), true);
    return;
  }
  load();
});

legend();
load();
