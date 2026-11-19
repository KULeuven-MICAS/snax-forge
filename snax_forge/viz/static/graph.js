// The DFG page's drawing of one step (api.py's ``drawing``, D185): the server decides every box,
// port, edge, text and outline; here each box is measured as drawn, ELK (vendor/, run in its
// worker) places the boxes and routes the edges, and the result is drawn: boxes as HTML in the
// page's style, edges, labels and outlines in one SVG layer under them, port names in one over
// them. A step's layout is kept with its view, so going back to a step does not lay it out again.
//
// Rows: a kernel's inputs in ELK's first layer, lifted back to one row after compaction; the
// front loops on the rows api.py gives (``row``). An outline's title goes where it crosses the
// fewest wires and boxes, along its top inside edge, then its bottom one.

import { h } from "./dom.js";

const SVG = "http://www.w3.org/2000/svg";
const LABEL_FONT = '600 11.5px ui-monospace, "SF Mono", Menlo, Consolas, monospace';
export const TASK_COLOURS = ["var(--task-0)", "var(--task-1)", "var(--task-2)", "var(--task-3)"];
const PAD = { group: 30, accelerator: 28, block: 26 }; // px over an outline's boxes for its title
const LIFT_GAP = 0.5;

let engine = null;
/** ELK in its worker, made on first use; ``ELK`` is vendor/elk-api.js's global. */
function elk() {
  if (!engine) engine = new globalThis.ELK({ workerUrl: "vendor/elk-worker.min.js" });
  return engine;
}

const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, String(v));
  return e;
};

const taskColour = (d, task) => TASK_COLOURS[Math.max(0, d.tasks.indexOf(task)) % TASK_COLOURS.length];

// -- boxes ------------------------------------------------------------------------------------

const KIND_CLASS = { tasklet: "c-tasklet", unit: "c-unit", link: "c-link", slicer: "c-packer", packer: "c-packer",
  copy: "c-copy", mux: "c-mux", demux: "c-demux", block: "c-block" };

/** Port names shown on a side: a unit's or tasklet's inputs when it has several, a mux's or
demux's own side when it has several. */
function shownNames(n, side) {
  const list = side === "in" ? n.ins : n.outs;
  if (!list || list.length < 2) return false;
  if (n.kind === "mux") return side === "in";
  if (n.kind === "demux") return side === "out";
  return side === "in" && (n.kind === "unit" || n.kind === "tasklet");
}

function chipsEl(d, n) {
  if (!n.chips || !n.chips.length) return null;
  return h("div", { class: "c-chips" }, n.chips.map(([node, task]) => (task
    ? h("span", { class: "c-chip", style: { "--t": taskColour(d, task) }, "data-sel": node, title: `${node} (${task})` }, `${node} · ${task}`)
    : h("span", { class: "c-chip map", "data-sel": node, title: `${node}: the linked nest's loop, drawn on this chain` }, node))));
}

/** One box as HTML (``data-id`` its drawing id, ``data-sel`` what a click selects). */
export function boxEl(d, n) {
  if (n.kind === "container") {
    return h("div", { class: `c-pill${n.transient ? " transient" : ""}${n.scalar ? " scalar" : ""}`, "data-id": n.id, "data-box": n.id, "data-sel": n.sel },
      h("b", {}, n.text), n.dtype ? h("span", { class: "c-dtype" }, n.dtype) : null,
      n.badge ? h("span", { class: "c-badge" }, n.badge) : null,
      n.share ? h("span", { class: "c-share" }, n.share) : null);
  }
  if (n.kind === "loop") {
    return h("div", { class: `c-map c-loop loop-${n.loop === "untagged" ? "none" : n.loop || "none"}`, "data-id": n.id, "data-sel": n.sel },
      h("div", { class: "c-bar" }, h("div", {}, n.text), n.sub ? h("div", { class: "c-loop-sub" }, n.sub) : null, chipsEl(d, n)));
  }
  const shaped = n.fold || n.kind === "mux" || n.kind === "demux";
  const cls = ["c-node", KIND_CLASS[n.kind] || "c-tasklet", n.fold ? "c-fold" : null,
    shownNames(n, "in") ? "has-pnames" : null, shownNames(n, "out") ? "has-onames" : null].filter(Boolean).join(" ");
  return h("div", { class: cls, "data-id": n.id, "data-sel": n.sel, "data-block": n.block || null,
    title: n.kind === "block" ? "double-click to expand" : null },
  shaped ? h("i", { class: "c-shape" }) : null,
  h("div", { class: "c-text" }, n.text), n.sub ? h("div", { class: "c-unit-sub" }, n.sub) : null, chipsEl(d, n));
}

// -- ELK's graph ------------------------------------------------------------------------------

const ctx = document.createElement("canvas").getContext("2d");
function textWidth(t) {
  ctx.font = LABEL_FONT;
  return Math.ceil(ctx.measureText(t).width);
}

/** ELK's graph: each box at its measured size, its ports in order (inputs on top, outputs at the
bottom; a box without named ports a port of its own per edge, so no two edges meet at one
point), the outlines as compound nodes, the edges with their labels. ``lanes``: a loop's data in
the order its ports keep on both sides, so a datum leaves a loop on the side it entered (else ELK
orders a loop's ports freely). Also returns each own port's box, side and datum. */
function elkGraph(d, sizes, lanes = new Map(), forced = false) {
  const byOutline = new Map([[null, []], ...d.outlines.map((o) => [o.id, []])]);
  const extra = new Map(); // a box -> its own ports
  const own = new Map(); // an own port -> {node, side, data}
  const addPort = (node, id, side, data) => {
    if (!extra.has(node)) extra.set(node, []);
    if (!extra.get(node).some((p) => p.id === id)) extra.get(node).push({ id, side, data });
    own.set(id, { node, side, data });
  };
  const nodes = new Map(d.nodes.map((n) => [n.id, n]));
  const fixed = fixedPorts(d);
  const named = (n) => Boolean((n.ins && n.ins.length) || (n.outs && n.outs.length));
  const ends = d.edges.map((e, i) => {
    const a = nodes.get(e.from), b = nodes.get(e.to);
    const src = e.from_port ? `${e.from}.${e.from_port}` : named(a) ? `${e.from}.__out` : `${e.from}.__o${i}`;
    const dst = e.to_port ? `${e.to}.${e.to_port}` : named(b) ? `${e.to}.__in` : `${e.to}.__i${i}`;
    if (!e.from_port) addPort(e.from, src, "SOUTH", e.data || null);
    if (!e.to_port) addPort(e.to, dst, "NORTH", e.data || null);
    return [src, dst];
  });
  for (const [node, list] of extra) { // a loop's ports by its lanes, the same order on both sides
    const order = lanes.get(node);
    if (!order) continue;
    const at = (p) => (order.includes(p.data) ? order.indexOf(p.data) : order.length);
    list.sort((p, q) => at(p) - at(q));
  }
  for (const n of d.nodes) {
    const [w, ht] = sizes.get(n.id);
    const mine = extra.get(n.id) || [];
    const ins = [...(n.ins || []).map((p) => `${n.id}.${p}`), ...mine.filter((p) => p.side === "NORTH").map((p) => p.id)];
    const outs = [...(n.outs || []).map((p) => `${n.id}.${p}`), ...mine.filter((p) => p.side === "SOUTH").map((p) => p.id)];
    // FIXED_ORDER counts clockwise: left to right on top, right to left at the bottom
    const ports = [
      ...ins.map((id, i) => ({ id, width: 7, height: 7, layoutOptions: { "elk.port.side": "NORTH", "elk.port.index": String(i) } })),
      ...outs.map((id, i) => ({ id, width: 7, height: 7, layoutOptions: { "elk.port.side": "SOUTH", "elk.port.index": String(outs.length - 1 - i) } })),
    ];
    const lo = { "elk.portConstraints": fixed.has(n.id) || lanes.has(n.id) ? "FIXED_ORDER" : "FIXED_SIDE", "elk.portAlignment.default": "DISTRIBUTED" };
    if (n.layer === "first") lo["elk.layered.layering.layerConstraint"] = "FIRST";
    if (n.row !== undefined && n.row !== null) lo["elk.layered.layering.layerChoiceConstraint"] = String(n.row);
    byOutline.get(n.parent || null).push({ id: n.id, width: w, height: ht, ports, layoutOptions: lo });
  }
  const made = new Map(d.outlines.map((o) => [o.id, {
    id: o.id, children: byOutline.get(o.id),
    layoutOptions: { "elk.padding": `[top=${PAD[o.kind] || 28},left=14,bottom=14,right=14]` } }]));
  for (const o of d.outlines) if (o.parent && made.has(o.parent)) made.get(o.parent).children.push(made.get(o.id));
  // the top level in the drawing's order, an outline where its first box is
  const outline = new Map(d.outlines.map((o) => [o.id, o]));
  const rootOf = (id) => {
    let o = outline.get(id);
    while (o.parent && outline.has(o.parent)) o = outline.get(o.parent);
    return o.id;
  };
  const top = [];
  for (const n of d.nodes) {
    const item = n.parent ? made.get(rootOf(n.parent)) : byOutline.get(null).find((c) => c.id === n.id);
    if (!top.includes(item)) top.push(item);
  }
  for (const o of d.outlines) if (!o.parent && !top.includes(made.get(o.id))) top.push(made.get(o.id)); // an empty one
  const edges = d.edges.map((e, i) => ({
    id: `e${i}`, sources: [ends[i][0]], targets: [ends[i][1]],
    labels: e.label ? [{ text: e.label, width: Math.ceil(textWidth(e.label) * (e.transpose ? 1.45 : 1)) + 6, height: e.transpose ? 20 : 15 }] : [],
  }));
  const graph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered", "elk.direction": "DOWN", "elk.hierarchyHandling": "INCLUDE_CHILDREN",
      "elk.edgeRouting": "ORTHOGONAL", "elk.layered.spacing.nodeNodeBetweenLayers": "34",
      "elk.layered.spacing.edgeNodeBetweenLayers": "14", "elk.spacing.nodeNode": "26",
      "elk.spacing.edgeEdge": "12", "elk.spacing.edgeNode": "14", "elk.spacing.edgeLabel": "3",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      "elk.layered.compaction.postCompaction.strategy": "EDGE_LENGTH",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF", "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
      "elk.layered.edgeLabels.sideSelection": "ALWAYS_UP", "elk.edgeLabels.placement": "CENTER",
      "elk.padding": "[top=20,left=20,bottom=20,right=20]",
    },
    children: top, edges,
  };
  if (forced) graph.layoutOptions["elk.layered.crossingMinimization.forceNodeModelOrder"] = "true";
  return { graph, own };
}

/** Each loop's lanes after layout ``L``: a loop chain (loops joined by their hops) carries its
data side by side, so every loop of a chain orders its ports by one order, the data's left to
right where they first enter the chain (from a container, a copy or a unit). A chain whose
loops already keep it gets none. The leaves below then take their inputs in that order, their
ports being free (``fixedPorts``). */
function chainLanes(d, L, own) {
  const loops = new Set(d.nodes.filter((n) => n.kind === "loop").map((n) => n.id));
  const chain = new Map([...loops].map((id) => [id, id])); // union-find over the hops
  const root = (id) => (chain.get(id) === id ? id : root(chain.get(id)));
  for (const e of d.edges) if (loops.has(e.from) && loops.has(e.to)) chain.set(root(e.from), root(e.to));
  const entry = new Map(); // a chain -> datum -> x where it enters
  d.edges.forEach((e, i) => {
    if (!loops.has(e.to) || loops.has(e.from) || !e.data || !L.edges[i] || !L.edges[i].pts.length) return;
    const c = root(e.to);
    if (!entry.has(c)) entry.set(c, new Map());
    const x = L.edges[i].pts[0].x;
    const m = entry.get(c);
    if (!m.has(e.data) || x < m.get(e.data)) m.set(e.data, x);
  });
  const sides = new Map(); // a loop -> {NORTH, SOUTH}: its data left to right in L
  for (const [id, p] of own) {
    if (!p.data || !loops.has(p.node)) continue;
    const at = L.boxes.get(p.node) && L.boxes.get(p.node).ports.get(id);
    if (!at) continue;
    if (!sides.has(p.node)) sides.set(p.node, { NORTH: [], SOUTH: [] });
    sides.get(p.node)[p.side].push([at.x, p.data]);
  }
  const order = (list) => [...new Set([...list].sort((a, b) => a[0] - b[0]).map((x) => x[1]))];
  const lanes = new Map();
  const changes = new Set();
  for (const [node, sd] of sides) {
    const m = entry.get(root(node)) || new Map();
    const mine = [...new Set([...order(sd.NORTH), ...order(sd.SOUTH)])];
    if (mine.length < 2) continue;
    const at = (x) => (m.has(x) ? m.get(x) : Infinity);
    const want = [...mine].sort((a, b) => at(a) - at(b) || mine.indexOf(a) - mine.indexOf(b));
    lanes.set(node, want);
    const keeps = (side) => order(side).join("|") === want.filter((x) => order(side).includes(x)).join("|");
    if (!keeps(sd.NORTH) || !keeps(sd.SOUTH)) changes.add(root(node));
  }
  for (const node of [...lanes.keys()]) if (!changes.has(root(node))) lanes.delete(node);
  return lanes;
}

/** A box whose ports keep the server's order: a temporal mux or demux (its ports grouped by
task) and a unit a mux feeds (its inputs under the mux's outputs). Any other box's ports are
free, so ELK puts an input where its edge comes from and two edges into one box do not cross. */
function fixedPorts(d) {
  const fed = new Set(d.nodes.filter((n) => n.kind === "mux" && n.width_of).map((n) => n.width_of));
  return new Set(d.nodes.filter((n) => n.kind === "mux" || n.kind === "demux" || fed.has(n.id)).map((n) => n.id));
}

function collectEdges(root) {
  const out = [];
  const walk = (n) => {
    for (const e of n.edges || []) out.push(e);
    for (const c of n.children || []) walk(c);
  };
  walk(root);
  return out;
}

/** Every box's and outline's place on the canvas (ELK gives a child's relative to its parent),
its ports' centres; every edge's points and label. */
function absolute(d, root) {
  const pos = new Map([["root", [0, 0]]]);
  const boxes = new Map();
  const walk = (n, ox, oy) => {
    for (const c of n.children || []) {
      const x = ox + c.x, y = oy + c.y;
      pos.set(c.id, [x, y]);
      boxes.set(c.id, { x, y, w: c.width, h: c.height,
        ports: new Map((c.ports || []).map((p) => [p.id, { x: x + p.x + p.width / 2, y: y + p.y + p.height / 2 }])) });
      walk(c, x, y);
    }
  };
  walk(root, 0, 0);
  const edges = collectEdges(root).map((e) => {
    const [ox, oy] = pos.get(e.container || "root") || [0, 0];
    const sec = (e.sections || [])[0];
    const pts = sec ? [sec.startPoint, ...(sec.bendPoints || []), sec.endPoint].map((p) => ({ x: ox + p.x, y: oy + p.y })) : [];
    const l = (e.labels || [])[0];
    return { i: Number(e.id.slice(1)), src: e.sources[0], dst: e.targets[0], pts,
      label: l && l.x !== undefined ? { x: ox + l.x + 3, y: oy + l.y + 11.5 } : null };
  }).sort((a, b) => a.i - b.i);
  return { boxes, edges, width: root.width, height: root.height };
}

/** The kernel's inputs in one row on top: compaction moves a source towards what it feeds, so
each is put back on the top row and its edges start there. */
function liftInputs(d, L) {
  const first = d.nodes.filter((n) => n.layer === "first" && !n.parent).map((n) => n.id);
  if (!first.length) return;
  const top = Math.min(...first.map((id) => L.boxes.get(id).y));
  for (const id of first) {
    const b = L.boxes.get(id);
    const dy = b.y - top;
    if (dy < LIFT_GAP) continue;
    b.y -= dy;
    for (const p of b.ports.values()) p.y -= dy;
    for (const e of L.edges) {
      if (!e.src.startsWith(`${id}.`) || !e.pts.length) continue;
      e.pts[0] = { x: e.pts[0].x, y: e.pts[0].y - dy };
    }
  }
}

// a drawing's text -> its layout, the last KEEP: a step whose drawing did not change (after an
// edit elsewhere, a Reload, a step seen before) is not laid out again
const layouts = new Map();
const KEEP = 400;

/** Drawing ``d`` laid out: each box measured in ``probe`` (a hidden element in the page, so the
page's fonts apply), then ELK. */
export async function layout(d, probe, seed = null) {
  const key = JSON.stringify(d);
  if (layouts.has(key)) return layouts.get(key);
  const L = await layOut(seeded(d, seed), probe, Boolean(seed) && seeded(d, seed) !== d);
  layouts.set(key, L);
  if (layouts.size > KEEP) layouts.delete(layouts.keys().next().value);
  return L;
}

/** Drawing ``d`` laid out again from scratch, ELK free to choose every order (clean graph);
the layout kept for it from now on. */
export async function relayout(d, probe) {
  const L = await layOut(d, probe);
  layouts.set(JSON.stringify(d), L);
  return L;
}

const SEED_SHARE = 0.6; // of a step's boxes the step before must hold to keep its order

/** Drawing ``d`` with its boxes in the left-to-right order ``seed`` (the layout of a step shown
before) had them, a new box beside the boxes it connects to; ``d`` itself when the two share
too few boxes. Laid out with that order forced, a step keeps its neighbour's arrangement and
only what changed moves (the order a step is laid out in is the first one it is seen from). */
function seeded(d, seed) {
  if (!seed) return d;
  if (seeded.memo && seeded.memo[0] === d && seeded.memo[1] === seed) return seeded.memo[2];
  const x = new Map();
  for (const n of d.nodes) {
    const b = seed.boxes.get(n.id);
    if (b) x.set(n.id, b.x + b.w / 2);
  }
  let out = d;
  if (x.size >= SEED_SHARE * d.nodes.length) {
    for (let pass = 0; pass < 4; pass++) { // a new box: the mean of its placed neighbours
      for (const n of d.nodes) {
        if (x.has(n.id)) continue;
        const near = d.edges.flatMap((e) => (e.from === n.id ? [e.to] : e.to === n.id ? [e.from] : [])).filter((id) => x.has(id));
        if (near.length) x.set(n.id, near.reduce((a, id) => a + x.get(id), 0) / near.length);
      }
    }
    let last = -Infinity; // still none: just after the box before it in the drawing's order
    for (const n of d.nodes) {
      if (!x.has(n.id)) x.set(n.id, last + 1e-3);
      last = x.get(n.id);
    }
    const at = new Map(d.nodes.map((n, i) => [n.id, i]));
    out = { ...d, nodes: [...d.nodes].sort((a, b) => x.get(a.id) - x.get(b.id) || at.get(a.id) - at.get(b.id)) };
  }
  seeded.memo = [d, seed, out];
  return out;
}

async function layOut(d, probe, forced = false) {
  probe.replaceChildren(...d.nodes.map((n) => boxEl(d, n)));
  const sizes = new Map([...probe.children].map((el) => [el.dataset.id, [Math.ceil(el.offsetWidth), Math.ceil(el.offsetHeight)]]));
  probe.replaceChildren();
  // a mux as wide as the unit it feeds, so its outputs drop straight into the unit's inputs
  for (const n of d.nodes) {
    if (n.width_of && sizes.has(n.width_of)) sizes.set(n.id, [Math.max(sizes.get(n.id)[0], sizes.get(n.width_of)[0]), sizes.get(n.id)[1]]);
  }
  const t0 = performance.now();
  let { graph, own } = elkGraph(d, sizes, new Map(), forced);
  let L = absolute(d, await elk().layout(graph));
  // a datum that crosses another inside a loop chain reads as if it went where the other goes:
  // lay out again with each such chain's ports in one order, the order its data enter it
  const lanes = chainLanes(d, L, own);
  if (lanes.size) {
    ({ graph, own } = elkGraph(d, sizes, lanes, forced));
    L = absolute(d, await elk().layout(graph));
  }
  L.ms = performance.now() - t0;
  liftInputs(d, L);
  return L;
}

// -- drawing ----------------------------------------------------------------------------------

/** Points with no repeats and no middle point on a straight run. */
function cleanPts(pts) {
  const out = [];
  for (const p of pts) {
    const q = out[out.length - 1];
    if (q && Math.abs(q.x - p.x) < 0.5 && Math.abs(q.y - p.y) < 0.5) continue;
    out.push(p);
  }
  return out.filter((p, i) => {
    if (i === 0 || i === out.length - 1) return true;
    const a = out[i - 1], c = out[i + 1];
    return !((Math.abs(a.x - p.x) < 0.5 && Math.abs(c.x - p.x) < 0.5) || (Math.abs(a.y - p.y) < 0.5 && Math.abs(c.y - p.y) < 0.5));
  });
}

/** A right-angled path with rounded corners. */
function rounded(pts, r = 9) {
  if (!pts.length) return "";
  let path = `M${pts[0].x},${pts[0].y}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const p = pts[i - 1], q = pts[i], n = pts[i + 1];
    const d1 = Math.hypot(q.x - p.x, q.y - p.y), d2 = Math.hypot(n.x - q.x, n.y - q.y);
    const rr = Math.min(r, d1 / 2, d2 / 2);
    const a = { x: q.x + ((p.x - q.x) * rr) / d1, y: q.y + ((p.y - q.y) * rr) / d1 };
    const b = { x: q.x + ((n.x - q.x) * rr) / d2, y: q.y + ((n.y - q.y) * rr) / d2 };
    path += ` L${a.x},${a.y} Q${q.x},${q.y} ${b.x},${b.y}`;
  }
  const last = pts[pts.length - 1];
  return `${path} L${last.x},${last.y}`;
}

const edgeColour = (d, e) => (e.task && d.tasks.length && e.style !== "wire" ? taskColour(d, e.task)
  : e.style === "link" || e.style === "local" ? "var(--busy)" : "var(--edge)");

let uid = 0;

/** Drawing ``d`` with layout ``L`` into ``canvas`` (emptied first): the outlines and edges, the
boxes, the port names. Returns the boxes by id. */
export function render(canvas, d, L) {
  const k = ++uid;
  canvas.replaceChildren();
  canvas.classList.add("v3");
  canvas.style.width = `${L.width}px`;
  canvas.style.height = `${L.height}px`;
  const svg = svgEl("svg", { class: "dfg-edges", width: L.width, height: L.height });
  const defs = svgEl("defs");
  const colours = [];
  const marker = (colour) => {
    let i = colours.indexOf(colour);
    if (i < 0) {
      i = colours.push(colour) - 1;
      for (const [id, fill] of [[`m${k}-${i}`, colour], [`m${k}-${i}-sel`, "var(--hl)"]]) {
        const m = svgEl("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7,
          orient: "auto-start-reverse", markerUnits: "userSpaceOnUse" });
        const p = svgEl("path", { d: "M0,0 L10,5 L0,10 z" });
        p.style.fill = fill;
        m.append(p);
        defs.append(m);
      }
    }
    return `m${k}-${i}`;
  };
  svg.append(defs);
  // outlines first, under everything
  const outlines = d.outlines.map((o) => {
    const b = L.boxes.get(o.id);
    if (!b) return null;
    const cls = { group: "group-box", accelerator: "acc-box", block: "block-box" }[o.kind] || "group-box";
    const g = svgEl("g", { class: cls, "data-sel": o.sel, "data-group": o.kind === "group" ? o.sel : null, "data-block": o.block || null });
    const rect = { x: b.x, y: b.y, width: b.w, height: b.h, rx: o.kind === "block" ? 10 : 6 };
    const t = svgEl("text", { class: "v3-title" });
    t.textContent = o.text;
    g.append(svgEl("rect", { ...rect, class: "outline-hit" }), svgEl("rect", rect), t);
    if (o.kind === "block") {
      const tip = svgEl("title");
      tip.textContent = "double-click to collapse";
      g.append(tip);
    }
    svg.append(g);
    return { o, b, t };
  }).filter(Boolean);
  const edgeLayer = svgEl("g");
  const labels = svgEl("g");
  const drawn = [];
  d.edges.forEach((e, i) => {
    const le = L.edges[i];
    if (!le || !le.pts.length) return;
    const pts = cleanPts(le.pts);
    const path = rounded(pts);
    const colour = edgeColour(d, e);
    const hit = svgEl("path", { d: path, class: "edge-hit", "data-sel": e.id });
    const tip = svgEl("title");
    tip.textContent = [e.sub, e.task ? `task ${e.task}` : null].filter(Boolean).join(" · ") || e.id;
    hit.append(tip);
    const p = svgEl("path", { d: path, class: `edge v3 ${e.style}`, "data-sel": e.id });
    p.style.setProperty("--c", colour);
    if (e.style !== "empty") {
      p._marker = marker(colour);
      p.setAttribute("marker-end", `url(#${p._marker})`);
    }
    edgeLayer.append(hit, p);
    if (e.label && le.label) {
      const t = svgEl("text", { x: le.label.x, y: le.label.y + (e.transpose ? 4.5 : 0), class: `edge-label v3-label ${e.style}${e.transpose ? " transpose" : ""}`, "data-sel": e.id });
      if (e.task && d.tasks.length) t.style.fill = colour;
      t.textContent = e.label;
      labels.append(t);
    }
    drawn.push(pts);
  });
  svg.append(edgeLayer, labels);
  canvas.append(svg);
  const els = new Map();
  for (const n of d.nodes) {
    const b = L.boxes.get(n.id);
    const el = boxEl(d, n);
    el.classList.add("v3-box");
    Object.assign(el.style, { left: `${b.x}px`, top: `${b.y}px`, width: `${b.w}px`, height: `${b.h}px` });
    canvas.append(el);
    els.set(n.id, el);
  }
  // port names, over the boxes
  const top = svgEl("svg", { class: "dfg-edges v3-top", width: L.width, height: L.height });
  for (const n of d.nodes) {
    const b = L.boxes.get(n.id);
    for (const side of ["in", "out"]) {
      if (!shownNames(n, side)) continue;
      for (const name of side === "in" ? n.ins : n.outs) {
        const p = b.ports.get(`${n.id}.${name}`);
        if (!p) continue;
        const task = (n.port_tasks || {})[name];
        const t = svgEl("text", { class: "v3-pname", x: p.x, y: p.y + (side === "in" ? 13 : -7) });
        if (task) t.style.fill = taskColour(d, task);
        t.textContent = task ? name.split("·")[0] : name; // a mux's port: its task by colour
        top.append(t);
      }
    }
  }
  canvas.append(top);
  // the titles, once the text is in the page and measurable
  const boxes = d.nodes.map((n) => L.boxes.get(n.id)).map((b) => [b.x, b.y, b.x + b.w, b.y + b.h]);
  const segs = drawn.flatMap((p) => p.slice(1).map((q, i) => [Math.min(p[i].x, q.x), Math.min(p[i].y, q.y), Math.max(p[i].x, q.x), Math.max(p[i].y, q.y)]));
  requestAnimationFrame(() => outlines.forEach((ol) => placeTitle(ol, segs, boxes)));
  return els;
}

/** An outline's title where it crosses the fewest wires and boxes: along its top inside edge,
then its bottom one (half a crossing dearer), leftmost among equals. */
function placeTitle(ol, segs, boxes) {
  const r = ol.b;
  const tw = (ol.t.getComputedTextLength ? ol.t.getComputedTextLength() : 0) + 6;
  const th = 16;
  const hits = (x0, y0) => {
    const x1 = x0 + tw, y1 = y0 + th;
    let n = 0;
    for (const s of segs) if (s[0] <= x1 && s[2] >= x0 && s[1] <= y1 && s[3] >= y0) n++;
    for (const b of boxes) if (b[0] <= x1 && b[2] >= x0 && b[1] <= y1 && b[3] >= y0) n += 2;
    return n;
  };
  let best = null;
  for (const [y0, cost] of [[r.y + 4, 0], [r.y + r.h - th - 3, 0.5]]) {
    for (let x0 = r.x + 8; x0 + tw <= r.x + r.w - 6; x0 += 4) {
      const score = hits(x0, y0) + cost + (x0 - r.x) * 1e-4;
      if (!best || score < best.score) best = { score, x0, y0 };
    }
  }
  if (!best) best = { x0: r.x + 8, y0: r.y + 4 };
  ol.t.setAttribute("x", best.x0 + 2);
  ol.t.setAttribute("y", best.y0 + 12.5);
}

// -- pan and zoom -----------------------------------------------------------------------------

const MIN_SCALE = 0.1;
const MAX_SCALE = 3;

/** A panel's view of its drawing: ``world`` (the canvas) moved and scaled inside ``vp``. Drag the
empty canvas or scroll to pan, Ctrl (⌘) and scroll or pinch to zoom about the pointer. Below 42 %
the edge labels and port names go, below 30 % the chips, dtypes and badges (level of detail). */
export class Viewport {
  constructor(vp, world, onChange = () => {}) {
    Object.assign(this, { vp, world, onChange, s: 1, tx: 0, ty: 0, L: null });
    vp.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = vp.getBoundingClientRect();
      if (e.ctrlKey || e.metaKey) this.zoomAt(Math.exp(-e.deltaY * 0.0025), e.clientX - r.left, e.clientY - r.top);
      else this.set(this.s, this.tx - e.deltaX, this.ty - e.deltaY);
    }, { passive: false });
    let drag = null;
    vp.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || e.target.closest("[data-sel], select, button")) return;
      drag = { x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty, moved: false, id: e.pointerId };
    });
    vp.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.abs(dx) + Math.abs(dy) < 4) return;
      if (!drag.moved) {
        vp.setPointerCapture(drag.id);
        vp.classList.add("panning");
      }
      drag.moved = true;
      this.set(this.s, drag.tx + dx, drag.ty + dy);
    });
    const end = () => {
      if (drag && drag.moved) { // the click that ends a pan selects nothing
        vp.addEventListener("click", (ev) => ev.stopPropagation(), { capture: true, once: true });
        setTimeout(() => vp.classList.remove("panning"), 0);
      }
      drag = null;
    };
    vp.addEventListener("pointerup", end);
    vp.addEventListener("pointercancel", end);
  }

  set(s, tx, ty) {
    Object.assign(this, { s: Math.min(MAX_SCALE, Math.max(MIN_SCALE, s)), tx, ty });
    this.world.style.transform = `translate(${this.tx}px, ${this.ty}px) scale(${this.s})`;
    this.world.classList.toggle("lod-1", this.s < 0.42);
    this.world.classList.toggle("lod-2", this.s < 0.3);
    this.onChange(this);
  }

  zoomAt(f, cx, cy) {
    const s = Math.min(MAX_SCALE, Math.max(MIN_SCALE, this.s * f));
    this.set(s, cx - ((cx - this.tx) * s) / this.s, cy - ((cy - this.ty) * s) / this.s);
  }

  /** Zoom by ``f`` about the middle of the panel. */
  zoom(f) {
    const r = this.vp.getBoundingClientRect();
    this.zoomAt(f, r.width / 2, r.height / 2);
  }

  /** The whole drawing in the panel, at 100 % at most, centred across and from the top. */
  fit() {
    const r = this.vp.getBoundingClientRect();
    if (!this.L || !r.width) return;
    const s = Math.max(MIN_SCALE, Math.min(1, (r.width - 24) / this.L.width, (r.height - 24) / this.L.height));
    this.set(s, (r.width - this.L.width * s) / 2, 12);
  }

  /** Layout ``L`` shown where ``prev`` (a view of the step before) had its boxes: across, the
  boxes both steps hold keep their middle on screen; down, the topmost of them stays, so the
  inputs stay put and only what changed moves. Fit when they hold none. */
  show(L, prev) {
    this.L = L;
    const common = prev && prev.L ? [...L.boxes.keys()].filter((id) => prev.L.boxes.has(id) && !id.startsWith("grp:")
      && !id.startsWith("acc:") && !id.startsWith("block:")) : [];
    if (!common.length) {
      this.fit();
      return;
    }
    const at = (M) => {
      const c = common.map((id) => M.boxes.get(id));
      return [c.reduce((a, b) => a + b.x + b.w / 2, 0) / c.length, Math.min(...c.map((b) => b.y))];
    };
    const [ox, oy] = at(prev.L), [nx, ny] = at(L);
    this.set(prev.s, prev.tx + (ox - nx) * prev.s, prev.ty + (oy - ny) * prev.s);
  }
}
