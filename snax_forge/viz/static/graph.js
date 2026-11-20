// The DFG page's drawing of one step (api.py's ``drawing``, D185): the server decides every box,
// port, edge, text and outline; here each box is measured as drawn, ELK (vendor/, run in its
// worker) places the boxes and routes the edges, and the result is drawn: boxes as HTML in the
// page's style, edges, labels and outlines in one SVG layer under them, port names in one over
// them. A step's layout is kept with its view, so going back to a step does not lay it out again.
//
// Rows: every box on the row of its longest path from the kernel's inputs (ELK's partitions), the
// front loops on the rows api.py gives (``row``), a group's temporal muxes on one row and its
// demuxes on another (``rows``). An outline's title sits in its lower right corner.

import { h } from "./dom.js";

const SVG = "http://www.w3.org/2000/svg";
const LABEL_FONT = '600 11.5px ui-monospace, "SF Mono", Menlo, Consolas, monospace';
export const TASK_COLOURS = ["var(--task-0)", "var(--task-1)", "var(--task-2)", "var(--task-3)"];
// an outline's title: its lines (split after the name, ``group mvt_1`` over ``shared by mvt,
// mvt_1``) in its lower right corner, in a band of their own under what it holds
const TITLE_LINE = 15;
const titleLines = (text) => {
  const i = text.indexOf(" · ");
  return i < 0 ? [text] : [text.slice(0, i), text.slice(i + 3)];
};
const padBottom = (o) => 10 + TITLE_LINE * titleLines(o.text).length;

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
    return h("div", { class: `c-map c-loop loop-${n.loop === "untagged" ? "none" : n.loop || "none"}${n.folded ? " c-folded" : ""}`, "data-id": n.id, "data-sel": n.sel,
      "data-chain": n.folded ? n.sel : n.id, title: n.folded ? "a folded loop chain: double-click to unfold" : "double-click to fold its loop chain into one box" },
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
function elkGraph(d, sizes, lanes = new Map(), forced = false, inOrder = new Map()) {
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
  const row = rows(d);
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
    const ins = [...(inOrder.get(n.id) || n.ins || []).map((p) => `${n.id}.${p}`), ...mine.filter((p) => p.side === "NORTH").map((p) => p.id)];
    const outs = [...(n.outs || []).map((p) => `${n.id}.${p}`), ...mine.filter((p) => p.side === "SOUTH").map((p) => p.id)];
    // FIXED_ORDER counts clockwise: left to right on top, right to left at the bottom
    const ports = [
      ...ins.map((id, i) => ({ id, width: 7, height: 7, layoutOptions: { "elk.port.side": "NORTH", "elk.port.index": String(i) } })),
      ...outs.map((id, i) => ({ id, width: 7, height: 7, layoutOptions: { "elk.port.side": "SOUTH", "elk.port.index": String(outs.length - 1 - i) } })),
    ];
    const lo = { "elk.portConstraints": fixed.has(n.id) || lanes.has(n.id) ? "FIXED_ORDER" : "FIXED_SIDE", "elk.portAlignment.default": "DISTRIBUTED" };
    if (n.layer === "first") lo["elk.layered.layering.layerConstraint"] = "FIRST";
    lo["elk.partitioning.partition"] = String(row.get(n.id));
    byOutline.get(n.parent || null).push({ id: n.id, width: w, height: ht, ports, layoutOptions: lo });
  }
  const made = new Map(d.outlines.map((o) => [o.id, {
    id: o.id, children: byOutline.get(o.id),
    layoutOptions: { "elk.padding": `[top=14,left=14,bottom=${padBottom(o)},right=14]` } }]));
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
  // a group's muxes on one row, its demuxes on another: ELK's partitions hold only outside an
  // outline, so inside one each switch gets two edges that are never drawn, from what feeds the
  // group's lowest switch and into what that switch feeds, which leave it no other layer
  edges.push(...switchRows(d, row, ends).map(([src, dst], k) => ({ id: `z${k}`, sources: [src], targets: [dst] })));
  const graph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered", "elk.direction": "DOWN", "elk.hierarchyHandling": "INCLUDE_CHILDREN",
      "elk.edgeRouting": "ORTHOGONAL", "elk.layered.spacing.nodeNodeBetweenLayers": "34",
      "elk.layered.spacing.edgeNodeBetweenLayers": "14", "elk.spacing.nodeNode": "26",
      "elk.spacing.edgeEdge": "12", "elk.spacing.edgeNode": "14", "elk.spacing.edgeLabel": "3",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      "elk.partitioning.activate": "true", "elk.layered.compaction.postCompaction.strategy": "NONE",
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

/** Each temporal mux whose inputs cross on their way in after layout ``L``: its inputs in a new
order, each task's inputs still together and the tasks in the server's order, but within a task
by where their edges come from. A mux routes any input to any output, so the order of a task's
inputs is free (``a·mvt`` may sit right of ``b·mvt``); the edges then come in without crossing. */
function muxOrders(d, L) {
  const from = new Map(); // "box.port" -> the x its edge comes from
  d.edges.forEach((e, i) => {
    if (e.to_port && L.edges[i] && L.edges[i].pts.length) from.set(`${e.to}.${e.to_port}`, L.edges[i].pts[0].x);
  });
  const out = new Map();
  for (const n of d.nodes) {
    if (n.kind !== "mux" || !n.ins || n.ins.length < 2) continue;
    const tasks = n.port_tasks || {};
    const groups = [];
    for (const p of n.ins) {
      const last = groups[groups.length - 1];
      if (last && tasks[last[0]] === tasks[p]) last.push(p);
      else groups.push([p]);
    }
    const x = (p) => (from.has(`${n.id}.${p}`) ? from.get(`${n.id}.${p}`) : Infinity);
    const order = groups.flatMap((g) => [...g].sort((a, b) => x(a) - x(b) || g.indexOf(a) - g.indexOf(b)));
    if (order.join("|") !== n.ins.join("|")) out.set(n.id, order);
  }
  return out;
}

/** Each box's row (ELK's partition, which keeps rows in order): the longest path to it from the
inputs, no less than the row api.py gives a front loop (``row``); in each outline the temporal
muxes share one row, the lowest any of them needs, and the demuxes another, so a group's per-task
switches read as one stage. (ELK's layerChoiceConstraint is not read by elkjs's layering.) */
function rows(d) {
  const preds = new Map(d.nodes.map((n) => [n.id, []]));
  for (const e of d.edges) if (preds.has(e.to) && preds.has(e.from) && e.from !== e.to) preds.get(e.to).push(e.from);
  const least = new Map(d.nodes.filter((n) => n.row != null).map((n) => [n.id, n.row]));
  const groups = [];
  for (const kind of ["mux", "demux"]) {
    const by = new Map();
    for (const n of d.nodes.filter((x) => x.kind === kind)) {
      if (!by.has(n.parent || null)) by.set(n.parent || null, []);
      by.get(n.parent || null).push(n.id);
    }
    groups.push(...[...by.values()].filter((ids) => ids.length > 1));
  }
  let rank = new Map();
  for (let pass = 0; pass < 4; pass++) {
    rank = new Map();
    const of = (id) => {
      if (rank.has(id)) return rank.get(id);
      rank.set(id, 0); // a cycle (none expected) stops here
      const r = Math.max(least.get(id) || 0, ...preds.get(id).map((p) => of(p) + 1));
      rank.set(id, r);
      return r;
    };
    d.nodes.forEach((n) => of(n.id));
    let moved = false;
    for (const ids of groups) {
      const r = Math.max(...ids.map((id) => rank.get(id)));
      for (const id of ids) if ((least.get(id) || 0) < r) { least.set(id, r); moved = true; }
    }
    if (!moved) break;
  }
  return rank;
}

/** The edges (port to port) that put each temporal mux of a group on the row of the group's
lowest one (its anchor), and each demux on the row of the lowest demux: from the port feeding the
anchor into the switch's first input, from the switch's first output into the port the anchor
feeds. */
function switchRows(d, row, ends) {
  const out = [];
  for (const kind of ["mux", "demux"]) {
    const by = new Map();
    for (const n of d.nodes.filter((x) => x.kind === kind)) {
      if (!by.has(n.parent || null)) by.set(n.parent || null, []);
      by.get(n.parent || null).push(n);
    }
    for (const ns of by.values()) {
      if (ns.length < 2) continue;
      const anchor = ns.reduce((a, b) => (row.get(b.id) > row.get(a.id) ? b : a));
      const into = d.edges.findIndex((e) => e.to === anchor.id);
      const from = d.edges.findIndex((e) => e.from === anchor.id);
      for (const n of ns) {
        if (n === anchor) continue;
        const myIn = d.edges.findIndex((e) => e.to === n.id), myOut = d.edges.findIndex((e) => e.from === n.id);
        if (into >= 0 && myIn >= 0) out.push([ends[into][0], ends[myIn][1]]);
        if (from >= 0 && myOut >= 0) out.push([ends[myOut][0], ends[from][1]]);
      }
    }
  }
  return out;
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
  const edges = collectEdges(root).filter((e) => !e.id.startsWith("z")).map((e) => {
    const [ox, oy] = pos.get(e.container || "root") || [0, 0];
    const sec = (e.sections || [])[0];
    const pts = sec ? [sec.startPoint, ...(sec.bendPoints || []), sec.endPoint].map((p) => ({ x: ox + p.x, y: oy + p.y })) : [];
    const l = (e.labels || [])[0];
    return { i: e.id.startsWith("z") ? -1 : Number(e.id.slice(1)), src: e.sources[0], dst: e.targets[0], pts,
      label: l && l.x !== undefined ? { x: ox + l.x + 3, y: oy + l.y + 11.5 } : null };
  }).sort((a, b) => a.i - b.i);
  return { boxes, edges, width: root.width, height: root.height };
}

// a drawing's text -> its layout, the last KEEP: a step whose drawing did not change (after an
// edit elsewhere, a Reload, a step seen before) is not laid out again
const layouts = new Map();
const firsts = new Map(); // a drawing's text -> the layout it got first, for reset
const KEEP = 400;

/** Drawing ``d`` laid out: each box measured in ``probe`` (a hidden element in the page, so the
page's fonts apply), then ELK. */
export async function layout(d, probe, seed = null) {
  const key = JSON.stringify(d);
  if (layouts.has(key)) return layouts.get(key);
  const L = await layOut(seeded(d, seed), probe, Boolean(seed) && seeded(d, seed) !== d);
  layouts.set(key, L);
  if (!firsts.has(key)) firsts.set(key, L);
  if (firsts.size > KEEP) firsts.delete(firsts.keys().next().value);
  if (layouts.size > KEEP) layouts.delete(layouts.keys().next().value);
  return L;
}

/** Drawing ``d`` laid out again (clean graph), the layout kept for it from now on: with ``at``
(the boxes' places after a drag, id -> {x, w}) in the left-to-right order they are in now, so
every edge is routed again round the arrangement made by hand; else from scratch, ELK free to
choose every order. */
export async function relayout(d, probe, at = null) {
  const L = at ? await layOut(seeded(d, { boxes: at }), probe, true) : await layOut(d, probe);
  layouts.set(JSON.stringify(d), L);
  return L;
}

/** Drawing ``d``'s first layout again, the moves by hand dropped (reset). */
export async function resetLayout(d, probe) {
  const key = JSON.stringify(d);
  const L = firsts.get(key);
  if (!L) return layout(d, probe);
  L.moves = new Map();
  layouts.set(key, L);
  return L;
}

/** The first loop of the chain loop ``id`` is in (its hops followed up). */
export function chainHead(d, id) {
  const loops = new Set(d.nodes.filter((n) => n.kind === "loop").map((n) => n.id));
  let at = id;
  for (let prev; (prev = d.edges.find((e) => e.to === at && loops.has(e.from) && loops.has(at)));) at = prev.from;
  return at;
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
  // a temporal mux's inputs of one task put in the order their edges come in, so they do not cross
  const muxes = muxOrders(d, L);
  if (muxes.size) {
    ({ graph, own } = elkGraph(d, sizes, lanes, forced, muxes));
    const again = absolute(d, await elk().layout(graph));
    if (crossings(again) <= crossings(L)) L = again;
  }
  L.ms = performance.now() - t0;
  return L;
}

/** How many pairs of edge segments cross in layout ``L`` (the check that a new order helped). */
function crossings(L) {
  const segs = L.edges.flatMap((e) => e.pts.slice(1).map((q, i) => [e.pts[i], q]));
  let n = 0;
  for (let i = 0; i < segs.length; i++) {
    for (let j = i + 1; j < segs.length; j++) {
      const [a, b] = segs[i], [c, e] = segs[j];
      const h1 = Math.abs(a.y - b.y) < 0.5, h2 = Math.abs(c.y - e.y) < 0.5;
      if (h1 === h2) continue; // parallel runs do not cross
      const [hh, vv] = h1 ? [[a, b], [c, e]] : [[c, e], [a, b]];
      const y = hh[0].y, x = vv[0].x;
      if (x > Math.min(hh[0].x, hh[1].x) + 0.5 && x < Math.max(hh[0].x, hh[1].x) - 0.5
        && y > Math.min(vv[0].y, vv[1].y) + 0.5 && y < Math.max(vv[0].y, vv[1].y) - 0.5) n++;
    }
  }
  return n;
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

/** A right-angled route from an output (below a box) to an input (above one), for an edge whose
end was moved by hand. */
function reroute(a, b) {
  const gap = 14;
  if (b.y - a.y > 2 * gap) {
    const my = b.y - gap - Math.min(18, (b.y - a.y) / 4);
    return [a, { x: a.x, y: my }, { x: b.x, y: my }, b];
  }
  const side = Math.max(a.x, b.x) + 30;
  return [a, { x: a.x, y: a.y + gap }, { x: side, y: a.y + gap }, { x: side, y: b.y - gap }, { x: b.x, y: b.y - gap }, b];
}

/** Drawing ``d`` with layout ``L`` into ``canvas`` (emptied first): the outlines and edges, the
boxes, the port names. A box dragged moves by hand (``L.moves``, kept with the layout while the
page is open): its edges are routed again as right angles and its outlines grow round it; an
outline's title or frame drags all it holds. ``scale()`` is the view's zoom, so a drag follows the
pointer. */
export function render(canvas, d, L, scale = () => 1) {
  return new Drawn(canvas, d, L, scale);
}

class Drawn {
  constructor(canvas, d, L, scale) {
    Object.assign(this, { canvas, d, L, scale });
    if (!L.moves) L.moves = new Map();
    const k = ++uid;
    canvas.replaceChildren();
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
    // outlines first, under everything; each holds its boxes and the outlines inside it
    const inside = new Map(d.outlines.map((o) => [o.id, []]));
    for (const n of d.nodes) if (n.parent && inside.has(n.parent)) inside.get(n.parent).push(n.id);
    const kids = (oid) => [...inside.get(oid), ...d.outlines.filter((o) => o.parent === oid).flatMap((o) => kids(o.id))];
    this.outlines = d.outlines.map((o) => {
      const b = L.boxes.get(o.id);
      if (!b) return null;
      const cls = { group: "group-box", accelerator: "acc-box", block: "block-box" }[o.kind] || "group-box";
      const g = svgEl("g", { class: cls, "data-sel": o.sel, "data-group": o.kind === "group" ? o.sel : null, "data-block": o.block || null });
      const rect = { x: b.x, y: b.y, width: b.w, height: b.h, rx: o.kind === "block" ? 10 : 6 };
      const rects = [svgEl("rect", { ...rect, class: "outline-hit", "data-frame": o.id }), svgEl("rect", rect)];
      const t = svgEl("text", { class: "c-title", "data-outline": o.id, "text-anchor": "end" });
      const lines = titleLines(o.text).map((line) => {
        const span = svgEl("tspan");
        span.textContent = line;
        t.append(span);
        return span;
      });
      const tip = svgEl("title");
      tip.textContent = `${o.kind === "block" ? "double-click to collapse; " : ""}drag the title or the frame to move what it holds`;
      t.append(tip);
      g.append(...rects, t);
      svg.append(g);
      return { o, base: b, r: b, rects, t, lines, holds: kids(o.id) };
    }).filter(Boolean);
    const edgeLayer = svgEl("g");
    const labels = svgEl("g");
    this.edges = [];
    d.edges.forEach((e, i) => {
      const le = L.edges[i];
      if (!le || !le.pts.length) return;
      const colour = edgeColour(d, e);
      const hit = svgEl("path", { class: "edge-hit", "data-sel": e.id });
      const tip = svgEl("title");
      tip.textContent = [e.sub, e.task ? `task ${e.task}` : null].filter(Boolean).join(" · ") || e.id;
      hit.append(tip);
      const path = svgEl("path", { class: `edge ${e.style}`, "data-sel": e.id, "data-edge": e.id });
      path.style.setProperty("--c", colour);
      if (e.style !== "empty") {
        path._marker = marker(colour);
        path.setAttribute("marker-end", `url(#${path._marker})`);
      }
      edgeLayer.append(hit, path);
      let label = null;
      if (e.label && le.label) {
        label = svgEl("text", { class: `c-label ${e.style}${e.transpose ? " transpose" : ""}`, "data-sel": e.id, "data-edge": e.id });
        if (e.task && d.tasks.length) label.style.fill = colour;
        label.textContent = e.label;
        labels.append(label);
      }
      this.edges.push({ e, src: le.src, dst: le.dst, base: cleanPts(le.pts), path, hit, label,
        lpos: le.label ? { x: le.label.x, y: le.label.y + (e.transpose ? 4.5 : 0) } : null });
    });
    svg.append(edgeLayer, labels);
    canvas.append(svg);
    this.els = new Map();
    for (const n of d.nodes) {
      const b = L.boxes.get(n.id);
      const el = boxEl(d, n);
      el.classList.add("c-box");
      Object.assign(el.style, { width: `${b.w}px`, height: `${b.h}px` });
      canvas.append(el);
      this.els.set(n.id, el);
    }
    // port names, over the boxes
    const top = svgEl("svg", { class: "dfg-edges c-over", width: L.width, height: L.height });
    this.pnames = [];
    for (const n of d.nodes) {
      const b = L.boxes.get(n.id);
      for (const side of ["in", "out"]) {
        if (!shownNames(n, side)) continue;
        for (const name of side === "in" ? n.ins : n.outs) {
          const p = b.ports.get(`${n.id}.${name}`);
          if (!p) continue;
          const task = (n.port_tasks || {})[name];
          const t = svgEl("text", { class: "c-pname" });
          if (task) t.style.fill = taskColour(d, task);
          t.textContent = task ? name.split("·")[0] : name; // a mux's port: its task by colour
          top.append(t);
          this.pnames.push({ t, node: n.id, p, dy: side === "in" ? 13 : -7 });
        }
      }
    }
    canvas.append(top);
    this.place(d.nodes.map((n) => n.id), true);
    this.titles();
    this.dragging();
  }

  /** Box ``id`` where it is drawn: its layout place and any move by hand. */
  box(id) {
    const b = this.L.boxes.get(id);
    const [dx, dy] = this.L.moves.get(id) || [0, 0];
    return { x: b.x + dx, y: b.y + dy, w: b.w, h: b.h };
  }

  port(pid, node) {
    const p = this.L.boxes.get(node).ports.get(pid);
    const [dx, dy] = this.L.moves.get(node) || [0, 0];
    return { x: p.x + dx, y: p.y + dy };
  }

  /** The boxes ``ids`` put where they are, and what hangs on them drawn again. */
  place(ids, all = false) {
    const { L, d } = this;
    const touched = new Set(ids);
    for (const id of ids) {
      const b = this.box(id);
      Object.assign(this.els.get(id).style, { left: `${b.x}px`, top: `${b.y}px` });
    }
    for (const x of this.edges) {
      if (!all && !touched.has(x.e.from) && !touched.has(x.e.to)) continue;
      const moved = L.moves.has(x.e.from) || L.moves.has(x.e.to);
      x.pts = moved ? cleanPts(reroute(this.port(x.src, x.e.from), this.port(x.dst, x.e.to))) : x.base;
      const path = rounded(x.pts);
      x.path.setAttribute("d", path);
      x.hit.setAttribute("d", path);
      if (x.label) {
        const at = moved ? { x: x.pts[0].x + 5, y: (x.pts[0].y + x.pts[1].y) / 2 + 4 } : x.lpos;
        x.label.setAttribute("x", at.x);
        x.label.setAttribute("y", at.y);
      }
    }
    for (const pn of this.pnames) {
      if (!all && !touched.has(pn.node)) continue;
      const [dx, dy] = L.moves.get(pn.node) || [0, 0];
      pn.t.setAttribute("x", pn.p.x + dx);
      pn.t.setAttribute("y", pn.p.y + dy + pn.dy);
    }
    // an outline holds its boxes: grown round them once one of them moved, inner ones first
    const rects = new Map();
    for (const ol of [...this.outlines].reverse()) {
      let r = ol.base;
      if (ol.holds.some((id) => L.moves.has(id))) {
        const pad = padBottom(ol.o);
        const bs = ol.holds.map((id) => this.box(id));
        for (const o of d.outlines.filter((x) => x.parent === ol.o.id)) if (rects.has(o.id)) bs.push(rects.get(o.id));
        const x0 = Math.min(...bs.map((b) => b.x)) - 14, y0 = Math.min(...bs.map((b) => b.y)) - 14;
        const x1 = Math.max(...bs.map((b) => b.x + b.w)) + 14, y1 = Math.max(...bs.map((b) => b.y + b.h)) + pad;
        r = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
      }
      rects.set(ol.o.id, r);
      ol.r = r;
      for (const rect of ol.rects) {
        rect.setAttribute("x", r.x);
        rect.setAttribute("y", r.y);
        rect.setAttribute("width", r.w);
        rect.setAttribute("height", r.h);
      }
    }
    if (!all) this.titles();
  }

  /** Every outline's title in its lower right corner, right-aligned, one line under another. */
  titles() {
    for (const ol of this.outlines) {
      const r = ol.r;
      const x = r.x + r.w - 8;
      const last = r.y + r.h - 7;
      ol.lines.forEach((span, i) => {
        span.setAttribute("x", x);
        span.setAttribute("y", last - TITLE_LINE * (ol.lines.length - 1 - i));
      });
    }
  }

  /** A press on a box (or an outline's title) that moves drags it; one that does not is a
  click, which selects as before. */
  dragging() {
    const start = (ev, ids) => {
      if (ev.button !== 0) return;
      ev.stopPropagation();
      const was = ids.map((id) => this.L.moves.get(id) || [0, 0]);
      const at = { x: ev.clientX, y: ev.clientY };
      let moved = false;
      const move = (e) => {
        const dx = (e.clientX - at.x) / this.scale(), dy = (e.clientY - at.y) / this.scale();
        if (!moved && Math.abs(dx) + Math.abs(dy) < 4 / this.scale()) return;
        if (!moved) for (const id of ids) this.els.get(id).classList.add("moving");
        moved = true;
        ids.forEach((id, i) => this.L.moves.set(id, [was[i][0] + dx, was[i][1] + dy]));
        this.place(ids);
      };
      const up = () => {
        removeEventListener("pointermove", move);
        removeEventListener("pointerup", up);
        for (const id of ids) this.els.get(id).classList.remove("moving");
        if (moved) this.canvas.addEventListener("click", (e) => e.stopPropagation(), { capture: true, once: true });
      };
      addEventListener("pointermove", move);
      addEventListener("pointerup", up);
    };
    this.canvas.addEventListener("pointerdown", (ev) => {
      const t = ev.target.closest("[data-outline]") || ev.target.closest("rect[data-frame]");
      if (t) {
        const id = t.dataset.outline || t.dataset.frame;
        const ol = this.outlines.find((x) => x.o.id === id);
        if (ol) start(ev, ol.holds);
        return;
      }
      const el = ev.target.closest(".c-box");
      if (el) start(ev, [el.dataset.id]);
    });
  }

  /** The canvas places of the boxes, for the minimap and find: id -> {x, y, w, h}. */
  places() {
    return new Map(this.d.nodes.map((n) => [n.id, this.box(n.id)]));
  }
}

// -- what a selection lights ------------------------------------------------------------------

// a box data pass through on their way, and the port each side of it that carries them on: a
// mux's ``a·mvt`` in becomes its ``a`` out, a demux's ``out`` in its ``out·mvt`` out
const THROUGH = {
  mux: { on: (inP, outP, task) => outP === inP.split("·")[0], back: (outP, inP, task) => inP.split("·")[0] === outP && (!task || inP.endsWith(`·${task}`)) },
  demux: { on: (inP, outP, task) => outP.split("·")[0] === inP && (!task || outP.endsWith(`·${task}`)), back: (outP, inP) => inP === outP.split("·")[0] },
  slicer: { on: () => true, back: () => true },
  packer: { on: () => true, back: () => true },
  link: { on: () => true, back: () => true },
};

/** The edges a selection traces: an edge's whole route, every hop of its datum to the same port
(``data``, ``node``, ``port``), on through the muxes, slicers, packers and link FIFOs it passes;
a container's every route. */
export function routes(d, sels) {
  const kind = new Map(d.nodes.map((n) => [n.id, n.kind]));
  const out = new Set();
  const seeds = [];
  for (const s of sels) {
    const e = d.edges.find((x) => x.id === s);
    if (e) seeds.push(...(e.data && e.node ? d.edges.filter((x) => x.data === e.data && x.node === e.node && x.port === e.port) : [e]));
    else if (d.nodes.some((n) => n.kind === "container" && n.sel === s)) seeds.push(...d.edges.filter((x) => x.data === s));
  }
  const todo = seeds.map((e) => [e, e.task]);
  while (todo.length) {
    const [e, task] = todo.pop();
    if (out.has(e.id)) continue;
    out.add(e.id);
    const on = THROUGH[kind.get(e.to)];
    if (on) for (const x of d.edges) if (x.from === e.to && on.on(e.to_port || "", x.from_port || "", task)) todo.push([x, task]);
    const back = THROUGH[kind.get(e.from)];
    if (back) for (const x of d.edges) if (x.to === e.from && back.back(e.from_port || "", x.to_port || "", task)) todo.push([x, task]);
  }
  return out;
}

/** The boxes and edges upstream and downstream of the selection (focus): every box a path of
edges joins to a selected box, either way, and the edges between them. */
export function lineage(d, sels) {
  const boxes = new Set();
  for (const n of d.nodes) {
    if (sels.includes(n.sel) || sels.includes(n.id) || (n.chips || []).some(([x]) => sels.includes(x))) boxes.add(n.id);
  }
  for (const e of d.edges) if (sels.includes(e.id)) boxes.add(e.from).add(e.to);
  if (!boxes.size) return null;
  const lit = new Set(boxes);
  for (const [a, b] of [["from", "to"], ["to", "from"]]) {
    const todo = [...boxes];
    const seen = new Set(boxes);
    while (todo.length) {
      const id = todo.pop();
      for (const e of d.edges) {
        if (e[a] !== id || seen.has(e[b])) continue;
        seen.add(e[b]);
        lit.add(e[b]);
        todo.push(e[b]);
      }
    }
  }
  return { boxes: lit, edges: new Set(d.edges.filter((e) => lit.has(e.from) && lit.has(e.to)).map((e) => e.id)) };
}

// -- folding a loop chain -------------------------------------------------------------------

/** Drawing ``d`` with the loop chains whose first loop is in ``folded`` drawn as one box each
(``fold:<first loop>``): the edges into any of its loops go into it, those out of any leave it,
the hops between its loops go. A chain is the loops joined by their hops. */
export function foldChains(d, folded) {
  if (!folded || !folded.size) return d;
  const loops = new Map(d.nodes.filter((n) => n.kind === "loop").map((n) => [n.id, n]));
  const up = new Map();
  for (const e of d.edges) if (loops.has(e.from) && loops.has(e.to)) up.set(e.to, e.from);
  const head = (id) => (up.has(id) ? head(up.get(id)) : id);
  const into = new Map(); // a folded loop -> its box
  const chains = new Map();
  for (const id of loops.keys()) {
    const h = head(id);
    if (!folded.has(h)) continue;
    into.set(id, `fold:${h}`);
    if (!chains.has(h)) chains.set(h, []);
    chains.get(h).push(loops.get(id));
  }
  if (!into.size) return d;
  const nodes = [];
  for (const n of d.nodes) {
    if (!into.has(n.id)) {
      nodes.push(n);
      continue;
    }
    const h = head(n.id);
    if (n.id !== h) continue;
    const chain = chains.get(h);
    nodes.push({ id: `fold:${h}`, kind: "loop", loop: chain[0].loop, sel: h, folded: true, row: n.row,
      text: `${h} ▸ ${chain.length} loop${chain.length > 1 ? "s" : ""}`, sub: chain.map((x) => x.sub).join(" › "),
      chips: chain.flatMap((x) => x.chips || []) });
  }
  const seen = new Set();
  const edges = [];
  for (const e of d.edges) {
    const a = into.get(e.from) || e.from, b = into.get(e.to) || e.to;
    if (a === b) continue;
    const id = `${a}${e.from_port && !into.has(e.from) ? `.${e.from_port}` : ""}->${b}${e.to_port && !into.has(e.to) ? `.${e.to_port}` : ""}#${e.data || ""}`;
    if (seen.has(id)) continue; // one edge per datum between a folded chain and a box
    seen.add(id);
    edges.push({ ...e, from: a, to: b });
  }
  return { ...d, nodes, edges };
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

  /** The canvas point (``cx``, ``cy``) in the middle of the panel, at zoom ``s``. */
  centre(cx, cy, s = this.s) {
    const r = this.vp.getBoundingClientRect();
    this.set(s, r.width / 2 - cx * s, r.height / 2 - cy * s);
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

// -- the minimap ------------------------------------------------------------------------------

/** A small map of the whole step in the panel's corner, the part in view outlined; a click or a
drag on it moves the view there. Shown only when the step does not fit the panel. */
export class Minimap {
  constructor(vp, view) {
    this.view = view;
    this.el = document.createElement("canvas");
    this.el.className = "c-minimap";
    this.el.title = "the whole step: click or drag to move the view";
    vp.append(this.el);
    const go = (e) => {
      const r = this.el.getBoundingClientRect();
      if (!this.k) return;
      this.view.centre((e.clientX - r.left) / this.k, (e.clientY - r.top) / this.k);
    };
    this.el.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      this.el.setPointerCapture(e.pointerId);
      go(e);
      const move = (ev) => go(ev);
      this.el.addEventListener("pointermove", move);
      this.el.addEventListener("pointerup", () => this.el.removeEventListener("pointermove", move), { once: true });
    });
    this.el.addEventListener("click", (e) => e.stopPropagation());
  }

  draw(drawn) {
    if (drawn) this.drawn = drawn;
    const v = this.view, L = v.L;
    if (!L || !this.drawn) return;
    const r = v.vp.getBoundingClientRect();
    const fits = L.width * v.s <= r.width + 1 && L.height * v.s <= r.height + 1;
    this.el.hidden = fits;
    if (fits) return;
    const W = 170;
    this.k = Math.min(W / L.width, 130 / L.height);
    const w = Math.ceil(L.width * this.k), h = Math.ceil(L.height * this.k);
    const dpr = window.devicePixelRatio || 1;
    Object.assign(this.el, { width: w * dpr, height: h * dpr });
    Object.assign(this.el.style, { width: `${w}px`, height: `${h}px` });
    const c = this.el.getContext("2d");
    c.setTransform(dpr * this.k, 0, 0, dpr * this.k, 0, 0);
    const css = getComputedStyle(v.vp);
    c.fillStyle = css.getPropertyValue("--edge") || "#888";
    for (const b of this.drawn.places().values()) c.fillRect(b.x, b.y, b.w, b.h);
    c.strokeStyle = css.getPropertyValue("--hl") || "#d4386a";
    c.lineWidth = 2 / this.k;
    c.strokeRect(-v.tx / v.s, -v.ty / v.s, r.width / v.s, r.height / v.s);
  }
}
