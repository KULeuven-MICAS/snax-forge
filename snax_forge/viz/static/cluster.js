// The cluster at one cycle (VIS3, D61, D62), drawn under the schedule and
// following its selected cycle. Plain HTML and CSS: every arrow is straight
// and sits in the grid column of the bank or block it points at, so nothing
// is measured. Top to bottom, the way requests go from memory to control:
//
//   L1 banks      grouped by superbank (wide_bits / width_bits banks); a bank
//                 names the port whose request it accepted (R/W, row) and the
//                 port its read data goes back to
//   arrow pairs   one per bank, between the banks and the interconnect
//   interconnect  status in the middle: grey without traffic, green with
//                 traffic and no conflict, red with the list of conflicts
//                 (served port first); a static description in the corner
//   arrow pairs   one per requester (streamer, DMA), centred on its box
//   requesters    streamers (one column per lane: port, FIFO slots, count),
//                 the DMA with L2 beside it
//   arrows        between a streamer and the accelerator it is attached to
//   accelerators  plain boxes over their attached streamers
//   controller    the command running in the cycle and any poll
//
// Every memory hop has two lanes, because both can happen in one cycle
// (D62): the request (teal, dashed, pointing to memory; red when stalled)
// and the response (green, solid, pointing back), which only reads have.
// A read's request is its `grant` event; its response is the `resp` event
// read_latency cycles later, and the FIFO slot fills the cycle after that
// (Queue, D32). A conflict turns the stalled streamer, its request arrow and
// the request arrow into the contested bank red; the served side stays teal.
// Colours are the schedule's (style.css): request teal, read data, busy and
// firing green, memory stall red, flow stall violet, FIFO lilac, controller
// blue.
//
// The DOM is built once per run from the cluster configuration and the
// profile's port list, so another cluster file gives another picture; a
// cycle change only updates it (update), which lets the FIFO slots animate.
//
// What each hop is read from (CONTRACTS.md section 7):
//   request lanes             grant / stall (L1), dma_beat (L2)
//   response lanes            resp (L1 from the xbar, L2 from the DMA)
//   streamer -> accelerator   a firing of the accelerator (it pops its inputs)
//   accelerator -> writer     a rise of a writer lane's count; there is no
//                             push event, so a push and a pop in the same
//                             cycle show no arrow (open item 25)

import { h, int } from "./dom.js";
import { GROUP, bankText, beatTraced, classAt, fifoCount, onPort } from "./events.js";

const BANKS_PER_LINE = 32; // more banks wrap onto a new line, a superbank at a time
const MAX_SLOTS = 8; // deeper FIFOs are drawn as a bar
const DIR_TEXT = { l2_to_l1: "L2 → L1", l1_to_l2: "L1 → L2" };

const views = new WeakMap(); // detail -> view, kept until Reload

/** The cluster view of a run: {el, update(cycle, data)}; built once per run. */
export function clusterView(detail) {
  if (!views.has(detail)) views.set(detail, build(detail));
  return views.get(detail);
}

// -- small parts ---------------------------------------------------------------------

/** A straight arrow; `v` vertical or `h` horizontal; `lane` req, resp or "" (data). `guide` draws a faint line when off. */
function arrow(axis, lane = "", guide = true) {
  return h("div", { class: `cl-arrow ${axis}${lane ? ` ${lane}` : ""}${guide ? "" : " none"}` }, h("i"));
}

/** A request lane and a response lane side by side (v) or stacked (h). */
function pair(axis) {
  const req = arrow(axis, "req");
  const resp = arrow(axis, "resp");
  return { el: h("div", { class: `cl-pair ${axis}` }, req, resp), req, resp };
}

/** Show an arrow pointing `dir` (up, down, left, right), or hide it with dir null. */
function setArrow(el, dir, bad = false, title = "") {
  el.classList.remove("on", "up", "down", "left", "right", "bad");
  if (dir) el.classList.add("on", dir);
  if (dir && bad) el.classList.add("bad");
  el.title = dir ? title : "";
}

/** A block: a box with its name, kind and class badge. */
function block(name, kind) {
  const cls = h("span", { class: "cl-cls" });
  const body = h("div", { class: "cl-body" });
  const el = h("div", { class: "cl-box" },
    h("div", { class: "cl-head" }, h("b", {}, name), kind ? h("small", {}, kind) : null, cls), body);
  return { el, cls, body };
}

/** Set a block's class badge and stripe (the schedule's colour of that class). */
function setClass(b, cls) {
  const group = cls ? GROUP[cls] || "idle" : "none";
  b.el.dataset.group = group;
  b.cls.replaceChildren(...(cls ? [h("i", { class: `key g-${group}` }), cls] : []));
}

function setText(el, text, kind = null) {
  if (el.textContent !== text) el.textContent = text;
  el.classList.toggle("req", kind === "req" && !!text);
  el.classList.toggle("resp", kind === "resp" && !!text);
}

const portText = (e) => `${e.port} ${e.k === "resp" ? "read data back" : e.k === "grant" ? "request accepted" : e.wider ? "stalled by a wider grant" : "stalled"}: ${e.k === "resp" ? "read" : e.w ? "write" : "read"} addr ${e.addr}, ${bankText(e.banks)}`;

// -- building ------------------------------------------------------------------------

function build(detail) {
  const cfg = detail.run.cluster;
  const { profile, trace } = detail;
  const comps = cfg.components;
  const l1 = cfg.l1;
  const nBanks = l1.n_banks;
  const group = Math.min(Math.max(Math.round((l1.wide_bits || l1.width_bits) / l1.width_bits), 1), nBanks);
  const xbar = comps.find((c) => c.kind === "xbar");
  const ports = profile.ports; // name -> {owner, width, ...}
  const portsOf = (owner) => Object.keys(ports).filter((p) => ports[p].owner === owner);
  const portLabel = (p) => (portsOf(ports[p]?.owner).length === 1 ? ports[p].owner : p);

  // L1 banks, a line of superbanks at a time.
  const bankCells = [];
  const bankPairs = [];
  const sbPerLine = Math.max(1, Math.floor(BANKS_PER_LINE / group));
  const nSb = Math.ceil(nBanks / group);
  const lines = [];
  for (let s0 = 0; s0 < nSb; s0 += sbPerLine) {
    const sbs = [];
    for (let s = s0; s < Math.min(s0 + sbPerLine, nSb); s++) sbs.push(s);
    const first = sbs[0] * group;
    const last = Math.min((sbs[sbs.length - 1] + 1) * group, nBanks);
    const heads = sbs.map((s, i) => h("div", { class: `cl-sb-head${i ? " sb-first" : ""}`, style: { gridColumn: `span ${Math.min(group, nBanks - s * group)}` } },
      group > 1 ? `superbank ${s}` : "", group > 1 ? h("small", {}, `banks ${s * group}–${Math.min((s + 1) * group, nBanks) - 1}`) : null));
    const cells = [];
    const pairs = [];
    for (let b = first; b < last; b++) {
      const sep = b > first && b % group === 0 ? " sb-first" : "";
      const who = h("span", { class: "who" });
      const acc = h("small", { class: "acc" });
      const back = h("small", { class: "back" });
      const cell = h("div", { class: `cl-bank${sep}` }, h("b", {}, String(b)), who, acc, back);
      bankCells[b] = { cell, who, acc, back };
      const p = pair("v");
      if (sep) p.el.classList.add("sb-first");
      bankPairs[b] = p;
      cells.push(cell);
      pairs.push(p.el);
    }
    lines.push(h("div", { class: "cl-banks", style: { gridTemplateColumns: `repeat(${last - first}, minmax(0, 1fr))` } }, heads, cells, pairs));
  }
  const l1Box = h("div", { class: "cl-l1" },
    h("div", { class: "cl-l1-head" }, h("b", {}, "L1"), h("small", {}, `${nBanks} banks × ${l1.width_bits} bit, ${l1.rows} rows, read latency ${l1.read_latency}`)),
    lines);

  // Interconnect: status in the middle, description in the corner.
  const widths = {};
  for (const p of Object.values(ports)) widths[p.width] = (widths[p.width] ?? 0) + 1;
  const byWidth = Object.entries(widths).sort((a, b) => Number(a[0]) - Number(b[0])).map(([w, n]) => `${n} × ${w} bit`).join(", ");
  const nPorts = Object.keys(ports).length;
  const status = h("div", { class: "cl-xbar-status" });
  const xbarBox = xbar ? h("div", { class: "cl-xbar" },
    h("div", { class: "cl-xbar-name" }, h("b", {}, xbar.name), h("small", {}, "interconnect")),
    h("dl", { class: "cl-xbar-desc" },
      h("dt", {}, "requester side"), h("dd", {}, `${nPorts} port${nPorts === 1 ? "" : "s"}: ${byWidth || "none"}`),
      h("dt", {}, "memory side"), h("dd", {}, `${nBanks} banks × ${l1.width_bits} bit${group > 1 ? `, superbanks of ${group}` : ""}`),
      h("dt", {}, "arbitration"), h("dd", {}, "wider first, then round robin")),
    status) : null;

  // Requesters in one grid: streamers grouped under their accelerator, then the DMA and L2.
  const streamers = comps.filter((c) => c.kind === "streamer");
  const accels = comps.filter((c) => c.kind === "accel");
  const dmas = comps.filter((c) => c.kind === "dma");
  const ctls = comps.filter((c) => c.kind === "controller");
  const order = [];
  for (const a of accels) {
    for (const s of Object.values(a.attach || {})) {
      if (streamers.some((c) => c.name === s) && !order.includes(s)) order.push(s);
    }
  }
  for (const s of streamers) if (!order.includes(s.name)) order.push(s.name);
  const cols = order.map((name) => ({ kind: "streamer", name }));
  for (const d of dmas) cols.push({ kind: "dma", name: d.name });
  const hasL2 = !!cfg.l2 && dmas.length > 0;
  if (hasL2) cols.push({ kind: "l2arrow" }, { kind: "l2" });
  const colOf = (name) => cols.findIndex((c) => c.name === name) + 1;
  const template = cols.map((c) => ({ streamer: "minmax(7.5rem, 1fr)", dma: "minmax(12rem, 1.3fr)", l2arrow: "2.6rem", l2: "minmax(8rem, 0.7fr)" })[c.kind]).join(" ");
  const place = (el, col, row, span = 1) => { el.style.gridColumn = `${col} / span ${span}`; el.style.gridRow = String(row); return el; };
  const reqItems = [];

  const parts = { streamers: {}, dmas: {}, accels: {}, ctls: {} };
  for (const name of order) {
    const col = colOf(name);
    const sp = profile.streamers[name];
    const b = block(name, sp?.write ? "writer" : "reader");
    const chips = portsOf(name).map((p) => ({ port: p, el: h("span", { class: "cl-port" }, p.startsWith(`${name}.`) ? p.slice(name.length + 1) : p) }));
    const depth = sp?.fifo.depth ?? 0;
    const nLanes = sp?.fifo.hist.length ?? 0;
    const lanes = [];
    for (let l = 0; l < nLanes; l++) {
      const slots = depth <= MAX_SLOTS ? Array.from({ length: depth }, () => h("i", { class: "slot" })) : [];
      const fill = depth > MAX_SLOTS ? h("i", { class: "fill" }) : null;
      lanes.push({ slotsEl: h("div", { class: `slots${fill ? " bar" : ""}` }, slots, fill), slots, fill, count: h("small", { class: "count" }) });
    }
    // One grid column per lane: its port on top, then its FIFO slots, then its count, so they line up.
    const lined = chips.length === nLanes;
    const grid = h("div", { class: "cl-lanes", style: { gridTemplateColumns: `repeat(${Math.max(nLanes, 1)}, 1.45rem)` } },
      lined ? chips.map((c) => c.el) : null, lanes.map((x) => x.slotsEl), lanes.map((x) => x.count));
    b.body.append(
      !lined && chips.length ? h("div", { class: "cl-ports" }, chips.map((c) => c.el)) : null,
      nLanes ? grid : null,
      nLanes ? h("small", { class: "cl-sub" }, `${sp.fifo.name}, depth ${depth}`) : null);
    const up = pair("v");
    const attachedTo = accels.find((a) => Object.values(a.attach || {}).includes(name));
    const down = arrow("v", "", !!attachedTo);
    reqItems.push(place(up.el, col, 1), place(b.el, col, 2), place(down, col, 3));
    parts.streamers[name] = { b, chips, lanes, depth, up, down, fifo: sp?.fifo.name, write: !!sp?.write, accel: attachedTo?.name };
  }

  for (const d of dmas) {
    const col = colOf(d.name);
    const b = block(d.name, "DMA");
    const chips = portsOf(d.name).map((p) => ({ port: p, el: h("span", { class: "cl-port" }, `${ports[p].width} bit`) }));
    const task = h("div", { class: "cl-line" });
    const rd = h("div", { class: "cl-line" });
    const back = h("div", { class: "cl-line" });
    const wr = h("div", { class: "cl-line" });
    b.body.append(chips.length ? h("div", { class: "cl-ports" }, chips.map((c) => c.el)) : null, task, rd, back, wr);
    const up = pair("v");
    reqItems.push(place(up.el, col, 1), place(b.el, col, 2));
    parts.dmas[d.name] = { b, chips, task, rd, back, wr, up };
  }

  let l2 = null;
  if (hasL2) {
    const col = cols.length - 1;
    const b = block("L2", `${int(cfg.l2.size_bytes / 1024)} KiB, read latency ${cfg.l2.read_latency}`);
    const req = h("div", { class: "cl-line" });
    const back = h("div", { class: "cl-line" });
    b.body.append(req, back);
    b.el.classList.add("plain");
    b.el.dataset.group = "none";
    const p = pair("h");
    reqItems.push(place(p.el, col, 2), place(b.el, col + 1, 2));
    l2 = { b, req, back, p, dma: dmas[0].name };
  }

  for (const a of accels) {
    const mine = Object.values(a.attach || {}).map(colOf).filter((c) => c > 0);
    const c0 = mine.length ? Math.min(...mine) : 1;
    const c1 = mine.length ? Math.max(...mine) : Math.max(order.length, 1);
    const p = a.params || {};
    const facts = [p.op, p.lanes !== undefined ? `${p.lanes} lane${p.lanes === 1 ? "" : "s"}` : null,
      p.latency !== undefined ? `L ${p.latency}` : null, p.ii !== undefined ? `II ${p.ii}` : null].filter((x) => x !== null && x !== undefined);
    const b = block(a.name, a.accel);
    const fire = h("div", { class: "cl-line" });
    b.body.append(h("div", { class: "cl-line muted" }, facts.join(", ")), fire);
    reqItems.push(place(b.el, c0, 4, c1 - c0 + 1));
    parts.accels[a.name] = { b, fire };
  }

  for (const c of ctls) {
    const b = block(c.name, "controller");
    const cmd = h("div", { class: "cl-line" });
    const poll = h("div", { class: "cl-line" });
    b.body.append(cmd, poll);
    b.el.classList.add("ctl");
    b.el.style.gridColumn = "1 / -1";
    b.el.style.gridRow = "5";
    reqItems.push(b.el);
    parts.ctls[c.name] = { b, cmd, poll };
  }

  const req = h("div", { class: "cl-req", style: { gridTemplateColumns: template || "1fr" } }, reqItems);
  const note = h("p", { class: "note cl-note" });
  const legend = h("ul", { class: "legend" },
    h("li", {}, h("i", { class: "key g-req dashed" }), "Request (accepted)"),
    h("li", {}, h("i", { class: "key g-busy" }), "Read data back, firing"),
    h("li", {}, h("i", { class: "key g-mem" }), "Stall, contested bank"),
    h("li", {}, h("i", { class: "key g-fifo" }), "FIFO slot in use"),
    h("li", {}, "Requests point to memory, read data points back"));
  const el = h("div", { class: "cluster" }, legend, note, h("div", { class: "scroll" }, h("div", { class: "cl-frame" }, l1Box, xbarBox, req)));

  const cmds = [];
  let cmdsFrom = null; // the task events the command list was taken from

  // -- one cycle ----------------------------------------------------------------------

  function update(t, data = {}) {
    const { tasks = [], fifo = new Map(), events = [] } = data;
    if (cmdsFrom !== tasks) {
      cmds.length = 0;
      for (const e of tasks) if (e.k === "cmd") cmds.push(e);
      cmdsFrom = tasks;
    }
    const level = trace?.level ?? "off";
    const beat = level === "beat";
    const on = t !== null && t !== undefined && level !== "off";
    const cls = (name) => (on ? classAt(trace.intervals[name], t) : null);
    const xbarOn = on && !!xbar && beatTraced(trace, xbar.name, t);
    const mine = (src) => (on && beatTraced(trace, src, t) ? events.filter((e) => e.src === src) : null);

    // The xbar's events of the cycle, by bank and by owner.
    const portEvs = xbarOn ? events.filter((e) => onPort(e) && e.mem === "l1") : [];
    const grants = portEvs.filter((e) => e.k === "grant");
    const stalls = portEvs.filter((e) => e.k === "stall");
    const resps = portEvs.filter((e) => e.k === "resp");
    const bank = [];
    const at = (b) => (bank[b] ??= { grant: null, stalls: [], resp: null });
    for (const g of grants) for (const b of g.banks) at(b).grant = g;
    for (const s of stalls) for (const b of s.banks) at(b).stalls.push(s);
    for (const r of resps) for (const b of r.banks) at(b).resp = r;
    const evsOf = {};
    for (const e of portEvs) (evsOf[ports[e.port]?.owner ?? e.port] ??= []).push(e);

    // Note above the picture.
    let msg = "";
    if (level === "off") msg = "This run was traced at level off: only the cluster's structure is shown. Run it again with --trace beat.";
    else if (!on) msg = "Select a cycle in the schedule.";
    else if (!beat) msg = "Task-level trace: classes, tasks and commands only. Run with --trace beat for requests, read data, FIFO counts, firings and DMA beats.";
    else if (!xbarOn) msg = "The xbar's beat events were filtered out of this cycle (see the header): no requests or read data to show.";
    setText(note, msg);
    el.classList.toggle("task-only", !beat);

    // Banks and their arrow pairs.
    for (let b = 0; b < nBanks; b++) {
      const st = bank[b];
      const { cell, who, acc, back } = bankCells[b];
      const g = st?.grant;
      const r = st?.resp;
      const contested = !!st?.stalls.length;
      cell.classList.toggle("req", !!g);
      cell.classList.toggle("resp", !!r);
      cell.classList.toggle("bad", contested);
      setText(who, g ? portLabel(g.port) : "");
      setText(acc, g ? `${g.w ? "W" : "R"} r${g.row}` : "");
      setText(back, r ? `↓ ${portLabel(r.port)}` : "");
      cell.title = st ? [g, ...st.stalls, r].filter(Boolean).map(portText).join("\n") : `bank ${b}`;
      const any = g ?? st?.stalls[0];
      setArrow(bankPairs[b].req, any ? "up" : null, contested,
        contested ? `bank ${b} contested: ${g ? `${g.port} served, ` : ""}${st.stalls.map((s) => s.port).join(", ")} stalled` : any ? `${any.port} ${any.w ? "writes" : "reads"} bank ${b}` : "");
      setArrow(bankPairs[b].resp, r ? "down" : null, false, r ? `bank ${b} read data goes back to ${r.port}` : "");
    }

    // Interconnect status.
    if (xbarBox) {
      xbarBox.classList.remove("idle", "ok", "bad");
      const conflicts = new Map(); // stalled banks + winners -> {banks, winners, stalled}
      for (const s of stalls) {
        const winners = [...new Set(s.banks.map((b) => bank[b]?.grant).filter(Boolean))];
        const key = `${s.banks.join(",")}|${winners.map((g) => g.port).join(",")}`;
        if (!conflicts.has(key)) conflicts.set(key, { banks: s.banks, winners, stalled: [] });
        conflicts.get(key).stalled.push(s);
      }
      if (!xbarOn) {
        xbarBox.classList.add("idle");
        status.replaceChildren(h("span", { class: "muted" }, !on ? "" : beat ? "not traced in this cycle" : "no beat trace"));
      } else if (conflicts.size) {
        xbarBox.classList.add("bad");
        status.replaceChildren(h("ul", { class: "cl-conflicts" }, [...conflicts.values()].map((c) => {
          const wider = c.stalled.some((s) => s.wider);
          const served = c.winners.length ? `${c.winners.map((g) => g.port).join(", ")} served${wider ? " (wider grant)" : ""}` : "no grant";
          return h("li", {}, h("b", {}, `${bankText(c.banks)}:`), ` ${served}, `, h("span", { class: "stalled" }, `${c.stalled.map((s) => s.port).join(", ")} stalled`));
        })));
      } else if (portEvs.length) {
        xbarBox.classList.add("ok");
        status.replaceChildren();
      } else {
        xbarBox.classList.add("idle");
        status.replaceChildren();
      }
    }

    // Port chips and requester arrow pairs, shared by streamers and DMAs.
    const chipState = (chips) => {
      for (const c of chips) {
        const evs = (evsOf[ports[c.port]?.owner] ?? []).filter((e) => e.port === c.port);
        c.el.classList.toggle("req", evs.some((e) => e.k === "grant"));
        c.el.classList.toggle("bad", evs.some((e) => e.k === "stall"));
        c.el.classList.toggle("resp", evs.some((e) => e.k === "resp"));
        c.el.title = evs.length ? evs.map(portText).join("\n") : `${c.port}: nothing in this cycle`;
      }
    };
    const reqPair = (p, evs) => {
      const reqs = (evs ?? []).filter((e) => e.k !== "resp");
      const bad = reqs.some((e) => e.k === "stall");
      const back = (evs ?? []).filter((e) => e.k === "resp");
      setArrow(p.req, reqs.length ? "up" : null, bad, reqs.map(portText).join("\n"));
      setArrow(p.resp, back.length ? "down" : null, false, back.map(portText).join("\n"));
      return bad;
    };

    // Streamers.
    for (const [name, p] of Object.entries(parts.streamers)) {
      setClass(p.b, cls(name));
      p.b.el.classList.toggle("bad", reqPair(p.up, evsOf[name]));
      chipState(p.chips);
      const fifoOn = on && beatTraced(trace, p.fifo, t);
      let rise = false;
      p.lanes.forEach((lane, l) => {
        const n = on ? fifoCount(fifo, trace, p.fifo, l, t) : null;
        lane.slotsEl.classList.toggle("unknown", fifoOn && n === null);
        lane.slotsEl.classList.toggle("untraced", !fifoOn);
        lane.slots.forEach((s, i) => s.classList.toggle("full", n !== null && i < n));
        if (lane.fill) lane.fill.style.height = `${n === null ? 0 : (100 * n) / p.depth}%`;
        setText(lane.count, !fifoOn ? "–" : n === null ? "?" : String(n));
        lane.slotsEl.title = !fifoOn ? `lane ${l}: not traced in this cycle` : n === null ? `lane ${l}: count unknown (set before the trace window)` : `lane ${l}: ${n} of ${p.depth}`;
        if (p.write && n !== null) {
          const before = fifoCount(fifo, trace, p.fifo, l, t - 1);
          if (before !== null && n > before) rise = true;
        }
      });
      if (p.accel) {
        const fired = !p.write && !!(mine(p.accel) ?? []).some((e) => e.k === "fire");
        setArrow(p.down, fired ? "down" : rise ? "up" : null, false,
          fired ? `${p.accel} fires: pops a beat from ${name}` : rise ? `${p.accel} pushes into ${name}` : "");
      }
    }

    // DMA and L2: its L1 side comes from the xbar, its L2 side from its own beats and responses.
    for (const [name, p] of Object.entries(parts.dmas)) {
      setClass(p.b, cls(name));
      p.b.el.classList.toggle("bad", reqPair(p.up, evsOf[name]));
      chipState(p.chips);
      const task = on ? (detail.tasks[name] ?? []).find((s) => s.start <= t && t < s.done) : null;
      setText(p.task, task ? `${DIR_TEXT[task.direction] ?? task.direction ?? "task"}, start in ${task.start}, done in ${task.done}` : on ? "no task" : "");
      const own = mine(name);
      const rd = own?.find((e) => e.k === "dma_beat" && e.side === "src");
      const wr = own?.find((e) => e.k === "dma_beat" && e.side === "dst");
      const l2back = own?.find((e) => e.k === "resp");
      const l1back = (evsOf[name] ?? []).find((e) => e.k === "resp");
      const MEM = (m) => m.toUpperCase();
      setText(p.rd, own === null && on && beat ? "beats not traced in this cycle" : rd ? `read request, beat ${rd.i}, ${MEM(rd.mem)} addr ${rd.addr}` : "", "req");
      setText(p.back, l2back ? `read data back, beat ${l2back.i}, L2 addr ${l2back.addr}` : l1back ? `read data back, L1 addr ${l1back.addr}` : "", "resp");
      setText(p.wr, wr ? `write request, beat ${wr.i}, ${MEM(wr.mem)} addr ${wr.addr}` : "", "req");
      if (l2 && l2.dma === name) {
        const l2req = rd?.mem === "l2" ? rd : wr?.mem === "l2" ? wr : null;
        setArrow(l2.p.req, l2req ? "right" : null, false, l2req ? `${name} ${l2req.side === "src" ? "reads" : "writes"} L2 addr ${l2req.addr}` : "");
        setArrow(l2.p.resp, l2back ? "left" : null, false, l2back ? `L2 read data of beat ${l2back.i} goes back to ${name}` : "");
        setText(l2.req, l2req ? `${l2req.side === "src" ? "read" : "write"} request, addr ${l2req.addr}` : "", "req");
        setText(l2.back, l2back ? `read data back, addr ${l2back.addr}` : "", "resp");
      }
    }

    // Accelerators.
    for (const [name, p] of Object.entries(parts.accels)) {
      setClass(p.b, cls(name));
      const fire = (mine(name) ?? []).find((e) => e.k === "fire");
      setText(p.fire, fire ? `firing ${fire.n}` : "", "resp");
    }

    // Controller.
    for (const [name, p] of Object.entries(parts.ctls)) {
      setClass(p.b, cls(name));
      const c = on ? cmds.find((e) => e.src === name && e.t <= t && t <= e.last) : null;
      const what = c ? (c.op === "wait" ? `wait ${c.block} (${c.mode})` : `${c.op} ${c.reg ?? ""}${c.value !== undefined ? ` = ${c.value}` : ""}`) : "";
      setText(p.cmd, c ? `pc ${c.pc}: ${what}, cycles ${c.t}–${c.last}` : on ? "no command" : "");
      const poll = (mine(name) ?? []).find((e) => e.k === "poll");
      setText(p.poll, poll ? `poll ${poll.block}: busy = ${poll.value}` : "");
      p.poll.classList.toggle("waiting", !!poll?.value);
      p.poll.classList.toggle("done", !!poll && !poll.value);
    }
  }

  update(null);
  return { el, update };
}
