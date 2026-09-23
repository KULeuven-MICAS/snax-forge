// The cluster at one cycle (VIS3, D61), drawn under the schedule and
// following its selected cycle. Plain HTML and CSS: every arrow is straight
// and sits in the grid column of the bank or block it points at, so nothing
// is measured. Top to bottom, the way requests go from memory to control:
//
//   L1 banks      grouped by superbank (wide_bits / width_bits banks); a bank
//                 accessed in the cycle is highlighted and names the port
//                 that was served, with R/W and the row
//   arrows        one per bank, between the banks and the interconnect
//   interconnect  status in the middle: grey without requests, green with
//                 requests and no conflict, red with the list of conflicts
//                 (served port first); a static description in the corner
//   arrows        one per requester (streamer, DMA), centred on its box
//   requesters    streamers (port chips, FIFO lanes as `depth` slots), the
//                 DMA with L2 beside it
//   arrows        between a streamer and the accelerator it is attached to
//   accelerators  plain boxes over their attached streamers
//   controller    the command running in the cycle and any poll
//
// Arrows point the way data moves: down for reads, up for writes. Green is
// a served access, red a stall (the stalled streamer, its arrow, and the
// arrow into the contested bank); the served side stays green. Colours are
// the schedule's (style.css): busy green, memory stall red, flow stall
// violet, FIFO lilac, controller blue.
//
// The DOM is built once per run from the cluster configuration and the
// profile's port list, so another cluster file gives another picture; a
// cycle change only updates it (update), which lets the FIFO slots animate.
//
// What each hop is read from (CONTRACTS.md section 7):
//   bank and requester arrows  grant / stall events of the cycle
//   streamer -> accelerator    a firing of the accelerator (it pops its inputs)
//   accelerator -> writer      a rise of a writer lane's count; there is no
//                              push event, so a push and a pop in the same
//                              cycle show no arrow (open item 25)
//   L2 <-> DMA, DMA <-> L1     dma_beat events: the read and write sides are
//                              independent (D34), each side has its own hop
// A read's arrow is drawn in its grant cycle; the data reaches the FIFO
// read_latency cycles later, where the lane count rises.

import { h, int } from "./dom.js";
import { GROUP, bankText, beatTraced, classAt, fifoCount } from "./events.js";

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

/** A straight arrow; `v` vertical or `h` horizontal. `guide` draws a faint line when off. */
function arrow(axis, guide = true) {
  return h("div", { class: `cl-arrow ${axis}${guide ? "" : " none"}` }, h("i"));
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

function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

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
  const bankArrows = [];
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
    const arrows = [];
    for (let b = first; b < last; b++) {
      const sep = b > first && b % group === 0 ? " sb-first" : "";
      const who = h("span", { class: "who" });
      const acc = h("small", { class: "acc" });
      const cell = h("div", { class: `cl-bank${sep}` }, h("b", {}, String(b)), who, acc);
      bankCells[b] = { cell, who, acc };
      const a = arrow("v");
      if (sep) a.classList.add("sb-first");
      bankArrows[b] = a;
      cells.push(cell);
      arrows.push(a);
    }
    lines.push(h("div", { class: "cl-banks", style: { gridTemplateColumns: `repeat(${last - first}, minmax(0, 1fr))` } }, heads, cells, arrows));
  }
  const l1Box = h("div", { class: "cl-l1" },
    h("div", { class: "cl-l1-head" }, h("b", {}, "L1"), h("small", {}, `${nBanks} banks × ${l1.width_bits} bit, ${l1.rows} rows`)),
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
  const template = cols.map((c) => ({ streamer: "minmax(7.5rem, 1fr)", dma: "minmax(11rem, 1.3fr)", l2arrow: "2.4rem", l2: "minmax(7rem, 0.7fr)" })[c.kind]).join(" ");
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
      const count = h("small", {});
      const el = h("div", { class: `cl-lane${fill ? " bar" : ""}` }, h("div", { class: "slots" }, slots, fill), count);
      lanes.push({ el, slots, fill, count });
    }
    b.body.append(
      chips.length ? h("div", { class: "cl-ports", title: "xbar ports" }, chips.map((c) => c.el)) : null,
      nLanes ? h("div", { class: "cl-fifo", title: `${sp.fifo.name}, depth ${depth}` }, lanes.map((x) => x.el)) : null,
      nLanes ? h("small", { class: "cl-sub" }, `${sp.fifo.name}, depth ${depth}`) : null);
    const up = arrow("v");
    const attachedTo = accels.find((a) => Object.values(a.attach || {}).includes(name));
    const down = arrow("v", !!attachedTo);
    reqItems.push(place(up, col, 1), place(b.el, col, 2), place(down, col, 3));
    parts.streamers[name] = { b, chips, lanes, depth, up, down, fifo: sp?.fifo.name, write: !!sp?.write, accel: attachedTo?.name };
  }

  for (const d of dmas) {
    const col = colOf(d.name);
    const b = block(d.name, "DMA");
    const chips = portsOf(d.name).map((p) => ({ port: p, el: h("span", { class: "cl-port" }, `${ports[p].width} bit`) }));
    const task = h("div", { class: "cl-line" });
    const rd = h("div", { class: "cl-line" });
    const wr = h("div", { class: "cl-line" });
    b.body.append(chips.length ? h("div", { class: "cl-ports" }, chips.map((c) => c.el)) : null, task, rd, wr);
    const up = arrow("v");
    reqItems.push(place(up, col, 1), place(b.el, col, 2));
    parts.dmas[d.name] = { b, chips, task, rd, wr, up };
  }

  let l2 = null;
  if (hasL2) {
    const col = cols.length - 1;
    const b = block("L2", `${int(cfg.l2.size_bytes / 1024)} KiB`);
    const line = h("div", { class: "cl-line" });
    b.body.append(line);
    b.el.classList.add("plain");
    b.el.dataset.group = "none";
    const a = arrow("h");
    reqItems.push(place(a, col, 2), place(b.el, col + 1, 2));
    l2 = { b, line, a, dma: dmas[0].name };
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
    h("li", {}, h("i", { class: "key g-busy" }), "Served access, beat, firing"),
    h("li", {}, h("i", { class: "key g-mem" }), "Stall, contested bank"),
    h("li", {}, h("i", { class: "key g-fifo" }), "FIFO slot in use"),
    h("li", {}, "Arrows point the way data moves: down for reads, up for writes"));
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

    // Grants and stalls by bank and by owner.
    const grants = xbarOn ? events.filter((e) => e.k === "grant" && e.mem === "l1") : [];
    const stalls = xbarOn ? events.filter((e) => e.k === "stall" && e.mem === "l1") : [];
    const bank = [];
    for (const g of grants) for (const b of g.banks) (bank[b] ??= { grant: null, stalls: [] }).grant = g;
    for (const s of stalls) for (const b of s.banks) (bank[b] ??= { grant: null, stalls: [] }).stalls.push(s);
    const reqOf = {};
    for (const e of [...grants, ...stalls]) (reqOf[ports[e.port]?.owner ?? e.port] ??= []).push(e);

    // Note above the picture.
    let msg = "";
    if (level === "off") msg = "This run was traced at level off: only the cluster's structure is shown. Run it again with --trace beat.";
    else if (!on) msg = "Select a cycle in the schedule.";
    else if (!beat) msg = "Task-level trace: classes, tasks and commands only. Run with --trace beat for grants, FIFO counts, firings and DMA beats.";
    else if (!xbarOn) msg = "The xbar's beat events were filtered out of this cycle (see the header): no grants or stalls to show.";
    setText(note, msg);
    el.classList.toggle("task-only", !beat);

    // Banks and their arrows.
    for (let b = 0; b < nBanks; b++) {
      const st = bank[b];
      const { cell, who, acc } = bankCells[b];
      const g = st?.grant;
      const contested = !!st?.stalls.length;
      cell.classList.toggle("on", !!g);
      cell.classList.toggle("bad", contested);
      setText(who, g ? portLabel(g.port) : "");
      setText(acc, g ? `${g.w ? "W" : "R"} r${g.row}` : "");
      cell.title = st ? [g, ...st.stalls].filter(Boolean).map((e) => `${e.port} ${e.k === "grant" ? "served" : "stalled"}: ${e.w ? "write" : "read"} addr ${e.addr}, row ${e.row}`).join("\n") : `bank ${b}`;
      const any = g ?? st?.stalls[0];
      setArrow(bankArrows[b], any ? (any.w ? "up" : "down") : null, contested,
        contested ? `bank ${b} contested: ${g ? `${g.port} served, ` : ""}${st.stalls.map((s) => s.port).join(", ")} stalled` : any ? `${any.port} ${any.w ? "writes" : "reads"} bank ${b}` : "");
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
      } else if (grants.length) {
        xbarBox.classList.add("ok");
        status.replaceChildren();
      } else {
        xbarBox.classList.add("idle");
        status.replaceChildren();
      }
    }

    // Port chips, shared by streamers and DMAs.
    const chipState = (chips) => {
      for (const c of chips) {
        const evs = (reqOf[ports[c.port]?.owner] ?? []).filter((e) => e.port === c.port);
        const g = evs.find((e) => e.k === "grant");
        const s = evs.find((e) => e.k === "stall");
        c.el.classList.toggle("ok", !!g);
        c.el.classList.toggle("bad", !!s);
        c.el.title = evs.length ? evs.map((e) => `${e.port} ${e.k === "grant" ? "served" : e.wider ? "stalled by a wider grant" : "stalled"}: ${e.w ? "write" : "read"} addr ${e.addr}, ${bankText(e.banks)}`).join("\n") : `${c.port}: no request`;
      }
    };
    const reqArrow = (a, evs) => {
      const any = evs?.[0];
      const bad = !!evs?.some((e) => e.k === "stall");
      setArrow(a, any ? (any.w ? "up" : "down") : null, bad, any ? (bad ? "stalled" : any.w ? "writes to L1" : "reads from L1") : "");
    };

    // Streamers.
    for (const [name, p] of Object.entries(parts.streamers)) {
      setClass(p.b, cls(name));
      const evs = reqOf[name];
      p.b.el.classList.toggle("bad", !!evs?.some((e) => e.k === "stall"));
      chipState(p.chips);
      reqArrow(p.up, evs);
      const fifoOn = on && beatTraced(trace, p.fifo, t);
      let rise = false;
      p.lanes.forEach((lane, l) => {
        const n = on ? fifoCount(fifo, trace, p.fifo, l, t) : null;
        lane.el.classList.toggle("unknown", fifoOn && n === null);
        lane.el.classList.toggle("untraced", !fifoOn);
        lane.slots.forEach((s, i) => s.classList.toggle("full", n !== null && i < n));
        if (lane.fill) lane.fill.style.height = `${n === null ? 0 : (100 * n) / p.depth}%`;
        setText(lane.count, !fifoOn ? "–" : n === null ? "?" : String(n));
        lane.el.title = !fifoOn ? `lane ${l}: not traced in this cycle` : n === null ? `lane ${l}: count unknown (set before the trace window)` : `lane ${l}: ${n} of ${p.depth}`;
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

    // DMA and L2.
    for (const [name, p] of Object.entries(parts.dmas)) {
      setClass(p.b, cls(name));
      const evs = reqOf[name];
      p.b.el.classList.toggle("bad", !!evs?.some((e) => e.k === "stall"));
      chipState(p.chips);
      reqArrow(p.up, evs);
      const task = on ? (detail.tasks[name] ?? []).find((s) => s.start <= t && t < s.done) : null;
      setText(p.task, task ? `${DIR_TEXT[task.direction] ?? task.direction ?? "task"}, start in ${task.start}, done in ${task.done}` : on ? "no task" : "");
      const beats = mine(name);
      const rd = beats?.find((e) => e.k === "dma_beat" && e.side === "src");
      const wr = beats?.find((e) => e.k === "dma_beat" && e.side === "dst");
      const beatText = (e, verb) => (e ? `${verb} beat ${e.i}, ${e.mem.toUpperCase()} addr ${e.addr}` : "");
      setText(p.rd, beats === null && on && beat ? "beats not traced in this cycle" : beatText(rd, "read"));
      setText(p.wr, beatText(wr, "write"));
      p.rd.classList.toggle("hit", !!rd);
      p.wr.classList.toggle("hit", !!wr);
      if (l2 && l2.dma === name) {
        const l2rd = rd?.mem === "l2" ? rd : null;
        const l2wr = wr?.mem === "l2" ? wr : null;
        setArrow(l2.a, l2rd ? "left" : l2wr ? "right" : null, false, l2rd ? `${name} reads L2` : l2wr ? `${name} writes L2` : "");
        setText(l2.line, l2rd ? `read addr ${l2rd.addr}` : l2wr ? `write addr ${l2wr.addr}` : "");
        l2.line.classList.toggle("hit", !!(l2rd || l2wr));
      }
    }

    // Accelerators.
    for (const [name, p] of Object.entries(parts.accels)) {
      setClass(p.b, cls(name));
      const fire = (mine(name) ?? []).find((e) => e.k === "fire");
      setText(p.fire, fire ? `firing ${fire.n}` : "");
      p.fire.classList.toggle("hit", !!fire);
    }

    // Controller.
    for (const [name, p] of Object.entries(parts.ctls)) {
      setClass(p.b, cls(name));
      const c = on ? cmds.find((e) => e.src === name && e.t <= t && t <= e.last) : null;
      const what = c ? (c.op === "wait" ? `wait ${c.block} (${c.mode})` : `${c.op} ${c.reg ?? ""}${c.value !== undefined ? ` = ${c.value}` : ""}`) : "";
      setText(p.cmd, c ? `pc ${c.pc}: ${what}, cycles ${c.t}–${c.last}` : on ? "no command" : "");
      const poll = (mine(name) ?? []).find((e) => e.k === "poll");
      setText(p.poll, poll ? `poll ${poll.block}: busy = ${poll.value}` : "");
      p.poll.classList.toggle("hit", !!poll);
      p.poll.classList.toggle("waiting", !!poll?.value);
    }
  }

  update(null);
  return { el, update };
}
