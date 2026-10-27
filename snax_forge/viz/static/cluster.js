// The cluster at one cycle (D57, D61, D62), drawn under the schedule and
// following its selected cycle: the data side only, no core or controller
// box (the cores are the schedule's lanes). Plain HTML and CSS: every arrow
// is straight and sits in the grid column of the bank or block it points at,
// so nothing is measured. Top to bottom, the way requests go from memory to
// the accelerator:
//
//   L1 banks      every bank, grouped by superbank; a bank shows its access
//                 (the port or the DMA, read or write, row) and its read data
//                 going back; red when it refused a requester
//   arrow pairs   one per bank, between the banks and the interconnect
//   interconnect  grey without traffic, green with traffic, red with its
//                 conflicts (the port served, the ports refused)
//   arrow pairs   one per mover lane (its crossbar port) and one for the DMA
//   requesters    the movers (one column per lane: port, FIFO slots, count),
//                 the DMA with L2 beside it (its own pair of lanes)
//   arrows        reader → accelerator on a fire, accelerator → writer on a
//                 push
//   accelerator   its class, task, firing and push
//
// Every memory hop has two lanes, because both can happen in one cycle
// (D62): the request (teal, dashed, pointing to memory; red when refused)
// and the response (green, solid, pointing back), which only reads have.
//
// The page is built once per run from the run detail's cluster (the movers,
// their channels and FIFO depths, the accelerator) and the banks of the first
// /cluster answer (their superbanks); a cycle change only updates it in
// place from /api/run/<n>/cluster?t=T (update), so the FIFO slots animate.
// Everything shown is that answer's: a field the run's level cannot give is
// absent from it and drawn as nothing, with the answer's `needs` in the note.

import { CLASS_GROUP, h, int } from "./dom.js";
import { DIRECTION_TEXT, elemsText, eventText } from "./events.js";

const MAX_SLOTS = 8; // deeper FIFOs are drawn as a bar
const LANE = 1.45; // rem, a lane's column at its narrowest
const LANE_MAX = 2; // and at its widest
const BOX_PAD = 1.35; // rem, a box's padding and borders
const MIN_MOVER = 7.5; // rem
const FRAME_MIN = 44; // rem, style.css .cl-frame

const views = new WeakMap(); // detail -> view, kept until Reload

/** The cluster view of a run, {el, update(answer)}; built once per run from the first answer. */
export function clusterView(detail, first) {
  if (!views.has(detail)) views.set(detail, build(detail, first));
  return views.get(detail);
}

// -- small parts ---------------------------------------------------------------------

/** A straight arrow; `v` vertical or `h` horizontal; `lane` req, resp or "" (data). */
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
  const group = cls ? CLASS_GROUP[cls] || "idle" : "none";
  b.el.dataset.group = group;
  b.cls.replaceChildren(...(cls ? [h("i", { class: `key g-${group}` }), cls] : []));
}

function setText(el, text, kind = null) {
  if (el.textContent !== text) el.textContent = text;
  el.classList.toggle("req", kind === "req" && !!text);
  el.classList.toggle("resp", kind === "resp" && !!text);
}

const lines = (xs) => xs.filter(Boolean).join("\n");

/** A bank's access as its tooltip says it. */
function accessText(a) {
  const what = elemsText(a);
  return `${a.port} ${a.op === "w" ? "writes" : "reads"} row ${a.row}${a.i !== undefined ? ` (DMA copy ${a.id} beat ${a.i})` : ""}${what ? `: ${what}` : ""}`;
}

// -- building ------------------------------------------------------------------------

function build(detail, first) {
  const acc = detail.cluster.accelerator;
  const movers = acc.movers;
  // Each crossbar port's short label in a bank (a0 for acc_a.lane0), from the cluster.
  const short = { dma: "dma" };
  for (const m of movers) for (let l = 0; l < m.lanes; l++) short[`${m.name}.lane${l}`] = `${m.port}${l}`;

  // L1: every bank of the answer, grouped by its superbank.
  const bankCells = [];
  const bankPairs = [];
  const groups = new Map(); // superbank -> [bank]
  for (const b of first.banks) {
    if (!groups.has(b.superbank)) groups.set(b.superbank, []);
    groups.get(b.superbank).push(b.bank);
  }
  const heads = [];
  const cells = [];
  const pairs = [];
  let column = 0;
  for (const [sb, banks] of groups) {
    const sep = column ? " sb-first" : "";
    heads.push(h("div", { class: `cl-sb-head${sep}`, style: { gridColumn: `span ${banks.length}` } },
      `superbank ${sb}`, h("small", {}, `banks ${banks[0]}–${banks[banks.length - 1]}`)));
    banks.forEach((b, i) => {
      const who = h("span", { class: "who" });
      const acc = h("small", { class: "acc" });
      const back = h("small", { class: "back" });
      const cell = h("div", { class: `cl-bank${i === 0 ? sep : ""}` }, h("b", {}, String(b)), who, acc, back);
      const p = pair("v");
      if (i === 0 && sep) p.el.classList.add("sb-first");
      bankCells[b] = { cell, who, acc, back };
      bankPairs[b] = p;
      cells.push(cell);
      pairs.push(p.el);
    });
    column += banks.length;
  }
  const c = detail.cluster;
  const l1Box = h("div", { class: "cl-l1" },
    h("div", { class: "cl-l1-head" }, h("b", {}, "L1"), h("small", {}, `${int(c.banks)} banks of ${c.bank_bytes} B, ${int(c.l1_size / 1024)} KiB`)),
    h("div", { class: "cl-banks", style: { gridTemplateColumns: `repeat(${column}, minmax(0, 1fr))` } }, heads, cells, pairs));

  const status = h("div", { class: "cl-xbar-status" });
  const xbarBox = h("div", { class: "cl-xbar" },
    h("div", { class: "cl-xbar-name" }, h("b", {}, "interconnect"), h("small", {}, "crossbar")),
    h("dl", { class: "cl-xbar-desc" },
      h("dt", {}, "requester side"), h("dd", {}, [...movers.map((m) => `${m.name} ${m.lanes} channels`), "the DMA"].join(", ")),
      h("dt", {}, "memory side"), h("dd", {}, `${int(c.banks)} banks`)),
    status);

  // Requesters in one grid: the movers as the streamer orders them, then the DMA and L2.
  const cols = [...movers.map((m) => ({ kind: "mover", m })), { kind: "dma" }, { kind: "l2arrow" }, { kind: "l2" }];
  const narrowest = (x) => (x.kind === "mover"
    ? Math.max(MIN_MOVER, x.m.lanes * LANE + Math.max(x.m.lanes - 1, 0) * 0.25 + BOX_PAD)
    : { dma: 12, l2arrow: 2.6, l2: 7 }[x.kind]);
  const track = (x) => (x.kind === "mover" ? `minmax(${narrowest(x).toFixed(2)}rem, ${Math.max(1, x.m.lanes / 4)}fr)`
    : { dma: "minmax(12rem, 1.3fr)", l2arrow: "2.6rem", l2: "minmax(7rem, 0.7fr)" }[x.kind]);
  const frameMin = Math.max(FRAME_MIN, cols.reduce((a, x) => a + narrowest(x), 0) + 0.6 * (cols.length - 1));
  const place = (el, col, row, span = 1) => { el.style.gridColumn = `${col} / span ${span}`; el.style.gridRow = String(row); return el; };
  const items = [];

  const parts = { movers: {}, lanes: {} };
  movers.forEach((m, i) => {
    const col = i + 1;
    const b = block(m.name, m.write ? "writer" : "reader");
    const task = h("div", { class: "cl-line" });
    const lanes = [];
    const ups = [];
    for (let l = 0; l < m.lanes; l++) {
      const port = `${m.name}.lane${l}`;
      const slots = m.fifo_depth <= MAX_SLOTS ? Array.from({ length: m.fifo_depth }, () => h("i", { class: "slot" })) : [];
      const fill = m.fifo_depth > MAX_SLOTS ? h("i", { class: "fill" }) : null;
      const x = { port, chip: h("span", { class: "cl-port" }, `ch ${l}`), slotsEl: h("div", { class: `slots${fill ? " bar" : ""}` }, slots, fill), slots, fill, count: h("small", { class: "count" }), up: pair("v"), depth: m.fifo_depth };
      lanes.push(x);
      ups.push(x.up.el);
      parts.lanes[port] = x;
    }
    const grid = (cells) => h("div", { class: "cl-lanes", style: { gridTemplateColumns: `repeat(${Math.max(m.lanes, 1)}, minmax(${LANE}rem, ${LANE_MAX}rem))` } }, cells);
    b.body.append(task, grid([lanes.map((x) => x.chip), lanes.map((x) => x.slotsEl), lanes.map((x) => x.count)]),
      h("small", { class: "cl-sub" }, `${m.lanes} channel${m.lanes === 1 ? "" : "s"} (num_channel), FIFO depth ${m.fifo_depth}`));
    const down = arrow("v");
    items.push(place(grid(ups), col, 1), place(b.el, col, 2), place(down, col, 3));
    parts.movers[m.name] = { b, task, down, write: m.write };
  });

  const dcol = movers.length + 1;
  const dma = block("dma", "DMA");
  const dmaLines = { copies: h("div", { class: "cl-line" }), beat: h("div", { class: "cl-line" }), burst: h("div", { class: "cl-line" }), back: h("div", { class: "cl-line" }) };
  dma.body.append(dmaLines.copies, dmaLines.beat, dmaLines.burst, dmaLines.back);
  const dmaUp = pair("v");
  items.push(place(dmaUp.el, dcol, 1), place(dma.el, dcol, 2));
  const l2 = block("L2", "beside the cluster");
  const l2Lines = { req: h("div", { class: "cl-line" }), back: h("div", { class: "cl-line" }) };
  l2.body.append(l2Lines.req, l2Lines.back);
  l2.el.classList.add("plain");
  l2.el.dataset.group = "none";
  const l2Pair = pair("h");
  items.push(place(l2Pair.el, dcol + 1, 2), place(l2.el, dcol + 2, 2));

  const u = acc.unit || {};
  const facts = [u.op !== undefined ? `op ${u.op}` : null, u.lanes !== undefined ? `${u.lanes} lanes` : null,
    u.latency !== undefined ? `latency ${u.latency}` : null, u.ii !== undefined ? `II ${u.ii}` : null].filter(Boolean);
  const unit = block(acc.name, "accelerator");
  const unitLines = { facts: h("div", { class: "cl-line muted" }, facts.join(", ")), task: h("div", { class: "cl-line" }), fire: h("div", { class: "cl-line" }), push: h("div", { class: "cl-line" }) };
  unit.body.append(unitLines.facts, unitLines.task, unitLines.fire, unitLines.push);
  items.push(place(unit.el, 1, 4, Math.max(movers.length, 1)));

  const req = h("div", { class: "cl-req", style: { gridTemplateColumns: cols.map(track).join(" ") } }, items);
  const note = h("p", { class: "note cl-note" });
  const legend = h("ul", { class: "legend" },
    h("li", {}, h("i", { class: "key g-req dashed" }), "Request (accepted)"),
    h("li", {}, h("i", { class: "key g-busy" }), "Read data back, fire, push"),
    h("li", {}, h("i", { class: "key g-mem" }), "Refused, contested bank"),
    h("li", {}, h("i", { class: "key g-fifo" }), "FIFO slot in use"),
    h("li", {}, "Requests point to memory, read data points back"));
  const el = h("div", { class: "cluster" }, legend, note,
    h("div", { class: "scroll" }, h("div", { class: "cl-frame", style: { minWidth: `${frameMin.toFixed(2)}rem` } }, l1Box, xbarBox, req)));

  // -- one cycle ----------------------------------------------------------------------

  /** Draw an answer of /cluster, or clear the picture (null). */
  function update(a) {
    const on = !!a;
    const needs = on ? Object.keys(a.needs ?? {}) : [];
    let msg = "";
    if (!on) msg = "Select a cycle in the schedule, or step with the arrow keys.";
    else if (a.level === "off") msg = "This run was traced at level off: only the cluster's structure is shown.";
    else if (needs.length) msg = `Traced at ${a.level}: ${needs.join(", ")} need${needs.length === 1 ? "s" : ""} --trace ${[...new Set(Object.values(a.needs))].join(" or ")}.`;
    if (on && a.filtered) msg += `${msg ? " " : ""}The beat lines cover ${a.filtered.sources.join(", ")} only.`;
    setText(note, msg);
    el.classList.toggle("task-only", !on || !("access" in (a.banks[0] ?? {})));

    for (const bk of on ? a.banks : first.banks.map((b) => ({ bank: b.bank }))) {
      const { cell, who, acc: what, back } = bankCells[bk.bank];
      const access = bk.access ?? [];
      const backs = bk.back ?? [];
      const stalled = bk.stalled ?? [];
      cell.classList.toggle("req", access.length > 0);
      cell.classList.toggle("resp", backs.length > 0);
      cell.classList.toggle("bad", stalled.length > 0);
      setText(who, access.map((x) => short[x.port] ?? x.port).join(" "));
      setText(what, access.map((x) => `${x.op === "w" ? "W" : "R"} r${x.row}`).join(" "));
      setText(back, backs.length ? `↓ ${backs.map((e) => short[e.src] ?? e.src).join(" ")}` : "");
      cell.title = lines([`bank ${bk.bank}`, ...access.map(accessText),
        ...backs.map((e) => `read data back to ${e.src} (asked in cycle ${e.t})`),
        stalled.length ? `served ${stalled[0].served}, refused ${stalled.slice(1).map((s) => s.port).join(", ")}` : null]);
      setArrow(bankPairs[bk.bank].req, access.length || stalled.length ? "up" : null, stalled.length > 0,
        lines([...access.map(accessText), ...stalled.slice(1).map((s) => `${s.port} refused`)]));
      setArrow(bankPairs[bk.bank].resp, backs.length ? "down" : null, false, lines(backs.map((e) => `bank ${bk.bank} read data back to ${e.src}`)));
    }

    const x = on ? a.interconnect : {};
    xbarBox.classList.remove("idle", "ok", "bad");
    if (x.conflicts?.length) {
      xbarBox.classList.add("bad");
      status.replaceChildren(h("ul", { class: "cl-conflicts" }, x.conflicts.map((k) =>
        h("li", {}, h("b", {}, `bank ${k.bank}:`), ` ${k.served} served, `, h("span", { class: "stalled" }, `${k.refused.join(", ")} refused`)))));
    } else {
      xbarBox.classList.add(x.traffic ? "ok" : "idle");
      status.replaceChildren(h("span", { class: "muted" }, !on ? "" : !("traffic" in x) ? `no beat lines at ${a.level}` : x.traffic ? "" : "no traffic"));
    }

    for (const m of movers) {
      const p = parts.movers[m.name];
      const lane0 = on ? a.movers[`${m.name}.lane0`] : null;
      setClass(p.b, lane0?.class ?? null);
      setText(p.task, lane0?.task ? `task ${lane0.task}` : on ? "no task" : "");
      let bad = false;
      for (let l = 0; l < m.lanes; l++) {
        const q = parts.lanes[`${m.name}.lane${l}`];
        const s = on ? a.movers[q.port] : {};
        const g = s.grant ?? null;
        const st = s.stall ?? null;
        const back = s.back ?? [];
        bad ||= !!st;
        q.chip.classList.toggle("req", !!g);
        q.chip.classList.toggle("bad", !!st);
        q.chip.classList.toggle("resp", back.length > 0);
        q.chip.title = lines([q.port, g ? `${g.op === "w" ? "write" : "read"} accepted: ${eventText(g)}` : null, st ? eventText(st) : null,
          ...back.map((e) => `read data back (asked in cycle ${e.t}): ${elemsText(e)}`)]);
        setArrow(q.up.req, g || st ? "up" : null, !!st, g ? eventText(g) : st ? eventText(st) : "");
        setArrow(q.up.resp, back.length ? "down" : null, false, lines(back.map((e) => `${q.port} read data back: ${elemsText(e)}`)));
        const n = "fifo" in s ? s.fifo : null;
        q.slotsEl.classList.toggle("untraced", n === null);
        q.slots.forEach((slot, i) => slot.classList.toggle("full", n !== null && i < n));
        if (q.fill) q.fill.style.height = `${n === null ? 0 : (100 * n) / q.depth}%`;
        setText(q.count, n === null ? "–" : String(n));
        q.slotsEl.title = n === null ? `${q.port}: FIFO counts need --trace stall` : `${q.port}: ${n} of ${q.depth}`;
      }
      p.b.el.classList.toggle("bad", bad);
      const fire = on ? a.acc.fire : null;
      const push = on ? a.acc.push : null;
      if (p.write) setArrow(p.down, push ? "up" : null, false, push ? `${acc.name} pushes result ${push.n} into ${m.name}` : "");
      else setArrow(p.down, fire ? "down" : null, false, fire ? `${acc.name} fires ${fire.n}: pops beat ${fire.b?.[m.name]} of ${m.name}` : "");
    }

    const copies = on ? a.dma.copies : [];
    const beats = copies.flatMap((cp) => cp.beat ?? []);
    const bursts = copies.flatMap((cp) => cp.burst ?? []);
    const backs = copies.flatMap((cp) => cp.back ?? []);
    const l1back = backs.filter((e) => e.k === "dma");
    const l2back = backs.filter((e) => e.k === "burst");
    setText(dmaLines.copies, copies.map((cp) => `copy ${cp.id} ${cp.task}: ${DIRECTION_TEXT[cp.dir] ?? cp.dir}, ${int(cp.bytes)} B`).join("; ") || (on ? "no copy" : ""));
    setText(dmaLines.beat, beats.map((b) => `L1 ${b.op === "w" ? "write" : "read"}, beat ${b.i + 1} of ${b.n}`).join("; "), "req");
    setText(dmaLines.burst, bursts.map((b) => `L2 ${b.op === "w" ? "write" : "read"}, ${b.size} B`).join("; "), "req");
    setText(dmaLines.back, [...l1back.map((e) => `L1 data back, beat ${e.i}`), ...l2back.map(() => "L2 data back")].join("; "), "resp");
    dma.el.title = lines([...beats.map(eventText), ...bursts.map(eventText), ...backs.map((e) => `read data back (asked in cycle ${e.t}): ${eventText(e)}`)]);
    setArrow(dmaUp.req, beats.length ? "up" : null, false, lines(beats.map(eventText)));
    setArrow(dmaUp.resp, l1back.length ? "down" : null, false, lines(l1back.map((e) => `L1 read data back: ${eventText(e)}`)));
    setArrow(l2Pair.req, bursts.length ? "right" : null, false, lines(bursts.map(eventText)));
    setArrow(l2Pair.resp, l2back.length ? "left" : null, false, lines(l2back.map((e) => `L2 read data back: ${eventText(e)}`)));
    setText(l2Lines.req, bursts.map((b) => `${b.op === "w" ? "write" : "read"} ${b.size} B${b.region ? ` of ${b.region}` : ""}`).join("; "), "req");
    setText(l2Lines.back, l2back.length ? "read data back" : "", "resp");

    const un = on ? a.acc : {};
    setClass(unit, un.class ?? null);
    setText(unitLines.task, un.task ? `task ${un.task}` : on ? "no task" : "");
    const vals = (v) => Object.entries(v ?? {}).map(([k, xs]) => `${k} ${(xs ?? []).join(", ")}`).join("; ");
    setText(unitLines.fire, un.fire ? `fire ${un.fire.n}: ${vals(un.fire.vals)}` : "", "resp");
    setText(unitLines.push, un.push ? `push ${un.push.n}: ${(un.push.vals ?? []).join(", ")}` : "", "resp");
  }

  update(null);
  return { el, update };
}

/** The cycle's event list: one row per line of the answer, in trace order. */
export function eventList(a) {
  if (!a.events.length) return h("p", { class: "note" }, `No line in cycle ${int(a.t)}.`);
  return h("div", { class: "scroll" }, h("table", { class: "cycle-table" },
    h("thead", {}, h("tr", {}, h("th", {}, "Source"), h("th", {}, "Kind"), h("th", {}, "Line"))),
    h("tbody", {}, a.events.map((e) => h("tr", {}, h("th", {}, e.src), h("td", {}, e.k), h("td", {}, h("div", {}, eventText(e))))))));
}
