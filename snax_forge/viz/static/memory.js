// The memory tab (VIS4a, VIS4b, D95-D97): where a run's regions sit in each
// memory, and on a beat-level run how its words move.
//
// Everything shown is computed by the server (viz/memory.py): per memory its
// geometry, its regions and its rows, folded where a row repeats the one
// above with one constant step per region. This module only draws:
//
//   heading      the memory's geometry and how much of it the regions use
//   scale bar    the whole memory to scale, one segment per region
//   region table base, end, bytes, share, banks (or words) and rows touched
//   grid         L1 as banks × rows, L2 as beats of words; a cell is one word,
//                in its region's colour with the element's flat index; the
//                tooltip gives the index, the address, the bank and the row
//   fold lines   click to show the hidden rows, at most MAX_ROWS at a time;
//                an empty run is one line
//
// Region colours are a meaning of this tab only (D96): a region keeps its
// colour in every memory, so A in L2 and A in L1 match. The modes, the cycle
// overlay and the picked element are described under "drawing" below.
//
// The cluster view (VIS3) uses the same rows to name the element a bank
// holds in its tooltips: `ensureRows` loads the rows a cycle touches (a block
// of MAX_ROWS at a time, once per run) and `elementAt` names what is there.

import { h, int, pct } from "./dom.js";

export const MAX_ROWS = 256; // viz/memory.py MAX_ROWS: rows per request
const LABEL_EVERY_CELL = 32; // above this many columns only the first cell of a run is labelled
const N_COLOURS = 8; // style.css --region-0 ... --region-7

// -- data shared with the cluster view -------------------------------------------------

const store = new WeakMap(); // detail -> {view: Promise, rows: {mem: Map(row -> line)}, blocks: Set}

function stateOf(api, detail) {
  if (!store.has(detail)) {
    store.set(detail, { view: api.memory(detail.name), loaded: null, rows: {}, blocks: new Map() });
  }
  const s = store.get(detail);
  if (!s.loaded) s.loaded = s.view.then((v) => { s.data = v; return v; });
  return s;
}

/** The memory view of a run (/api/run/<name>/memory), fetched once per run. */
export function memoryData(api, detail) {
  return stateOf(api, detail).loaded;
}

function keep(s, mem, lines) {
  const m = (s.rows[mem] ??= new Map());
  for (const line of lines) if (line.kind === "row") m.set(line.row, line);
}

/** Load the rows `rows` of `mem`, a block of MAX_ROWS rows at a time, each block once. */
export async function ensureRows(api, detail, mem, rows) {
  const s = stateOf(api, detail);
  const view = await s.loaded;
  const m = view.memories.find((x) => x.mem === mem);
  if (!m || !m.regions.length) return;
  keep(s, mem, m.lines);
  const waits = [];
  for (const r of new Set(rows)) {
    if (s.rows[mem].has(r) || r < 0 || r >= m.geometry.rows) continue;
    const b = Math.floor(r / MAX_ROWS);
    const key = `${mem}|${b}`;
    if (!s.blocks.has(key)) {
      const from = b * MAX_ROWS;
      const to = Math.min(from + MAX_ROWS, m.geometry.rows);
      s.blocks.set(key, api.memoryRows(detail.name, mem, from, to).then((got) => keep(s, mem, got.rows)));
    }
    waits.push(s.blocks.get(key));
  }
  await Promise.all(waits);
}

/** Flat index -> index tuple of a region (row-major). */
function unravel(flat, shape) {
  const idx = [];
  for (let d = shape.length - 1; d >= 0; d--) {
    idx.unshift(flat % shape[d]);
    flat = Math.floor(flat / shape[d]);
  }
  return idx;
}

const elemText = (region, flat) => `${region.name}[${unravel(flat, region.shape).join(",")}]`;

/**
 * What the word at (column, row) of `mem` holds, as text (`A[17]`), "" for an
 * empty word, or null when the run names no regions or the row is not loaded.
 */
export function elementAt(detail, mem, column, row) {
  const s = store.get(detail);
  const m = s?.data?.memories.find((x) => x.mem === mem);
  const line = s?.rows[mem]?.get(row);
  if (!m || !m.regions.length || !line) return null;
  return (line.cells[column] ?? []).map(([ri, flat]) => elemText(m.regions[ri], flat)).join(" + ");
}

// -- drawing -----------------------------------------------------------------------------
//
// What the grid shows depends on `show` in the hash (D97, beat-level runs only):
//   (none)     region colours, the layout of VIS4a
//   conflicts  region colours, a red badge per word with the conflicts it took part in
//              (waited or won), and a count per bank column; rows fold by equal counts
//   arrival    a cycle per word, coloured on one scale: its first write
//   use        its first read by anything but a DMA
//   wait       the cycles between the two
// In every mode the selected cycle (`cycle`, shared with the schedule and stepped
// by the arrow keys) is drawn over the grid: a word requested in that cycle is
// outlined teal, a word whose read data comes back green and a word held back
// red, the colours of D61 and D62; a folded line says how many of its words moved.
// Clicking a word picks its element (`pick`): its words are outlined in every
// memory and a drawer shows its journey, its firings and, in L1, its conflicts.

const SHOWS = { "": "Layout", conflicts: "Conflicts", arrival: "Arrival", use: "First use", wait: "Wait" };
const SHOW_NOTES = {
  conflicts: "Red badge: the L1 conflicts a word took part in, waiting or served; the count over a bank is its conflicts. Rows fold only when their counts match.",
  arrival: "Colour and number: the cycle a word was first written. Rows fold when every word's time moves by one step.",
  use: "Colour and number: the cycle a word was first read by a streamer (not the DMA). Rows fold when every word's time moves by one step.",
  wait: "Colour and number: the cycles between a word's arrival and its first use. Rows fold when every word's wait moves by one step.",
};
const OVERLAY_KINDS = ["grant", "stall", "resp", "dma_beat"];

const expanded = new Map(); // "run|show|mem|from" -> number of hidden rows shown; kept until the page reloads
const marked = new WeakMap(); // detail -> Map(show -> Promise of the marked view)
const markedRows = new WeakMap(); // detail -> Map("show|mem" -> Map(row -> line))
const whole = new WeakMap(); // detail -> Promise of the whole run's conflicts
const journeys = new WeakMap(); // detail -> Map(pick -> Promise of a journey)
let current = null; // what renderMemory left on the page, so a cycle or pick change updates it

function cacheOf(map, detail) {
  if (!map.has(detail)) map.set(detail, new Map());
  return map.get(detail);
}

function kib(bytes) {
  return bytes >= 1024 && bytes % 1024 === 0 ? `${int(bytes / 1024)} KiB` : `${int(bytes)} B`;
}

function geometryText(m) {
  const g = m.geometry;
  if (m.mem === "l1") {
    return `${g.columns} banks × ${int(g.rows)} rows of ${g.word_bytes * 8}-bit words, ${kib(g.size_bytes)}; word-interleaved, superbanks of ${g.group}`;
  }
  return `${int(g.columns * g.rows)} words of ${g.word_bytes * 8} bits, drawn as ${int(g.rows)} beats of ${g.columns} words, ${kib(g.size_bytes)}`;
}

function colourOf(colours, name) {
  if (!colours.has(name)) colours.set(name, colours.size % N_COLOURS);
  return colours.get(name);
}

function swatch(ci) {
  return h("i", { class: `key region r${ci}` });
}

function scaleBar(m, colours) {
  const g = m.geometry;
  const segs = m.regions.map((r) => {
    const left = (100 * (r.start - g.base_addr)) / g.size_bytes;
    const width = (100 * (r.end - r.start)) / g.size_bytes;
    return h("div", { class: `mem-seg r${colourOf(colours, r.name)}`, style: { left: `${left}%`, width: `${width}%` },
      title: `${r.name}: bytes ${int(r.start)} to ${int(r.end)} (${int(r.bytes)} B, ${pct(r.bytes, g.size_bytes)} of ${m.mem.toUpperCase()})` },
    h("span", {}, r.name));
  });
  return h("div", { class: "mem-scale-wrap" },
    h("div", { class: "mem-scale", role: "img", "aria-label": `${m.mem.toUpperCase()} to scale with its regions` }, segs),
    h("div", { class: "mem-scale-ticks" }, h("span", {}, int(g.base_addr)), h("span", {}, int(g.base_addr + g.size_bytes))));
}

function regionTable(m, colours) {
  const col = m.mem === "l1" ? "banks" : "words";
  return h("div", { class: "scroll" }, h("table", { class: "mem-regions" },
    h("thead", {}, h("tr", {}, ["Region", "Shape", "Base", "End", "Bytes", "Share", m.mem === "l1" ? "Banks" : "Words of a beat", m.mem === "l1" ? "Rows" : "Beats"]
      .map((t, i) => h("th", { class: i >= 2 && i <= 5 ? "num" : null }, t)))),
    h("tbody", {}, m.regions.map((r) => h("tr", {},
      h("td", {}, swatch(colourOf(colours, r.name)), r.name),
      h("td", {}, `[${r.shape.join(" × ")}], strides ${r.strides.join(", ")} B`),
      h("td", { class: "num" }, int(r.start)),
      h("td", { class: "num" }, int(r.end)),
      h("td", { class: "num" }, int(r.bytes)),
      h("td", { class: "num" }, pct(r.bytes, m.geometry.size_bytes)),
      h("td", {}, spans(r[col])),
      h("td", {}, r.rows[0] === r.rows[1] ? String(r.rows[0]) : `${r.rows[0]}–${r.rows[1]}`))))));
}

/** [0,1,2,3,8] -> "0–3, 8". */
function spans(xs) {
  const out = [];
  for (let i = 0; i < xs.length; i++) {
    let j = i;
    while (j + 1 < xs.length && xs[j + 1] === xs[j] + 1) j++;
    out.push(i === j ? String(xs[i]) : `${xs[i]}–${xs[j]}`);
    i = j;
  }
  return out.join(", ");
}

/** 0..1 on the view's time scale, for the colour of a time mark. */
function scale(range, v) {
  if (!range || v === null || v === undefined) return null;
  return range[1] === range[0] ? 0.5 : (v - range[0]) / (range[1] - range[0]);
}

/** The grid of one memory; `redraw` rebuilds it after a fold is opened or closed. */
function grid(ctx, m, colours, redraw) {
  const g = m.geometry;
  const n = g.columns;
  const show = ctx.show;
  const timed = show && show !== "conflicts";
  const labelAll = n <= LABEL_EVERY_CELL;
  const el = h("div", { class: `mem-grid${timed ? " timed" : ""}`, style: { gridTemplateColumns: `4.2rem repeat(${n}, minmax(${labelAll ? "1.7rem" : "1.1rem"}, 1fr))` } });
  const sb = (c) => (c > 0 && c % g.group === 0 ? " sb-first" : "");
  const head = (c) => (labelAll || c % g.group === 0 ? String(c) : ""); // many columns: superbank starts only
  const bankCount = (c) => (show === "conflicts" && m.mem === "l1" ? ctx.conflicts?.banks?.[String(c)] ?? 0 : 0);
  el.append(h("div", { class: "mem-corner" }, g.row_label),
    ...Array.from({ length: n }, (_, c) => h("div", { class: `mem-col${sb(c)}`, title: `${g.column_label} ${c}${bankCount(c) ? `: ${bankCount(c)} conflicts over the run` : ""}` },
      head(c), bankCount(c) ? h("span", { class: "mem-badge" }, String(bankCount(c))) : null)));
  const cells = ctx.cells(m.mem);
  cells.clear();

  const rowEls = (line) => {
    const out = [h("div", { class: "mem-row-label", title: `${g.row_label} ${line.row}: bytes ${int(line.addr[0])}… (${g.column_label} 0)` },
      String(line.row), h("small", {}, `@${int(line.addr[0])}`))];
    let prev = null;
    line.cells.forEach((cell, c) => {
      const tip = [`${g.column_label} ${c}, ${g.row_label} ${line.row}, byte ${int(line.addr[c])}`];
      const mark = line.marks ? line.marks[c] : null;
      if (!cell.length) {
        const e = h("div", { class: `mem-cell empty${sb(c)}`, title: `${tip[0]}: empty` });
        cells.set(`${line.row}|${c}`, { el: e, tip: e.title, picks: [] });
        out.push(e);
        prev = null;
        return;
      }
      const [ri, flat] = cell[0];
      const r = m.regions[ri];
      for (const [rj, fj] of cell) tip.push(`${elemText(m.regions[rj], fj)}, flat index ${fj}`);
      let text;
      let cls = `mem-cell r${colourOf(colours, r.name)}${cell.length > 1 ? " multi" : ""}${sb(c)}`;
      const style = {};
      let badge = null;
      if (timed) {
        const t = scale(ctx.range(m.mem), mark);
        if (t === null) {
          cls += " untimed";
          text = "";
          tip.push(`${SHOWS[show]}: none`);
        } else {
          style["--t"] = t.toFixed(3);
          if (t > 0.6) cls += " dark-text";
          text = String(mark);
          tip.push(`${SHOWS[show]}: ${show === "wait" ? `${mark} cycles` : `cycle ${mark}`}`);
        }
      } else {
        const cont = prev && prev[0] === ri && prev[1] + 1 === flat;
        text = labelAll || !cont ? String(flat) : "";
        if (show === "conflicts" && mark) {
          badge = h("span", { class: "mem-badge" }, String(mark));
          tip.push(`${mark} conflict${mark === 1 ? "" : "s"} over the run`);
        }
      }
      const picks = cell.map(([rj, fj]) => `${m.regions[rj].name}:${fj}`);
      const e = h("div", { class: cls, style, title: tip.join("\n"), role: "button", tabindex: "0",
        onclick: () => ctx.setHash({ pick: picks[0] }),
        onkeydown: (ev) => { if (ev.key === "Enter") ctx.setHash({ pick: picks[0] }); } }, text, badge);
      cells.set(`${line.row}|${c}`, { el: e, tip: e.title, picks });
      out.push(e);
      prev = [ri, flat];
    });
    return out;
  };

  const spanText = (line) => Object.entries(line.span)
    .map(([ri, [a, b]]) => `${m.regions[ri].name}[${a}..${b}]`).join(", ");
  const stepText = (line) => Object.entries(line.step)
    .map(([ri, d]) => `${m.regions[ri].name} ${d >= 0 ? "+" : "−"}${Math.abs(d)} per ${g.row_label}`).join(", ");
  const markText = (line) => {
    if (show === "conflicts") return line.marked ? `; ${int(line.marked)} conflict${line.marked === 1 ? "" : "s"} inside` : "";
    if (timed && line.mark_span) {
      const d = line.mark_step;
      return `; ${SHOWS[show].toLowerCase()} ${d >= 0 ? "+" : "−"}${Math.abs(d)} per ${g.row_label} (${line.mark_span[0]}–${line.mark_span[1]})`;
    }
    return "";
  };

  const full = (child, cls, from, to) => {
    const moved = h("span", { class: "ov-count" });
    ctx.folds(m.mem).push({ from, to, moved });
    return h("div", { class: `mem-line ${cls}`, style: { gridColumn: `1 / span ${n + 1}` } }, child, moved);
  };
  ctx.folds(m.mem).length = 0;

  for (const line of m.lines) {
    if (line.kind === "row") {
      el.append(...rowEls(line));
    } else if (line.kind === "empty") {
      const k = line.to - line.from + 1;
      el.append(full(k === 1 ? `${g.row_label} ${int(line.from)} empty` : `${g.row_label}s ${int(line.from)}–${int(line.to)} empty (${int(k)} ${g.row_label}s)`, "empty", line.from, line.to));
    } else {
      const key = `${ctx.detail.name}|${show}|${m.mem}|${line.from}`;
      const shown = expanded.get(key) ?? 0;
      const rest = line.hidden - shown;
      if (shown) {
        el.append(full(h("button", { type: "button", class: "mem-fold", onclick: () => { expanded.delete(key); redraw(); } },
          `▴ fold ${g.row_label}s ${int(line.from)}–${int(line.to)} again`), "fold open", -1, -2));
        const loaded = ctx.rows(m.mem);
        for (let r = line.from; r < line.from + shown; r++) if (loaded.has(r)) el.append(...rowEls(loaded.get(r)));
      }
      if (rest > 0) {
        const a = line.from + shown;
        const more = Math.min(rest, MAX_ROWS);
        const label = shown
          ? `▾ ${int(rest)} more ${g.row_label}${rest === 1 ? "" : "s"} (${int(a)}–${int(line.to)}); show ${more === rest ? "them" : `the next ${int(more)}`}`
          : `▾ ${g.row_label}${line.hidden === 1 ? "" : "s"} ${int(line.from)}${line.hidden === 1 ? "" : `–${int(line.to)}`} folded (${int(line.hidden)}): ${spanText(line)}; ${stepText(line)}${markText(line)}`;
        el.append(full(h("button", { type: "button", class: "mem-fold", onclick: async (e) => {
          e.currentTarget.disabled = true;
          try {
            await ctx.load(m.mem, a, a + more);
            expanded.set(key, shown + more);
          } catch (err) {
            ctx.status(String(err.message || err), true);
          }
          redraw();
        } }, label), "fold", a, line.to));
      }
    }
  }
  return h("div", { class: "scroll mem-grid-wrap" }, el);
}

function memorySection(ctx, m, colours) {
  const name = m.mem.toUpperCase();
  const used = m.regions.length
    ? `${m.regions.length} region${m.regions.length === 1 ? "" : "s"} use ${int(m.used.words)} of ${int(m.geometry.columns * m.geometry.rows)} words (${pct(m.used.words, m.geometry.columns * m.geometry.rows)}). Rows that repeat the row above with one step per region are folded: click a fold to show its rows, a word to follow its element.`
    : `No region lives in ${name}.`;
  const holder = h("div", {});
  const redraw = () => {
    holder.replaceChildren(grid(ctx, m, colours, redraw));
    ctx.redrawn();
  };
  holder.replaceChildren(grid(ctx, m, colours, redraw));
  const range = ctx.show && ctx.show !== "conflicts" ? ctx.range(m.mem) : null;
  return h("section", { id: `memory-${m.mem}` },
    h("h2", {}, name, h("span", { class: "kind" }, geometryText(m))),
    h("p", { class: "note" }, used),
    m.regions.length ? [scaleBar(m, colours), regionTable(m, colours)] : null,
    range ? h("div", { class: "time-key" }, h("span", {}, String(range[0])), h("i", { class: "time-ramp" }), h("span", {}, String(range[1])),
      h("small", {}, ctx.show === "wait" ? "cycles waited" : "cycle")) : null,
    holder);
}

// -- the control bar, the cycle overlay and the picked element -----------------------------

function controls(ctx, beat) {
  if (!beat) {
    return { el: h("section", { class: "mem-controls" }, h("p", { class: "note" },
      "Run with --trace beat for the selected cycle's accesses, the conflicts, arrival and use times, and each element's journey.")) };
  }
  const input = h("input", { type: "number", min: 0, max: ctx.total - 1, value: ctx.cycle ?? "", inputmode: "numeric", "aria-label": "Cycle",
    onchange: (e) => ctx.setHash({ cycle: Math.min(Math.max(Number.parseInt(e.target.value, 10) || 0, 0), ctx.total - 1) }) });
  const step = (d) => ctx.setHash({ cycle: Math.min(Math.max((current?.cycle ?? -1) + d, 0), ctx.total - 1) });
  const summary = h("p", { class: "note ov-summary" });
  const el = h("section", { class: "mem-controls" },
    h("div", { class: "mem-bar" },
      h("div", { class: "mem-modes", role: "tablist", "aria-label": "Colour the words by" },
        Object.entries(SHOWS).map(([k, label]) => h("button", { type: "button", role: "tab", "aria-selected": String(k === ctx.show),
          onclick: () => ctx.setHash({ show: k || null }) }, label))),
      h("div", { class: "cycle-head" },
        h("button", { type: "button", onclick: () => step(-1), "aria-label": "Previous cycle" }, "‹"),
        h("label", {}, "Cycle ", input),
        h("button", { type: "button", onclick: () => step(1), "aria-label": "Next cycle" }, "›"),
        h("button", { type: "button", onclick: () => ctx.setHash({ cycle: null }) }, "Clear"))),
    ctx.show ? h("p", { class: "note" }, SHOW_NOTES[ctx.show]) : null,
    h("ul", { class: "legend" },
      [["ov-req", "Requested in the cycle"], ["ov-resp", "Read data back in the cycle"], ["ov-held", "Held back in the cycle"], ["picked", "The picked element"]]
        .map(([c, label]) => h("li", {}, h("i", { class: `key outline ${c}` }), label))),
    summary);
  return { el, input, summary };
}

/** Where the words of an event lie: [{mem, row, col}]. */
function wordsOf(e, geo) {
  if (e.mem === "l1" && e.banks) return e.banks.map((b) => ({ mem: "l1", row: e.row, col: b })); // word-interleaved (open item 18)
  if (e.mem === "l2" && geo.l2) {
    const g = geo.l2;
    const word = Math.floor((e.addr - g.base_addr) / g.word_bytes);
    return Array.from({ length: g.columns }, (_, j) => ({ mem: "l2", row: Math.floor((word + j) / g.columns), col: (word + j) % g.columns }));
  }
  return [];
}

function describeEvent(e, c) {
  if (e.k === "stall") {
    const s = c && c.served ? `: bank ${c.bank} served ${c.served.port} for ${c.served.element ?? "a word"}` : "";
    return `${e.port} held back (${c?.waiting?.element ?? `addr ${e.addr}`} waited${s})`;
  }
  if (e.k === "resp") return `${e.port ?? e.src} read data back`;
  if (e.k === "dma_beat") return `${e.src} ${e.side === "src" ? "reads" : "writes"} L2 beat ${e.i}`;
  return `${e.port} ${e.w ? "write" : "read"} accepted`;
}

/** Draw the accesses of `cycle` over the grids (null clears them). */
async function applyCycle(cur, cycle) {
  cur.cycle = cycle;
  for (const el of cur.decorated) {
    el.classList.remove("ov-req", "ov-resp", "ov-held");
    el.title = el.dataset.tip ?? el.title;
  }
  cur.decorated = [];
  for (const mem of Object.keys(cur.foldsBy)) for (const f of cur.foldsBy[mem]) f.moved.textContent = "";
  if (cur.bar?.input) cur.bar.input.value = cycle ?? "";
  if (cycle === null || !cur.beat) {
    if (cur.bar?.summary) cur.bar.summary.textContent = cur.beat ? "Pick a cycle (or step with the arrow keys) to see which words move in it." : "";
    return;
  }
  const [events, conf] = await Promise.all([
    cur.api.events(cur.detail.name, cycle, cycle + 1, { k: OVERLAY_KINDS }),
    cur.api.conflicts(cur.detail.name, cycle, cycle + 1),
  ]);
  if (current !== cur || cur.cycle !== cycle) return;
  const stalls = new Map((conf.conflicts ?? []).map((c) => [`${c.bank}|${c.waiting.port}`, c]));
  const counts = { req: 0, resp: 0, held: 0 };
  const hidden = { l1: [], l2: [] };
  for (const e of events) {
    if (e.k === "dma_beat" && e.mem !== "l2") continue; // the xbar's grant carries the L1 side
    const kind = e.k === "stall" ? "held" : e.k === "resp" ? "resp" : "req";
    for (const w of wordsOf(e, cur.geo)) {
      counts[kind]++;
      const cell = cur.cellsBy[w.mem]?.get(`${w.row}|${w.col}`);
      const c = e.k === "stall" ? stalls.get(`${w.col}|${e.port}`) : null;
      if (!cell) {
        hidden[w.mem]?.push({ row: w.row, kind });
        continue;
      }
      cell.el.dataset.tip ??= cell.tip;
      cell.el.classList.add(`ov-${kind}`);
      cell.el.title = `${cell.el.title}\ncycle ${cycle}: ${describeEvent(e, c)}`;
      cur.decorated.push(cell.el);
    }
  }
  for (const mem of Object.keys(cur.foldsBy)) {
    for (const f of cur.foldsBy[mem]) {
      const inside = (hidden[mem] ?? []).filter((x) => x.row >= f.from && x.row <= f.to);
      if (!inside.length) continue;
      const held = inside.filter((x) => x.kind === "held").length;
      f.moved.textContent = ` · cycle ${cycle}: ${inside.length} word${inside.length === 1 ? "" : "s"} moved here${held ? `, ${held} held back` : ""}`;
      f.moved.classList.toggle("bad", held > 0);
    }
  }
  const lines = (conf.conflicts ?? []).slice(0, 4).map((c) =>
    `bank ${c.bank}: ${c.waiting.port} waited for ${c.waiting.element ?? `row ${c.waiting.row}`} while ${c.served ? `${c.served.port} ${c.served.w ? "wrote" : "read"} ${c.served.element ?? `row ${c.served.row}`}` : "the bank was busy"}`);
  const more = (conf.conflicts?.length ?? 0) - lines.length;
  if (cur.bar?.summary) {
    cur.bar.summary.replaceChildren(h("span", {},  // h() drops nulls and flattens arrays
      `Cycle ${int(cycle)}: ${counts.req} word${counts.req === 1 ? "" : "s"} requested, ${counts.resp} read back, ${counts.held} held back.`,
      lines.map((l) => h("span", { class: "ov-conflict" }, l)),
      more > 0 ? h("span", { class: "ov-conflict" }, `and ${more} more`) : null));
  }
}

function hopText(x) {
  const where = x.mem === "l1" ? `bank ${x.bank}, row ${x.row}` : `beat ${x.beat}`;
  const served = x.served ? `; bank served ${x.served.port} for ${x.served.element ?? `row ${x.served.row}`}` : "";
  return `${x.mem.toUpperCase()} ${x.act} by ${x.by} (${where})${served}`;
}

/** The drawer of the picked element: its hops, firings and conflicts. */
async function applyPick(cur, pick) {
  cur.pick = pick;
  for (const el of cur.picked) el.classList.remove("picked");
  cur.picked = [];
  document.body.classList.toggle("drawer-open", !!pick); // wide screens make room for the drawer
  if (!pick) {
    cur.drawer.hidden = true;
    return;
  }
  for (const mem of Object.keys(cur.cellsBy)) {
    for (const cell of cur.cellsBy[mem].values()) {
      if (cell.picks.includes(pick)) {
        cell.el.classList.add("picked");
        cur.picked.push(cell.el);
      }
    }
  }
  cur.drawer.hidden = false;
  const body = cur.drawer.querySelector(".drawer-body");
  if (!cur.beat) {
    body.replaceChildren(h("p", { class: "note" }, "The journey needs a beat-level trace: run with --trace beat."));
    return;
  }
  body.replaceChildren(h("p", { class: "note" }, "Loading…"));
  let j;
  try {
    j = await journeyOf(cur.api, cur.detail, pick);
  } catch (e) {
    body.replaceChildren(h("p", { class: "note" }, String(e.message || e)));
    return;
  }
  if (current !== cur || cur.pick !== pick) return;
  const go = (t) => h("button", { type: "button", class: "cyc", onclick: () => cur.setHash({ cycle: t }), title: "Show this cycle here" }, String(t));
  const sched = (t) => h("button", { type: "button", class: "cyc link", onclick: () => cur.setHash({ view: "schedule", cycle: t }), title: "Open the schedule at this cycle" }, "→ schedule");
  cur.drawer.querySelector(".drawer-title").textContent = j.element ?? pick;
  const traced = tracersOf(cur.st());
  const on = traced.includes(pick);
  const toggle = h("button", { type: "button", class: "trace-toggle", "aria-pressed": String(on),
    title: "A row in the schedule with every cycle this element is touched",
    onclick: () => cur.setHash({ tracers: (on ? traced.filter((x) => x !== pick) : [...traced, pick]).join(",") || null }) },
  on ? "✓ Traced in the schedule" : "+ Trace in the schedule");
  if (!j.available) {
    body.replaceChildren(h("p", { class: "note" }, j.reason));
    return;
  }
  const hopRows = j.hops.map((x) => h("tr", { class: x.act === "held back" ? "bad" : null }, h("td", {}, go(x.t)), h("td", {}, hopText(x))));
  const firing = (f) => h("div", { class: "fire" },
    h("div", {}, go(f.t), ` firing ${f.n} of ${f.acc} (task ${f.task}, lane ${f.lane}) ${f.role} it: `,
      Object.entries(f.inputs).map(([p, xs]) => `${p} ${xs.join(", ")}`).join("; "),
      Object.keys(f.outputs).length ? ` → ${Object.entries(f.outputs).map(([p, xs]) => `${p} ${xs.join(", ")}`).join("; ")}` : ""),
    f.results ? Object.entries(f.results).map(([el, hs]) => h("div", { class: "note" }, `${el}: `,
      hs.map((x, i) => [i ? " · " : "", go(x.t), ` ${x.mem.toUpperCase()} ${x.act} by ${x.by}`]))) : null);
  const words = (cur.conflicts?.words ?? []).filter((w) => w.element === j.element);
  const held = words.flatMap((w) => w.held);
  const won = words.flatMap((w) => w.won);
  body.replaceChildren(h("div", {}, // h() drops nulls and flattens arrays
    h("p", { class: "note" }, `In ${j.memories.map((m) => m.toUpperCase()).join(" and ")}. Click a cycle to show it here.`),
    h("p", {}, toggle, " ", h("button", { type: "button", class: "cyc link", onclick: () => cur.setHash({ view: "schedule" }) }, "open the schedule")),
    j.filtered ? h("p", { class: "note" }, "This beat trace is filtered (see the header): hops outside it are missing.") : null,
    h("h3", {}, "Hops"),
    hopRows.length ? h("table", { class: "journey" }, h("tbody", {}, hopRows)) : h("p", { class: "note" }, "No access to this element was traced."),
    j.firings.length ? [h("h3", {}, "Firings"), j.firings.map(firing)] : null,
    held.length || won.length ? [h("h3", {}, "Conflicts in L1"),
      held.length ? h("p", {}, `Held back in ${held.length} cycle${held.length === 1 ? "" : "s"}: `, held.map((t) => [go(t), " ", sched(t), " "])) : null,
      won.length ? h("p", {}, `Served while another port waited, in ${won.length} cycle${won.length === 1 ? "" : "s"}: `, won.map((t) => [go(t), " ", sched(t), " "])) : null] : null));
}

/** The journey of element `pick` ("A:5"), fetched once per run; shared with the schedule's tracer rows. */
export function journeyOf(api, detail, pick) {
  const cache = cacheOf(journeys, detail);
  if (!cache.has(pick)) {
    const [region, index] = pick.split(":");
    cache.set(pick, api.journey(detail.name, region, index));
  }
  return cache.get(pick);
}

/** The traced elements of the hash (`tracers=A:5,B:5`), in order, without repeats. */
export function tracersOf(st) {
  return [...new Set((st.tracers ?? "").split(",").filter(Boolean))];
}

function drawer(setHash) {
  return h("aside", { class: "mem-drawer", hidden: true, "aria-label": "Picked element" },
    h("div", { class: "drawer-head" }, h("b", { class: "drawer-title" }),
      h("button", { type: "button", onclick: () => setHash({ pick: null }), "aria-label": "Close" }, "×")),
    h("div", { class: "drawer-body" }));
}

function cycleOf(st, total) {
  if (st.cycle === undefined || st.cycle === "") return null;
  const c = Number.parseInt(st.cycle, 10);
  return Number.isFinite(c) ? Math.min(Math.max(c, 0), total - 1) : null;
}

/**
 * Draw the memory tab of `detail` into `root`.
 * ctx: {api, status(msg, err), st (hash state), setHash}.
 */
export async function renderMemory(root, detail, ctx) {
  const { api, st, setHash } = ctx;
  const beat = detail.trace?.level === "beat";
  const show = beat && st.show in SHOWS ? st.show : "";
  const total = detail.run.total_cycles;
  const plain = await memoryData(api, detail);
  let view = plain;
  let conflicts = null;
  if (beat) {
    if (!whole.has(detail)) whole.set(detail, api.conflicts(detail.name, 0, total + 1));
    conflicts = await whole.get(detail);
    if (show) {
      const cache = cacheOf(marked, detail);
      if (!cache.has(show)) cache.set(show, api.memory(detail.name, show));
      view = await cache.get(show);
    }
  }
  const s = stateOf(api, detail);
  const rowsFor = (mem) => {
    if (!show) return (s.rows[mem] ??= new Map());
    const cache = cacheOf(markedRows, detail);
    if (!cache.has(`${show}|${mem}`)) cache.set(`${show}|${mem}`, new Map());
    return cache.get(`${show}|${mem}`);
  };
  for (const m of view.memories) for (const line of m.lines) if (line.kind === "row") rowsFor(m.mem).set(line.row, line);
  const cellsBy = {};
  const foldsBy = {};
  const cur = {
    detail, api, setHash, beat, show, total, st: ctx.hashState, cycle: null, pick: null, decorated: [], picked: [], cellsBy, foldsBy, conflicts,
    geo: Object.fromEntries(view.memories.map((m) => [m.mem, m.geometry])),
  };
  const draw = {
    detail,
    show,
    total,
    conflicts,
    status: ctx.status,
    setHash,
    cells: (mem) => (cellsBy[mem] ??= new Map()),
    folds: (mem) => (foldsBy[mem] ??= []),
    range: (mem) => view.memories.find((m) => m.mem === mem)?.mark_range ?? null,
    rows: rowsFor,
    load: async (mem, from, to) => {
      const got = await api.memoryRows(detail.name, mem, from, to, show || undefined);
      for (const line of got.rows) rowsFor(mem).set(line.row, line);
    },
    redrawn: () => { // a fold opened or closed: draw the cycle and the pick again on the new cells
      applyCycle(cur, cur.cycle);
      applyPick(cur, cur.pick);
    },
  };
  const colours = new Map();
  for (const m of plain.memories) for (const r of m.regions) colourOf(colours, r.name);
  const intro = view.has_regions
    ? null
    : h("section", {}, h("h2", {}, "Memory"),
      h("p", { class: "note" }, "This run names no regions, so only the memories' geometry is shown. The flow writes regions from the memory plan; a hand-written scenario may declare them (CONTRACTS.md section 6)."));
  const bar = view.has_regions ? controls({ ...draw, cycle: cycleOf(st, total) }, beat) : null;
  cur.bar = bar;
  cur.drawer = drawer(setHash);
  root.replaceChildren(...[intro, bar?.el, ...view.memories.map((m) => memorySection(draw, m, colours)), cur.drawer].filter(Boolean));
  current = cur;
  cur.tracers = st.tracers ?? "";
  await applyCycle(cur, cycleOf(st, total));
  await applyPick(cur, st.pick ?? null);
}

/**
 * A change of the hash that leaves the drawn grids as they are (the cycle or the
 * picked element): update in place. Returns false when a full render is needed.
 */
export async function updateMemory(detail, st) {
  const cur = current;
  if (!cur || cur.detail !== detail || (cur.beat && st.show in SHOWS ? st.show : "") !== cur.show) return false;
  const cycle = cycleOf(st, cur.total);
  if (cycle !== cur.cycle) await applyCycle(cur, cycle);
  if ((st.pick ?? null) !== cur.pick || (st.tracers ?? "") !== cur.tracers) {
    cur.tracers = st.tracers ?? "";
    await applyPick(cur, st.pick ?? null);
  }
  return true;
}

/** The memory tab's cycle moved by d, or null when there is none to step (arrow keys). */
export function stepMemoryCycle(d) {
  if (!current || !current.beat) return null;
  return Math.min(Math.max((current.cycle ?? -1) + d, 0), current.total - 1);
}
