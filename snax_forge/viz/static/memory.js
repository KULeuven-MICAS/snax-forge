// The memory tab (VIS4a, D95, D96): where a run's regions sit in each memory.
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
// colour in every memory, so A in L2 and A in L1 match.
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

const expanded = new Map(); // "run|mem|from" -> number of hidden rows shown; kept until the page reloads

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

/** The grid of one memory; `redraw` rebuilds it after a fold is opened or closed. */
function grid(ctx, m, colours, redraw) {
  const g = m.geometry;
  const n = g.columns;
  const labelAll = n <= LABEL_EVERY_CELL;
  const el = h("div", { class: "mem-grid", style: { gridTemplateColumns: `4.2rem repeat(${n}, minmax(${labelAll ? "1.7rem" : "1.1rem"}, 1fr))` } });
  const sb = (c) => (c > 0 && c % g.group === 0 ? " sb-first" : "");
  const head = (c) => (labelAll || c % g.group === 0 ? String(c) : ""); // many columns: superbank starts only
  el.append(h("div", { class: "mem-corner" }, g.row_label),
    ...Array.from({ length: n }, (_, c) => h("div", { class: `mem-col${sb(c)}`, title: `${g.column_label} ${c}` }, head(c))));

  const rowEls = (line) => {
    const out = [h("div", { class: "mem-row-label", title: `${g.row_label} ${line.row}: bytes ${int(line.addr[0])}… (${g.column_label} 0)` },
      String(line.row), h("small", {}, `@${int(line.addr[0])}`))];
    let prev = null;
    line.cells.forEach((cell, c) => {
      const tip = [`${g.column_label} ${c}, ${g.row_label} ${line.row}, byte ${int(line.addr[c])}`];
      if (!cell.length) {
        out.push(h("div", { class: `mem-cell empty${sb(c)}`, title: `${tip[0]}: empty` }));
        prev = null;
        return;
      }
      const [ri, flat] = cell[0];
      const r = m.regions[ri];
      for (const [rj, fj] of cell) tip.push(`${elemText(m.regions[rj], fj)}, flat index ${fj}`);
      const cont = prev && prev[0] === ri && prev[1] + 1 === flat;
      const text = labelAll || !cont ? String(flat) : "";
      out.push(h("div", { class: `mem-cell r${colourOf(colours, r.name)}${cell.length > 1 ? " multi" : ""}${sb(c)}`, title: tip.join("\n") }, text));
      prev = [ri, flat];
    });
    return out;
  };

  const spanText = (line) => Object.entries(line.span)
    .map(([ri, [a, b]]) => `${m.regions[ri].name}[${a}..${b}]`).join(", ");
  const stepText = (line) => Object.entries(line.step)
    .map(([ri, d]) => `${m.regions[ri].name} ${d >= 0 ? "+" : "−"}${Math.abs(d)} per ${g.row_label}`).join(", ");

  const full = (child, cls) => h("div", { class: `mem-line ${cls}`, style: { gridColumn: `1 / span ${n + 1}` } }, child);

  for (const line of m.lines) {
    if (line.kind === "row") {
      el.append(...rowEls(line));
    } else if (line.kind === "empty") {
      const k = line.to - line.from + 1;
      el.append(full(k === 1 ? `${g.row_label} ${int(line.from)} empty` : `${g.row_label}s ${int(line.from)}–${int(line.to)} empty (${int(k)} ${g.row_label}s)`, "empty"));
    } else {
      const key = `${ctx.detail.name}|${m.mem}|${line.from}`;
      const shown = expanded.get(key) ?? 0;
      const rest = line.hidden - shown;
      if (shown) {
        el.append(full(h("button", { type: "button", class: "mem-fold", onclick: () => { expanded.delete(key); redraw(); } },
          `▴ fold ${g.row_label}s ${int(line.from)}–${int(line.to)} again`), "fold open"));
        const loaded = ctx.rows(m.mem);
        for (let r = line.from; r < line.from + shown; r++) if (loaded.has(r)) el.append(...rowEls(loaded.get(r)));
      }
      if (rest > 0) {
        const a = line.from + shown;
        const more = Math.min(rest, MAX_ROWS);
        const label = shown
          ? `▾ ${int(rest)} more ${g.row_label}${rest === 1 ? "" : "s"} (${int(a)}–${int(line.to)}); show ${more === rest ? "them" : `the next ${int(more)}`}`
          : `▾ ${g.row_label}${line.hidden === 1 ? "" : "s"} ${int(line.from)}${line.hidden === 1 ? "" : `–${int(line.to)}`} folded (${int(line.hidden)}): ${spanText(line)}; ${stepText(line)}`;
        el.append(full(h("button", { type: "button", class: "mem-fold", onclick: async (e) => {
          e.currentTarget.disabled = true;
          try {
            await ctx.load(m.mem, a, a + more);
            expanded.set(key, shown + more);
          } catch (err) {
            ctx.status(String(err.message || err), true);
          }
          redraw();
        } }, label), "fold"));
      }
    }
  }
  return h("div", { class: "scroll mem-grid-wrap" }, el);
}

function memorySection(ctx, m, colours) {
  const name = m.mem.toUpperCase();
  const used = m.regions.length
    ? `${m.regions.length} region${m.regions.length === 1 ? "" : "s"} use ${int(m.used.words)} of ${int(m.geometry.columns * m.geometry.rows)} words (${pct(m.used.words, m.geometry.columns * m.geometry.rows)}). Rows that repeat the row above with one step per region are folded: click a fold to show its rows.`
    : `No region lives in ${name}.`;
  const holder = h("div", {});
  const redraw = () => holder.replaceChildren(grid(ctx, m, colours, redraw));
  redraw();
  return h("section", { id: `memory-${m.mem}` },
    h("h2", {}, name, h("span", { class: "kind" }, geometryText(m))),
    h("p", { class: "note" }, used),
    m.regions.length ? [scaleBar(m, colours), regionTable(m, colours)] : null,
    holder);
}

/**
 * Draw the memory tab of `detail` into `root`.
 * ctx: {api, status(msg, err)}.
 */
export async function renderMemory(root, detail, ctx) {
  const { api } = ctx;
  const view = await memoryData(api, detail);
  const s = stateOf(api, detail);
  for (const m of view.memories) keep(s, m.mem, m.lines);
  const draw = {
    detail,
    status: ctx.status,
    rows: (mem) => s.rows[mem] ?? new Map(),
    load: async (mem, from, to) => keep(s, mem, (await api.memoryRows(detail.name, mem, from, to)).rows),
  };
  const colours = new Map();
  for (const m of view.memories) for (const r of m.regions) colourOf(colours, r.name);
  const intro = view.has_regions
    ? null
    : h("section", {}, h("h2", {}, "Memory"),
      h("p", { class: "note" }, "This run names no regions, so only the memories' geometry is shown. The flow writes regions from the memory plan; a hand-written scenario may declare them (CONTRACTS.md section 6)."));
  root.replaceChildren(...[intro, ...view.memories.map((m) => memorySection(draw, m, colours))].filter(Boolean));
}
