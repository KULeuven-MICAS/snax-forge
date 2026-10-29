// Small DOM and number helpers shared by the viewer modules.
// Everything is built with textContent, never innerHTML, so names from a
// run directory cannot inject markup.

/** Colour group of each cycle class (style.css .g-* / .s-*): memory stalls warm, flow stalls violet. */
export const CLASS_GROUP = {
  busy: "busy", idle: "idle",
  stall_xbar: "mem", stall_l1: "mem",
  stall_fifo: "flow", stall_in: "flow", stall_out: "flow", stall_mem: "flow", flush: "flow",
  command: "command", wait: "wait",
  run: "busy", poll: "wait", barrier: "command", sleep: "idle", // a core's (D137)
};

/** Create an element: h("td", {class: "num"}, "12"). Children may be nested arrays or null. */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "style" && typeof v === "object") {
      for (const [prop, val] of Object.entries(v)) { // custom properties (--x) need setProperty
        if (prop.startsWith("--")) el.style.setProperty(prop, String(val));
        else el.style[prop] = val;
      }
    }
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

/** Integer with thin grouping. */
export const int = (n) => (n === null || n === undefined ? "–" : Number(n).toLocaleString("en-US"));

/** Percentage of part in whole, one decimal. */
export const pct = (part, whole) => (whole ? `${((100 * part) / whole).toFixed(1)}%` : "–");

/** A cycle kept inside a run of `total` cycles: 0 .. total - 1. */
export const clampCycle = (c, total) => Math.min(Math.max(c, 0), total - 1);

/** Fixed decimals for means. */
export const dec = (x, d = 2) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));

/** A table: cols = [{key, label, num, fmt}], rows = objects; rowClass(row) optional. */
export function table(cols, rows, rowClass = null) {
  return h("div", { class: "scroll" },
    h("table", {},
      h("thead", {}, h("tr", {}, cols.map((c) => h("th", { class: c.num ? "num" : null }, c.label)))),
      h("tbody", {}, rows.map((r) =>
        h("tr", { class: rowClass ? rowClass(r) : null },
          cols.map((c) => h("td", { class: c.num ? "num" : null }, c.fmt ? c.fmt(r[c.key], r) : r[c.key])))))));
}

/** A section with a heading and a one-line note under it. */
export function section(id, title, note, ...body) {
  return h("section", { id },
    h("h2", {}, title),
    note ? h("p", { class: "note" }, note) : null,
    body);
}

/** The units of an accelerator.json (file): each with its group, in group order. */
export function accelUnits(file) {
  return (file?.groups ?? []).flatMap((g) => g.units.map((u) => ({ group: g.name, unit: u })));
}

/** The units a run's accelerator name stands for: a linked group's (SNAX-GVSoC's one
 * component, named after the group), else the unit itself (report/design.py's members). */
export function unitsOf(file, name) {
  const linked = (file?.groups ?? []).find((g) => g.name === name && g.units.length > 1);
  return accelUnits(file).filter((x) => (linked ? x.group === name : x.unit.name === name));
}

/** One unit's facts, from accelerator.json: its group, datapath, op, input lanes and timing. */
export function unitFacts({ group, unit: u }) {
  const ins = Object.values(u.ports ?? {}).filter((p) => p.dir === "in");
  const t = u.timing ?? {};
  return [`${group}: ${u.name} ${u.datapath}`, u.params?.op !== undefined ? `op ${u.params.op}` : null,
    ins.length ? `${ins[0].lanes} lanes` : null, `latency ${t.latency}`, `II ${t.ii}`].filter(Boolean).join(", ");
}
