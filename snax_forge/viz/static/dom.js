// Small DOM and number helpers shared by the viewer modules (VIS1).
// Everything is built with textContent, never innerHTML, so names from a
// run directory cannot inject markup.

/** Create an element: h("td", {class: "num"}, "12"). Children may be nested arrays or null. */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "style" && typeof v === "object") Object.assign(el.style, v);
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
