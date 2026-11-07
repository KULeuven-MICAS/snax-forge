// Edit mode of the DFG page (D55, VIEW2): a session (sandbox/session.py) behind the edit
// routes of viz/dfg/server.py.
//
// The start dialog opens or starts a session; the top bar holds its name, the history, undo and
// redo, Export and Save (the params are in the left column, with what they configure); the left sidebar lists the transforms
// whose site accepts the selection, each with the form the API gives (its fields, types,
// defaults and choices), and Apply; the log at the bottom has a line per action, written by
// the server, and the working recipe. Nothing is computed here (D57): every value, choice
// and message comes from the API, and after a change the page asks for the session again.

import { dragHandle, h, keep, stored } from "./dom.js";

const $ = (id) => document.getElementById(id);
const enc = encodeURIComponent;

let ctx = null; // dfg.js's state and redraw (init)

/** A POST with a JSON body: ``{ok, status, body}``, a refusal's body too. */
export async function post(url, body = {}) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return { ok: r.ok, status: r.status, body: await r.json() };
}

async function get(url) {
  const r = await fetch(url);
  return { ok: r.ok, status: r.status, body: await r.json() };
}

// -- the log --------------------------------------------------------------------------

export function log(line, error = false) {
  if (!line) return;
  const list = $("log-lines");
  list.append(h("li", { class: error ? "error" : null }, linked(line)));
  list.scrollTop = list.scrollHeight;
}

/** A log line with its design checks' counts (``1 problem``, ``2 notes``, D156) as links to
 * the list under the Estimate. */
function linked(line) {
  const parts = [];
  let at = 0;
  for (const m of line.matchAll(/\b(\d+) (problems?|notes?)\b/g)) {
    parts.push(line.slice(at, m.index));
    const target = m[2].startsWith("problem") ? "checks-problems" : "checks-notes";
    parts.push(h("a", { href: `#${target}`, class: "log-check", onclick: (ev) => {
      ev.preventDefault();
      const el = $(target) || $("checks");
      if (el) el.scrollIntoView({ block: "nearest" });
    } }, m[0]));
    at = m.index + m[0].length;
  }
  parts.push(line.slice(at));
  return parts;
}

// The pane's height: dragged on its handle between three lines and 70% of the window (the
// CSS's min-height and max-height), kept in localStorage; a double-click collapses or reopens it.
const LOG_KEY = "snax-forge.dfg.log-height";

function collapse(on) {
  const pane = $("log");
  pane.classList.toggle("collapsed", on);
  $("log-toggle").setAttribute("aria-expanded", on ? "false" : "true");
}

function logPane() {
  const pane = $("log");
  const handle = $("log-handle");
  const kept = stored(LOG_KEY);
  if (kept) pane.style.height = `${kept}px`;
  dragHandle(handle, {
    delta: (dx, dy) => -dy, // up is taller
    measure: () => pane.getBoundingClientRect().height,
    place: (px) => {
      collapse(false);
      const css = getComputedStyle(pane);
      pane.style.height = `${Math.min(parseFloat(css.maxHeight), Math.max(parseFloat(css.minHeight), px))}px`;
    },
    done: () => keep(LOG_KEY, Math.round(pane.getBoundingClientRect().height)),
  });
  handle.addEventListener("dblclick", () => collapse(!pane.classList.contains("collapsed")));
  $("log-toggle").addEventListener("click", () => collapse(!pane.classList.contains("collapsed")));
}

function recipePanel() {
  const s = ctx.state.session;
  $("recipe").replaceChildren(
    h("p", {}, h("code", {}, s.command)),
    h("pre", { class: "mono" }, JSON.stringify(s.recipe, null, 1)));
}

// -- the session ------------------------------------------------------------------------

/** The session, or null before one is open. */
export async function session() {
  const r = await get("/api/session");
  return r.ok ? r.body : null;
}

/** After a change: the session again, every cached view dropped, the page redrawn. */
async function changed() {
  ctx.state.session = await session();
  ctx.clear();
  await ctx.show();
}

/** The top bar's edit parts from the session. */
export function top(s) {
  $("session-name").textContent = s.name;
  $("undo").disabled = s.undo === null;
  $("redo").disabled = s.redo === null;
  $("recipe").hidden || recipePanel();
}

/** A param set from its input in the left column (a symbol or a binding): every step replayed;
a refusal keeps the old value. */
export async function setParam(name, input) {
  const raw = input.value.trim();
  const value = /^-?\d+$/.test(raw) ? Number(raw) : raw;
  ctx.status("replaying…");
  const r = await post("/api/params", { params: { [name]: value } });
  log(r.body.log, !r.ok);
  ctx.status(r.ok ? "" : r.body.message, !r.ok);
  if (!r.ok) input.value = String(ctx.state.session.params[name]); // the old value stays
  await changed();
}

export function undo() {
  if (ctx.state.session && ctx.state.session.undo !== null) move(ctx.state.session.undo);
}

export function redo() {
  if (ctx.state.session && ctx.state.session.redo !== null) move(ctx.state.session.redo);
}

export async function move(k) {
  const r = await post("/api/cursor", { k });
  log(r.body.log, !r.ok);
  await changed();
}

// -- the start dialog ------------------------------------------------------------------

function startStatus(msg, err = false) {
  $("start-status").textContent = msg || "";
  $("start-status").classList.toggle("error", err);
}

async function open(body, what) {
  startStatus(what);
  const r = await post("/api/session", body);
  if (!r.ok) {
    startStatus(`${r.body.code}: ${r.body.message}`, true);
    return;
  }
  $("start").close();
  ctx.state.session = r.body;
  log(`session ${r.body.name} opened: ${r.body.kernel}, ${r.body.n} steps`);
  ctx.clear();
  await ctx.show();
}

function kernelForm(kernels) {
  const pick = h("select", { "aria-label": "kernel" },
    kernels.map((k) => h("option", { value: k.name, disabled: !!k.error, title: k.error || "" }, k.error ? `${k.name} (does not load)` : k.name)));
  const rows = h("tbody", {});
  const row = (name, value = "", fixed = false) => h("tr", {},
    h("td", {}, fixed ? h("b", { "data-name": name }, name) : h("input", { placeholder: "name", size: 6, "data-name": "" })),
    h("td", {}, h("input", { placeholder: fixed ? "value" : "value", size: 8, value })));
  const fill = () => {
    const k = kernels.find((x) => x.name === pick.value);
    rows.replaceChildren(...(k ? k.symbols : []).map((s) => row(s, "", true)));
  };
  pick.addEventListener("change", fill);
  fill();
  const platform = h("input", { value: ctx.state.platform || "", size: 32, "aria-label": "platform" });
  const params = () => {
    const out = {};
    for (const tr of rows.children) {
      const nameEl = tr.querySelector("[data-name]");
      const name = nameEl.tagName === "B" ? nameEl.textContent : nameEl.value.trim();
      const raw = tr.querySelector("td:last-child input").value.trim();
      if (name && raw !== "") out[name] = /^-?\d+$/.test(raw) ? Number(raw) : raw;
    }
    return out;
  };
  return h("div", {},
    h("p", {}, pick),
    h("table", { class: "dfg-params-table" }, h("thead", {}, h("tr", {}, h("th", {}, "param"), h("th", {}, "value"))), rows),
    h("p", {}, h("button", { type: "button", onclick: () => rows.append(row("")) }, "Add param")),
    h("p", {}, h("label", {}, "platform ", platform)),
    h("p", {}, h("button", { type: "button", class: "primary",
      onclick: () => open({ kernel: pick.value, params: params(), platform: platform.value || undefined },
        `importing ${pick.value}… (DaCe builds its SDFG, a few seconds)`) }, "Start")));
}

/** The recipes as one dropdown, an optgroup per folder (``recipes/<folder>/``), in the order
the server lists them; a recipe that does not load is listed, disabled, with its error. */
function recipeForm(recipes) {
  const groups = [];
  recipes.forEach((r, i) => {
    const opt = h("option", { value: i, disabled: !!r.error, title: r.error || r.path }, r.error ? `${r.name} (does not load)` : r.name);
    const last = groups[groups.length - 1];
    if (last && last.folder === r.folder) last.options.push(opt);
    else groups.push({ folder: r.folder, options: [opt] });
  });
  const pick = h("select", { "aria-label": "recipe" }, groups.map((g) => h("optgroup", { label: g.folder || "(no folder)" }, g.options)));
  const first = recipes.findIndex((r) => !r.error);
  if (first >= 0) pick.value = String(first);
  const about = h("small", {});
  const chosen = () => recipes[Number(pick.value)];
  const fill = () => {
    const r = chosen();
    about.textContent = !r ? "" : r.error ? ` ${r.error}` : ` ${r.kernel} · ${r.steps} steps`;
  };
  pick.addEventListener("change", fill);
  fill();
  return h("div", {}, h("p", {}, pick, about),
    h("p", {}, h("button", { type: "button", class: "primary",
      onclick: () => { const r = chosen(); if (r && !r.error) open({ recipe: r.path }, `replaying ${r.name}…`); } }, "Open")));
}

export async function start() {
  $("start").showModal();
  startStatus("loading the kernels…");
  const [kernels, recipes, sessions] = await Promise.all([get("/api/kernels"), get("/api/recipes"), get("/api/sessions")]);
  startStatus("");
  $("start-kernel").replaceChildren(kernelForm(kernels.body));
  $("start-recipe").replaceChildren(recipes.body.length ? recipeForm(recipes.body) : h("p", { class: "muted" }, "no recipe"));
  $("start-session").replaceChildren(sessions.body.length
    ? h("ul", { class: "dfg-pick" }, sessions.body.map((x) => h("li", {},
      h("button", { type: "button", disabled: !!x.error, title: x.error || "",
        onclick: () => open({ session: x.name }, `resuming ${x.name}…`) }, x.name),
      h("small", {}, x.error ? ` ${x.error}` : ` ${x.kernel} · ${x.steps} steps · at ${x.cursor} · ${x.date}`))))
    : h("p", { class: "muted" }, "no session yet"));
}

// -- the sidebar: what applies to the selection ------------------------------------------

/** A form whose bind list has nothing that fits: Apply would only be refused. */
const unfit = (form) => form.some((f) => f.name === "binding" && !f.choices.some((c) => c.ok));

function field(f) {
  const attrs = { "data-field": f.name, disabled: f.readonly };
  if (f.type === "choice" && f.name === "binding") { // bind: the options that fit, with their timing
    const fit = f.choices.map((c, i) => [c, i]).filter(([c]) => c.ok); // the index is the API's
    if (!fit.length) { // nothing fits: the refusals, so the fix each names is still there
      return h("div", { class: "dfg-choices", ...attrs }, h("p", { class: "muted" }, "no BRM in the library fits"),
        h("details", {}, h("summary", {}, `why (${f.choices.length})`),
          f.choices.map((c) => h("p", { class: "muted" }, `${c.label}: ${c.message}`))));
    }
    return h("div", { class: "dfg-choices", ...attrs }, fit.map(([c, i]) => {
      const t = c.timing;
      const timing = t ? `II ${t.ii} · latency ${t.latency} · span ${t.span}` : "";
      return h("label", { class: "dfg-choice" },
        h("input", { type: "radio", name: "binding", value: String(i),
          checked: f.value && c.value.brm === f.value.brm && c.value.implementation === f.value.implementation }),
        h("span", {}, c.label), h("small", {}, timing));
    }));
  }
  if (f.type === "choice") {
    return h("select", attrs, f.choices.map((c) => h("option", { value: c.value, selected: c.value === f.value }, c.label)));
  }
  if (f.type === "checklist") {
    const ticked = new Set(f.value || []);
    return h("div", { class: "dfg-choices", ...attrs }, f.choices.map((c) => h("label", { class: "dfg-choice" },
      h("input", { type: "checkbox", value: c.value, checked: ticked.has(c.value) }), h("span", {}, c.label))));
  }
  const input = h("input", { ...attrs, value: f.value === null || f.value === undefined ? "" : String(f.value),
    title: f.sweepable ? "a number, or an expression over the params (W, W * 2); a new name gets its value beside it" : f.hint || "" });
  return f.sweepable && !f.readonly ? sweepable(input, f.new_value) : input;
}

// A name in an int field: a param, or one of the grammar's functions (expr.py)
const NAMES = /[A-Za-z_][A-Za-z0-9_]*/g;
const FUNCTIONS = new Set(["max", "min"]);

/** A sweepable int field (a split's factor, a link's depth): its default a param name of its
own (``W``, ``D``), and beside it a value input per name it uses that the session does not have
yet (its ``new_value``, 4 or 1), which the apply sends as ``new_params`` (C§12); a number typed
instead becomes a param server-side. The values typed stay while the field changes. */
function sweepable(input, start = 4) {
  const extra = h("span", { class: "dfg-new-params" });
  const typed = {};
  const refresh = () => {
    const known = (ctx.state.session && ctx.state.session.params) || {};
    for (const el of extra.querySelectorAll("[data-new-param]")) typed[el.dataset.newParam] = el.value;
    const names = [...new Set(input.value.match(NAMES) || [])]
      .filter((n) => !FUNCTIONS.has(n) && !Object.prototype.hasOwnProperty.call(known, n));
    extra.replaceChildren(...names.map((n) => h("label", { class: "dfg-new-param", title: `${n} is not a param yet: its value, added to the session's params` },
      `new param ${n} = `, h("input", { "data-new-param": n, value: typed[n] ?? String(start), size: 4, inputmode: "numeric" }))));
  };
  input.addEventListener("input", refresh);
  refresh();
  return h("span", { class: "dfg-sweep" }, input, extra);
}

/** The new params a form's fields give (``sweepable``): name -> its value, an int. */
function newParams(el) {
  const out = {};
  for (const x of el.querySelectorAll("[data-new-param]")) {
    const raw = x.value.trim();
    if (/^-?\d+$/.test(raw)) out[x.dataset.newParam] = Number(raw); // else the server names it
  }
  return out;
}

/** The params a form gives, as the user left its fields. */
function params(form, el) {
  const out = {};
  for (const f of form) {
    const node = el.querySelector(`[data-field="${CSS.escape(f.name)}"]`);
    if (f.name === "binding") {
      const picked = node.querySelector("input:checked");
      if (picked) Object.assign(out, f.choices[Number(picked.value)].value);
    } else if (f.type === "checklist") {
      out[f.name] = [...node.querySelectorAll("input:checked")].map((x) => x.value);
    } else if (f.type === "int") {
      const raw = node.value.trim();
      out[f.name] = /^-?\d+$/.test(raw) ? Number(raw) : raw;
    } else {
      out[f.name] = node.value;
    }
  }
  return out;
}

/** Apply at the cursor; below the last step, the server's question first (``drop``), then the
steps after the cursor dropped. A factor's new param goes as ``new_params`` (``sweepable``). */
async function apply(transform, form, el) {
  const question = ctx.state.session.drop;
  if (question && !window.confirm(question)) return;
  const fresh = newParams(el);
  const body = { transform, params: params(form, el), drop: !!question, ...(Object.keys(fresh).length ? { new_params: fresh } : {}) };
  ctx.status(`applying ${transform}…`);
  const r = await post("/api/apply", body);
  log(r.body.log, !r.ok);
  ctx.status(r.ok ? "" : `${r.body.code}: ${r.body.message}`, !r.ok);
  if (r.ok) {
    ctx.state.t = null;
    await changed();
  }
}

/** The open form closed (Esc). */
export function closeForm() {
  ctx.state.t = null;
  ctx.writeHash();
  sidebar();
}

/** The transforms for the selection at step k (the left column's top); with Compare on, a note
(a transform applies to one step). */
export async function sidebar() {
  const st = ctx.state;
  const bar = $("transforms");
  if (st.cmp !== null) {
    bar.replaceChildren(h("h2", {}, "Transforms"), h("p", { class: "muted" }, "Compare is on: turn it off to apply a transform to the step shown."));
    return;
  }
  const sels = st.sels && st.sels.length ? st.sels : st.sel ? [st.sel] : [];
  if (!sels.length) {
    bar.replaceChildren(h("h2", {}, "Transforms"), h("p", { class: "muted" }, "Select a node, a container or a dashed group or accelerator outline: the transforms that apply there are listed here; shift-click adds to the selection. A dot marks where one applies."));
    return;
  }
  const v = await ctx.view(st.k);
  const kindsOf = (id) => ["node", "container", "group", "accelerator", "edge"].filter((kind) => v.details && v.details[`${kind}:${id}`]);
  const kinds = kindsOf(sels[0]);
  // the selection's sites (D72): one route for one item or several, in the order selected
  const r = await get(`/api/step/${st.k}/site?${sels.map((x) => `sel=${enc(x)}`).join("&")}`);
  const answers = r.ok ? r.body : {};
  const ok = Object.entries(answers).filter(([, a]) => a.ok);
  const no = Object.entries(answers).filter(([, a]) => !a.ok);
  const rows = ok.map(([name, a]) => {
    const isOpen = st.t === name;
    const formEl = isOpen ? h("form", { class: "dfg-form", onsubmit: (ev) => { ev.preventDefault(); apply(name, a.form, formEl); } },
      a.form.map((f) => h("label", { class: `dfg-field${f.readonly ? " readonly" : ""}` }, h("span", {}, f.name), field(f))),
      h("button", { type: "submit", class: "primary", disabled: unfit(a.form) }, "Apply")) : null;
    return h("div", { class: `dfg-transform${isOpen ? " open" : ""}` },
      h("button", { type: "button", class: "dfg-transform-name", "aria-expanded": isOpen ? "true" : "false",
        onclick: () => {
          if (isOpen) return closeForm();
          st.t = name;
          ctx.writeHash();
          return sidebar();
        } }, name),
      formEl);
  });
  bar.replaceChildren(...[
    sels.length === 1
      ? h("h2", {}, kinds.join(" · ") || "item", " ", h("b", {}, sels[0]))
      : h("h2", {}, `${sels.length} selected: `, sels.map((x, i) => [i ? ", " : "", `${kindsOf(x)[0] || "item"} `, h("b", {}, x)])),
    ...(rows.length ? rows : [h("p", { class: "muted" }, "No transform applies here.")]),
    no.length ? h("details", { class: "dfg-not-here" }, h("summary", {}, `Not here (${no.length})`),
      h("ul", {}, no.map(([name, a]) => h("li", {}, h("b", {}, name), " ", h("small", {}, a.message))))) : null,
  ].filter(Boolean)); // replaceChildren would write a null as "null"
}

// -- the graph's marks and diff ---------------------------------------------------------------

/** A dot on every item some site accepts, a highlight on what the step added or changed. */
export function decorate(v, root) {
  const marked = new Set([...(v.marks ? v.marks.nodes : []), ...(v.marks ? v.marks.containers : [])]);
  const d = v.diff || { nodes: { added: [], changed: [] }, containers: { added: [], changed: [] } };
  const diffed = new Set([...d.nodes.added, ...d.nodes.changed, ...d.containers.added, ...d.containers.changed]);
  for (const el of root.querySelectorAll("[data-sel]")) {
    const name = el.getAttribute("data-sel");
    if (el.namespaceURI !== "http://www.w3.org/1999/xhtml") continue; // edges and outlines stay plain
    el.classList.toggle("marked", marked.has(name));
    el.classList.toggle("diffed", diffed.has(name));
  }
}

// -- the buttons ------------------------------------------------------------------------------

function saveAnswer(body, ok) {
  const out = [];
  if (ok) {
    out.push(h("p", {}, `saved ${body.shown}`), h("pre", { class: "mono" },
      body.commands.map((c) => `${c.command}\n    ${c.what}`).join("\n")));
  } else {
    out.push(h("p", { class: "error" }, `${body.code}: ${body.message}`));
    if (body.problems) out.push(h("pre", { class: "mono" }, body.problems.join("\n")));
  }
  $("save-answer").replaceChildren(...out);
  $("save-force").hidden = ok || body.code !== "save.design";
}

async function save(force) {
  const name = $("save-name").value.trim();
  const r = await post("/api/save", { name, force });
  log(r.body.log, !r.ok);
  if (r.ok) for (const c of r.body.commands) log(`  ${c.command}  · ${c.what}`);
  saveAnswer(r.body, r.ok);
}

export function init(c) {
  ctx = c;
  for (const el of document.querySelectorAll("[data-edit]")) el.hidden = false;
  $("reload").hidden = true;
  $("undo").addEventListener("click", undo);
  $("redo").addEventListener("click", redo);
  $("export").addEventListener("click", async () => {
    const r = await post("/api/export", { k: ctx.state.k });
    log(r.body.log, !r.ok);
  });
  $("save").addEventListener("click", () => {
    $("save-answer").replaceChildren();
    $("save-force").hidden = true;
    $("save-dialog").showModal();
  });
  $("save-form").addEventListener("submit", (ev) => { ev.preventDefault(); save(false); });
  $("save-force").addEventListener("click", () => save(true));
  $("save-close").addEventListener("click", () => $("save-dialog").close());
  logPane();
  $("recipe-toggle").addEventListener("click", (ev) => {
    $("recipe").hidden = !$("recipe").hidden;
    ev.target.setAttribute("aria-expanded", $("recipe").hidden ? "false" : "true");
    if (!$("recipe").hidden) recipePanel();
  });
}
