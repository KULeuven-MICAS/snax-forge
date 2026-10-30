// Edit mode of the DFG page (D55, VIEW2): a session (sandbox/session.py) behind the edit
// routes of viz/dfg/server.py.
//
// The start dialog opens or starts a session; the top bar holds its name, its params as
// chips, the history, undo and redo, Export and Save; the left sidebar lists the transforms
// whose site accepts the selection, each with the form the API gives (its fields, types,
// defaults and choices), and Apply; the log at the bottom has a line per action, written by
// the server, and the working recipe. Nothing is computed here (D57): every value, choice
// and message comes from the API, and after a change the page asks for the session again.

import { h } from "./dom.js";

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
  list.append(h("li", { class: error ? "error" : null }, line));
  list.scrollTop = list.scrollHeight;
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
  $("params").replaceChildren(...Object.entries(s.params).map(([name, value]) =>
    h("label", { class: "dfg-chip", title: `param ${name}: a change replays every step` }, name,
      h("input", { value: String(value), size: Math.max(2, String(value).length), "data-param": name,
        onchange: (ev) => setParam(name, ev.target) }))));
  $("undo").disabled = s.undo === null;
  $("redo").disabled = s.redo === null;
  $("recipe").hidden || recipePanel();
}

async function setParam(name, input) {
  const raw = input.value.trim();
  const value = /^-?\d+$/.test(raw) ? Number(raw) : raw;
  ctx.status("replaying…");
  const r = await post("/api/params", { params: { [name]: value } });
  log(r.body.log, !r.ok);
  ctx.status(r.ok ? "" : r.body.message, !r.ok);
  if (!r.ok) input.value = String(ctx.state.session.params[name]); // the old value stays
  await changed();
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

export async function start() {
  $("start").showModal();
  startStatus("loading the kernels…");
  const [kernels, recipes, sessions] = await Promise.all([get("/api/kernels"), get("/api/recipes"), get("/api/sessions")]);
  startStatus("");
  $("start-kernel").replaceChildren(kernelForm(kernels.body));
  $("start-recipe").replaceChildren(recipes.body.length
    ? h("ul", { class: "dfg-pick" }, recipes.body.map((r) => h("li", {},
      h("button", { type: "button", disabled: !!r.error, title: r.error || r.path,
        onclick: () => open({ recipe: r.path }, `replaying ${r.name}…`) }, r.name),
      h("small", {}, r.error ? ` ${r.error}` : ` ${r.kernel} · ${r.steps} steps`))))
    : h("p", { class: "muted" }, "no recipe"));
  $("start-session").replaceChildren(sessions.body.length
    ? h("ul", { class: "dfg-pick" }, sessions.body.map((x) => h("li", {},
      h("button", { type: "button", disabled: !!x.error, title: x.error || "",
        onclick: () => open({ session: x.name }, `resuming ${x.name}…`) }, x.name),
      h("small", {}, x.error ? ` ${x.error}` : ` ${x.kernel} · ${x.steps} steps · at ${x.cursor} · ${x.date}`))))
    : h("p", { class: "muted" }, "no session yet"));
}

// -- the sidebar: what applies to the selection ------------------------------------------

function field(f) {
  const attrs = { "data-field": f.name, disabled: f.readonly };
  if (f.type === "choice" && f.name === "binding") { // bind: the list, with each option's timing
    return h("div", { class: "dfg-choices", ...attrs }, f.choices.map((c, i) => {
      const t = c.timing;
      const timing = t ? `II ${t.ii} · latency ${t.latency} · span ${t.span}` : "";
      return h("label", { class: `dfg-choice${c.ok ? "" : " refused"}`, title: c.message || "" },
        h("input", { type: "radio", name: "binding", value: String(i), disabled: !c.ok,
          checked: f.value && c.value.brm === f.value.brm && c.value.implementation === f.value.implementation }),
        h("span", {}, c.label), h("small", {}, c.ok ? timing : c.message));
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
  return h("input", { ...attrs, value: f.value === null || f.value === undefined ? "" : String(f.value),
    title: f.sweepable ? "a number, or an expression over the params (W, W * 2)" : "" });
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

async function apply(transform, form, el, drop = false) {
  const body = { transform, params: params(form, el), drop };
  ctx.status(`applying ${transform}…`);
  const r = await post("/api/apply", body);
  if (!r.ok && r.body.code === "session.drop" && !drop) {
    ctx.status("");
    if (window.confirm(r.body.question)) await apply(transform, form, el, true);
    return;
  }
  log(r.body.log, !r.ok);
  ctx.status(r.ok ? "" : `${r.body.code}: ${r.body.message}`, !r.ok);
  if (r.ok) {
    ctx.state.t = null;
    await changed();
  }
}

/** The sidebar for the selection at step k; a strip while Compare is on. */
export async function sidebar() {
  const st = ctx.state;
  const bar = $("sidebar");
  bar.classList.toggle("strip", st.cmp !== null);
  if (st.cmp !== null) {
    bar.replaceChildren(h("p", { class: "dfg-strip-note", title: "Compare is on" }, "Compare"));
    return;
  }
  if (!st.sel) {
    bar.replaceChildren(h("h2", {}, "Transforms"), h("p", { class: "muted" }, "Select a node or a container: the transforms that apply there are listed here. A dot marks where one applies."));
    return;
  }
  const v = await ctx.view(st.k);
  const kinds = ["node", "container", "group", "accelerator", "edge"].filter((kind) => v.details && v.details[`${kind}:${st.sel}`]);
  const r = await get(`/api/step/${st.k}/site/${enc(st.sel)}`);
  const answers = r.ok ? r.body : {};
  const ok = Object.entries(answers).filter(([, a]) => a.ok);
  const no = Object.entries(answers).filter(([, a]) => !a.ok);
  const rows = ok.map(([name, a]) => {
    const isOpen = st.t === name;
    const formEl = isOpen ? h("form", { class: "dfg-form", onsubmit: (ev) => { ev.preventDefault(); apply(name, a.form, formEl); } },
      a.form.map((f) => h("label", { class: `dfg-field${f.readonly ? " readonly" : ""}` }, h("span", {}, f.name), field(f))),
      h("button", { type: "submit", class: "primary" }, "Apply")) : null;
    return h("div", { class: `dfg-transform${isOpen ? " open" : ""}` },
      h("button", { type: "button", class: "dfg-transform-name", "aria-expanded": isOpen ? "true" : "false",
        onclick: () => { st.t = isOpen ? null : name; ctx.writeHash(); sidebar(); } }, name),
      formEl);
  });
  bar.replaceChildren(
    h("h2", {}, kinds.join(" · ") || "item", " ", h("b", {}, st.sel)),
    ...(rows.length ? rows : [h("p", { class: "muted" }, "No transform applies here.")]),
    no.length ? h("details", { class: "dfg-not-here" }, h("summary", {}, `Not here (${no.length})`),
      h("ul", {}, no.map(([name, a]) => h("li", {}, h("b", {}, name), " ", h("small", {}, a.message))))) : null,
  );
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
    out.push(h("p", {}, `saved ${body.recipe}`), h("pre", { class: "mono" }, body.commands.join("\n")));
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
  saveAnswer(r.body, r.ok);
}

export function init(c) {
  ctx = c;
  for (const el of document.querySelectorAll("[data-edit]")) el.hidden = false;
  $("sidebar").hidden = false;
  $("reload").hidden = true;
  $("undo").addEventListener("click", () => ctx.state.session.undo !== null && move(ctx.state.session.undo));
  $("redo").addEventListener("click", () => ctx.state.session.redo !== null && move(ctx.state.session.redo));
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
  $("log-toggle").addEventListener("click", (ev) => {
    const open = $("log").classList.toggle("collapsed");
    ev.target.setAttribute("aria-expanded", open ? "false" : "true");
  });
  $("recipe-toggle").addEventListener("click", (ev) => {
    $("recipe").hidden = !$("recipe").hidden;
    ev.target.setAttribute("aria-expanded", $("recipe").hidden ? "false" : "true");
    if (!$("recipe").hidden) recipePanel();
  });
}
