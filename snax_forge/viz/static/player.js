// Play / pause for the selected cycle. A play button steps the hash's
// `cycle` forward on a timer, as the right arrow key does (on the schedule to
// the next event cycle, the server's /step), so the schedule cursor, the
// cluster view and the memory tab's overlay all follow it. It stops where
// there is nothing left to step to, on another run or view, and on Reload;
// Space toggles it. A tick waits while a draw or a step is still running, so
// a slow cycle is never skipped and the page never falls behind.
//
// Playing is not kept in the hash: a refresh or a shared link shows the
// cycle, paused. The interval is kept in the browser (localStorage) when it
// can be, since it is a preference and not part of what is shown.
//
// app.js says how to step (setStepper); the views put a control in their
// cycle heads (playControl), and every control on the page shows one state.

import { h } from "./dom.js";

const SPEEDS = [[1000, "1 s"], [500, "0.5 s"], [250, "0.25 s"], [100, "0.1 s"]];
const KEY = "snax-forge.play.ms";

let ms = loadSpeed();
let timer = null;
let stepper = null; // {next(): Promise of a cycle | null, go(cycle), busy(): bool}
let stepping = false; // a next() is under way
const controls = new Set(); // {button, select, seen} on the page

function loadSpeed() {
  try {
    const v = Number(localStorage.getItem(KEY));
    return SPEEDS.some(([m]) => m === v) ? v : 500;
  } catch {
    return 500;
  }
}

function saveSpeed(v) {
  try {
    localStorage.setItem(KEY, String(v));
  } catch {
    // no storage (private window, blocked): the choice lasts until the page is left
  }
}

/** How to step: next() gives the cycle after the shown one, or null with none to step to. */
export function setStepper(s) {
  stepper = s;
}

/** Paint every control; one taken off the page (a redraw) is dropped, one not yet put on it is kept. */
function refresh() {
  for (const c of controls) {
    if (c.button.isConnected) c.seen = true;
    else if (c.seen) {
      controls.delete(c);
      continue;
    }
    c.button.textContent = timer ? "❚❚ Pause" : "▶ Play";
    c.button.setAttribute("aria-pressed", String(!!timer));
    c.button.title = timer ? "Stop stepping (Space)" : "Step the cycle forward on a timer (Space)";
    c.select.value = String(ms);
  }
}

async function tick() {
  if (!stepper || stepping || stepper.busy()) return; // the last cycle is still being stepped or drawn
  stepping = true;
  try {
    const n = await stepper.next();
    if (!timer) return; // stopped meanwhile
    if (n === null) stopPlaying();
    else stepper.go(n);
  } finally {
    stepping = false;
  }
}

function startPlaying() {
  if (timer || !stepper) return;
  timer = setInterval(tick, ms);
  tick();
  refresh();
}

export function stopPlaying() {
  if (!timer) return;
  clearInterval(timer);
  timer = null;
  refresh();
}

export function togglePlaying() {
  if (timer) stopPlaying();
  else startPlaying();
}

function setSpeed(v) {
  ms = v;
  saveSpeed(v);
  if (timer) { // restart at the new pace
    clearInterval(timer);
    timer = setInterval(tick, ms);
  }
  refresh();
}

/** A play / pause button and an interval choice, for a view's cycle head. */
export function playControl() {
  const button = h("button", { type: "button", class: "play", onclick: togglePlaying });
  const select = h("select", { class: "play-speed", "aria-label": "Interval between cycles", title: "Interval between cycles",
    onchange: (e) => setSpeed(Number(e.target.value)) },
  SPEEDS.map(([m, label]) => h("option", { value: String(m) }, `every ${label}`)));
  const c = { button, select, seen: false };
  controls.add(c);
  refresh();
  return h("span", { class: "player" }, button, select);
}
