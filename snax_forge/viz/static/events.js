// What the schedule (VIS2) and the cluster view (VIS3) both read from a run's
// trace (D57, D61): the event kinds, the whole-run caches, and small queries.
//
// Two whole-run fetches, each done once per run and kept until Reload:
//   task events  cmd, start, done (never filtered, D49)
//   FIFO counts  every `fifo` event, indexed per lane, so the count of a lane
//                at any cycle is one bisect (a fifo event is a state change:
//                its t is the first cycle the new count holds, D39)
//
// A beat event of source `src` in cycle t exists only if the run's beat
// filter kept it (D49). `beatTraced` answers that, so a view can say "not
// traced" instead of drawing a filtered cycle as an idle one.

import { CLASS_GROUP } from "./dom.js";

export const TASK_KINDS = ["cmd", "start", "done"];
export const BEAT_KINDS = ["grant", "stall", "fire", "dma_beat", "poll", "fifo"];

/** Colour group of each class that is drawn; idle is left out (left blank). */
export const GROUP = Object.fromEntries(Object.entries(CLASS_GROUP).filter(([c]) => c !== "idle"));

// -- whole-run caches ----------------------------------------------------------------

const taskCache = new WeakMap(); // detail -> Promise of the task events
const fifoCache = new WeakMap(); // detail -> Promise of Map "src|lane" -> {ts, counts}

/** The task events of the whole run (fetched once per run). */
export function taskEvents(api, detail) {
  if (!taskCache.has(detail)) {
    taskCache.set(detail, api.events(detail.name, 0, detail.run.total_cycles + 1, { k: TASK_KINDS }));
  }
  return taskCache.get(detail);
}

/** Every fifo event of the run, per lane: {ts: [t...], counts: [count...]} in cycle order. */
export function fifoIndex(api, detail) {
  if (!fifoCache.has(detail)) {
    const load = async () => {
      const index = new Map();
      if (detail.trace?.level !== "beat") return index;
      const evs = await api.events(detail.name, 0, detail.run.total_cycles + 1, { k: ["fifo"] });
      for (const e of evs) {
        const key = `${e.src}|${e.lane}`;
        if (!index.has(key)) index.set(key, { ts: [], counts: [] });
        const s = index.get(key);
        s.ts.push(e.t);
        s.counts.push(e.count);
      }
      return index;
    };
    fifoCache.set(detail, load());
  }
  return fifoCache.get(detail);
}

// -- queries -----------------------------------------------------------------------

/** Were the beat events of `src` in cycle t kept by the run's trace (D49)? */
export function beatTraced(trace, src, t) {
  if (!trace || trace.level !== "beat") return false;
  if (trace.filter_sources && !trace.filter_sources.includes(src)) return false;
  const w = trace.filter_window;
  return !w || (w[0] <= t && t < w[1]);
}

/**
 * The count of lane `lane` of FIFO `fifo` in cycle t, or null if the trace
 * cannot say: the FIFO's beat events were filtered out in t, or a window
 * filter starting after cycle 0 dropped the events that set the count before
 * the lane's first event in the window.
 */
export function fifoCount(index, trace, fifo, lane, t) {
  if (!beatTraced(trace, fifo, t)) return null;
  const s = index.get(`${fifo}|${lane}`);
  let lo = 0;
  let hi = s ? s.ts.length : 0;
  while (lo < hi) { // first event with ts > t
    const mid = (lo + hi) >> 1;
    if (s.ts[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0) return s.counts[lo - 1];
  const start = trace.filter_window ? trace.filter_window[0] : 0;
  return start > 0 ? null : 0;
}

/** The class of a component in cycle t, from its class runs [cls, start, stop). */
export function classAt(runs, t) {
  if (!runs) return null;
  let lo = 0;
  let hi = runs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (runs[mid][2] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo < runs.length && runs[lo][1] <= t ? runs[lo][0] : null;
}

/** Group events by owner: grants and stalls go to the port's owner, a FIFO to its streamer. */
export function byOwner(events, profile) {
  const fifoOwner = Object.fromEntries(Object.entries(profile.streamers).map(([n, st]) => [st.fifo.name, n]));
  const owner = (e) => {
    if (e.k === "grant" || e.k === "stall") return profile.ports[e.port]?.owner ?? e.src;
    if (e.k === "fifo") return fifoOwner[e.src] ?? e.src;
    return e.src;
  };
  const by = {};
  for (const e of events) (by[owner(e)] ??= []).push(e);
  return by;
}

/** Banks as text: "bank 8" or "banks 0–7". */
export function bankText(banks) {
  return banks.length > 1 ? `banks ${banks[0]}–${banks[banks.length - 1]}` : `bank ${banks[0]}`;
}

/** One line of text for an event, as the cycle's event list shows it. */
export function describe(e) {
  switch (e.k) {
    case "cmd": {
      const what = e.op === "wait" ? `wait ${e.block} (${e.mode})` : `${e.op} ${e.reg ?? ""}${e.value !== undefined ? ` = ${e.value}` : ""}`;
      return `pc ${e.pc}: ${what}, cycles ${e.t}–${e.last}`;
    }
    case "start": return "start landed (busy from the next cycle)";
    case "done": return "done (busy reads 0 from here)";
    case "grant":
    case "stall": {
      const head = e.k === "grant" ? "granted" : e.wider ? "stalled by a wider grant" : "stalled";
      return `${e.port} ${head}: ${e.w ? "write" : "read"} ${e.mem} addr ${e.addr}, ${bankText(e.banks)}, row ${e.row}`;
    }
    case "fire": return `firing ${e.n}`;
    case "dma_beat": return `${e.side === "src" ? "read" : "write"} beat ${e.i}, ${e.mem} addr ${e.addr}`;
    case "poll": return `poll ${e.block}: busy = ${e.value}`;
    case "fifo": return `${e.src} lane ${e.lane} holds ${e.count} from here`;
    default: return JSON.stringify(e);
  }
}
