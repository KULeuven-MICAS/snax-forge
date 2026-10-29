// Text of the trace's events and of a journey's hops (C§7, D57): how the
// cluster view's event list, the schedule's tooltips and the memory tab's
// drawer put one event in words. Formatting only: every field is the
// server's, nothing is looked up or counted here.

/** A DMA copy's direction as text. */
export const DIRECTION_TEXT = { l2_to_l1: "L2 → L1", l1_to_l2: "L1 → L2" };

const hex = (n) => `0x${Number(n).toString(16)}`;
const opText = (op) => (op === "w" ? "write" : "read");

/** An access's elements and their values: "A[4..7] = 5, 6, 7, 8"; "" without a region. */
export function elemsText(e) {
  if (!e.region || !e.elems?.length) return "";
  const xs = e.elems;
  const run = xs.every((x, i) => x === xs[0] + i);
  const which = xs.length === 1 ? `${xs[0]}` : run ? `${xs[0]}..${xs[xs.length - 1]}` : xs.join(", ");
  const vals = Array.isArray(e.vals) && e.vals.length ? ` = ${e.vals.join(", ")}` : "";
  return `${e.region}[${which}]${vals}`;
}

/** One line of text for a trace event (C§7), without its cycle. */
export function eventText(e) {
  const of = e.task ? ` (${e.task})` : "";
  const what = elemsText(e);
  switch (e.k) {
    case "start": return `${e.src} starts ${e.task}; busy from the next cycle`;
    case "done": return e.id !== undefined ? `DMA copy ${e.id}${of} done`
      : `${e.src} done with ${e.task}${e.firings !== undefined ? `, ${e.firings} firings` : ""}`;
    case "issue": return `DMA copy ${e.id}${of} issued: ${DIRECTION_TEXT[e.dir] ?? e.dir}, ${e.size * e.reps} B from ${hex(e.from)} to ${hex(e.to)}`;
    case "csr": return `${e.src} ${opText(e.op)}s CSR ${hex(e.csr)} of ${e.block} = ${e.value}${e.held ? `, held ${e.held} cycles` : ""}`;
    case "poll": return `${e.src} polls CSR ${hex(e.csr)} of ${e.block}: ${e.value}, ${e.count} reads until cycle ${e.last}`;
    case "barrier": return `${e.src} at a barrier until cycle ${e.last}`;
    case "state": return `${e.src} goes to ${e.mode}`;
    case "func": return `${e.src} enters ${e.name}`;
    case "grant": return `${e.src} ${opText(e.op)} accepted: bank ${e.bank}, row ${e.row}${what ? `, ${what}` : ""}${e.op === "r" ? `; data back in cycle ${e.back}` : ""}`;
    case "stall": return `${e.src} held back at bank ${e.bank}, row ${e.row}${what ? ` (${what})` : ""}: the bank served ${e.by}`;
    case "dma": return `DMA copy ${e.id} beat ${e.i} ${opText(e.op)}s L1 bank ${e.bank}, row ${e.row}${what ? `, ${what}` : ""}`;
    case "burst": return `DMA copy ${e.id} ${opText(e.op)}s ${e.size} B of L2 at ${hex(e.addr)}${what ? `, ${e.region}` : ""}`;
    case "fire": return `${e.src} fires ${e.n}${of}`;
    case "push": return `${e.src} pushes result ${e.n}${of}${e.first === e.last ? "" : `, folding firings ${e.first}–${e.last}`}`;
    case "class": return `${e.src} ${e.mode} from here${of}`;
    case "fifo": return `${e.src} lane ${e.lane} holds ${e.count} from here`;
    case "insn": return `${e.src} ${hex(e.pc)} in ${e.func}: ${e.text}`;
    default: return `${e.src} ${e.k}`;
  }
}

// -- a journey's hops and firings (/api/run/<n>/journey, D57) ----------------------

/** One hop of an element: "L1 read by acc_a.lane1 (bank 5, row 0) = 7; bank served ...". */
export function hopText(x) {
  const where = x.k === "dma" ? `DMA copy ${x.id} beat ${x.i}, bank ${x.bank}, row ${x.row}`
    : x.k === "burst" ? `DMA copy ${x.id}, ${x.size} B from ${x.addr}`
      : `bank ${x.bank}, row ${x.row}`;
  const value = x.value !== undefined ? ` = ${x.value}` : "";
  const served = x.served ? `; bank served ${x.served.port} for ${x.served.element ?? `row ${x.served.row}`}` : "";
  return `${x.mem.toUpperCase()} ${x.act} by ${x.by} (${where})${value}${served}`;
}

/** A firing's operands and results with their values: "acc_a A[4], A[5] (1, 2) → acc_out C[4], C[5] (101, 103)". */
export function operandsText(f) {
  const vals = (xs) => (xs && xs.length ? ` (${xs.join(", ")})` : "");
  const ins = Object.entries(f.inputs).map(([m, xs]) => `${m} ${xs.join(", ")}${vals(f.operands?.[m]?.vals)}`).join("; ");
  const outs = Object.entries(f.outputs).map(([m, xs]) => `${m} ${xs.join(", ")}${vals(f.push?.vals)}`).join("; ");
  return `${ins}${outs ? ` → ${outs}` : ""}`;
}
