const DATA_URL = "data/research-data.json";
const NS = "http://www.w3.org/2000/svg";

const COLORS = {
  N: "#246b9e",
  R: "#bd493d",
  U: "#71766f",
  partner: "#7058be",
  focal: "#c87920",
  partnerSoft: "#efecfa",
  focalSoft: "#fff0dc",
  changed: "#f5d74f",
  ip: "#27384b",
  h0: "#2377ad",
  h1: "#c64343",
  accent: "#116f62",
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

if (new URLSearchParams(location.search).has("present")) document.body.classList.add("present");
const formatInt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function svgEl(name, attributes = {}, content = "") {
  const node = document.createElementNS(NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (content) node.textContent = content;
  return node;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function nearestIndex(values, target) {
  let best = 0;
  let distance = Infinity;
  values.forEach((value, index) => {
    const next = Math.abs(value - target);
    if (next < distance) {
      best = index;
      distance = next;
    }
  });
  return best;
}

function drawCallout(svg, x, y, text) {
  const width = text.length * 6.6 + 16;
  svg.append(svgEl("rect", {
    x: x - width / 2,
    y: y - 12,
    width,
    height: 16,
    rx: 3,
    fill: "#fbfcfdee",
  }));
  svg.append(svgEl("text", { x, y, class: "loop-callout" }, text));
}

function seriesPath(values, x, y) {
  return values.map((value, index) =>
    (index ? "L" : "M") + x(index).toFixed(2) + "," + y(value).toFixed(2)
  ).join(" ");
}

async function loadData() {
  const response = await fetch(DATA_URL);
  if (!response.ok) throw new Error("Could not load " + DATA_URL + " (" + response.status + ")");
  return response.json();
}

function initMedia() {
  const videos = $$(".explanatory-video[data-autoplay]");
  if (reducedMotion) {
    videos.forEach((video) => video.pause());
    return;
  }

  if (!("IntersectionObserver" in window)) {
    videos.forEach((video) => video.play().catch(() => {}));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.play().catch(() => {});
      } else {
        entry.target.pause();
      }
    });
  }, { threshold: 0.35 });

  videos.forEach((video) => observer.observe(video));
}

function renderWaitingChart(hero) {
  const rows = [
    ["U", "Uniform sampling", hero.U.medianEpochs, "u"],
    ["R", "Soup", hero.R.medianEpochs, "r"],
    ["N", "Random-partner execution", hero.N.medianEpochs, "n"],
  ];
  const logs = rows.map((row) => Math.log10(row[2]));
  const min = Math.min(...logs) - 0.08;
  const max = Math.max(...logs);
  $("#waiting-chart").innerHTML = rows.map((row, index) => {
    const key = row[0];
    const label = row[1];
    const value = row[2];
    const css = row[3];
    const width = 23 + 77 * ((logs[index] - min) / (max - min));
    const display = value >= 70000
      ? "~" + formatInt.format(Math.round(value / 1000) * 1000)
      : formatInt.format(value);
    return '<div class="wait-row ' + css + '">' +
      '<div class="wait-label"><b>' + key + '</b><span>' + label + '</span></div>' +
      '<div class="wait-track"><span class="wait-bar" style="width:' + width +
      '%;animation-delay:' + (index * 90) + 'ms"></span></div>' +
      '<div class="wait-value">' + display + ' <small>epochs</small></div></div>';
  }).join("");
}

function phaseForStep(phases, step) {
  for (let index = phases.length - 1; index >= 0; index -= 1) {
    if (step >= phases[index].startStep) return { phase: phases[index], index };
  }
  return { phase: phases[0], index: 0 };
}

function traceStateAt(trace, step) {
  if (trace.states[step] && trace.states[step].step === step) return trace.states[step];
  return trace.states[nearestIndex(trace.states.map((state) => state.step), step)];
}

function normalizedLoopSpans(phase) {
  return (phase.loopSpans || []).map((span) => {
    if (Array.isArray(span)) {
      if (span.length === 2) return span;
      if (span.length === 3 && typeof span[0] === "string") {
        const offset = span[0].toUpperCase().startsWith("F") ? 16 : 0;
        return [offset + Number(span[1]), offset + Number(span[2])];
      }
    }
    if (span && typeof span === "object") {
      const offset = String(span.half || span.tape || "P").toUpperCase().startsWith("F") ? 16 : 0;
      return [offset + Number(span.start), offset + Number(span.end)];
    }
    return null;
  }).filter(Boolean);
}

function changedBetween(previous, current) {
  if (!previous || !current) return [];
  const before = previous.partner + previous.focal;
  const after = current.partner + current.focal;
  return Array.from(after).map((symbol, index) => symbol !== before[index] ? index : -1)
    .filter((index) => index >= 0);
}

function displaySymbol(symbol) {
  return symbol === "x" ? "·" : symbol;
}

function instructionAt(state) {
  const tape = state.partner + state.focal;
  return tape[state.pc] ? displaySymbol(tape[state.pc]) : "—";
}

function initTrace(trace) {
  const slider = $("#trace-slider");
  const milestones = trace.phases.map((phase) => phase.milestoneStep);
  const phaseTitles = [
    "The loop is completed at the boundary.",
    "The loop drives the rewrite.",
    "The replicator structure takes shape.",
    "A functional replicator is formed.",
  ];
  let timer = null;
  let lastState = null;
  let currentState = trace.states[0];
  let currentPhase = trace.phases[0];
  let currentChanges = [];

  slider.max = trace.maxStep;

  function stop() {
    if (timer) window.clearInterval(timer);
    timer = null;
    $("#trace-play").textContent = "Play";
    $("#trace-play").setAttribute("aria-pressed", "false");
  }

  function render(step, clearChanges = false) {
    const state = traceStateAt(trace, step);
    const phaseResult = phaseForStep(trace.phases, state.step);
    const phase = phaseResult.phase;
    const phaseIndex = phaseResult.index;
    currentChanges = clearChanges ? [] : changedBetween(lastState, state);
    currentState = state;
    currentPhase = phase;
    lastState = state;

    slider.value = state.step;
    $("#trace-step-label").textContent = "step " + state.step + " / " + trace.maxStep;
    $("#readout-ip").textContent = state.pc + " " + instructionAt(state);
    $("#readout-h0").textContent = state.h0;
    $("#readout-h1").textContent = state.h1;
    $("#phase-count").textContent = (phaseIndex + 1) + " / " + trace.phases.length;
    $("#phase-title").textContent = phaseTitles[phaseIndex];
    $("#phase-copy").textContent = phase.caption;

    const previousMilestone = milestones.some((value) => value < state.step);
    const nextMilestone = milestones.some((value) => value > state.step);
    $("#trace-prev").disabled = !previousMilestone;
    $("#trace-next").disabled = !nextMilestone;

    const status = $("#trace-status");
    status.classList.toggle("final", state.step === trace.maxStep);
    if (state.step === trace.maxStep) status.textContent = "functional replicator";
    else if (phase.id === "loop-completion") status.textContent = "loop completed";
    else status.textContent = "rewrite in progress";

    renderTraceSvg(trace, state, phase, currentChanges);
  }

  slider.addEventListener("input", () => {
    stop();
    render(Number(slider.value));
  });
  $("#trace-reset").addEventListener("click", () => {
    stop();
    lastState = null;
    render(0, true);
  });
  $("#trace-prev").addEventListener("click", () => {
    stop();
    const step = Number(slider.value);
    const previous = milestones.filter((value) => value < step).at(-1);
    if (previous !== undefined) render(previous);
  });
  $("#trace-next").addEventListener("click", () => {
    stop();
    const step = Number(slider.value);
    const next = milestones.find((value) => value > step);
    if (next !== undefined) render(next);
  });
  $("#trace-play").addEventListener("click", () => {
    if (timer) {
      stop();
      return;
    }
    if (Number(slider.value) >= trace.maxStep) {
      lastState = null;
      render(0, true);
    }
    $("#trace-play").textContent = "Pause";
    $("#trace-play").setAttribute("aria-pressed", "true");
    timer = window.setInterval(() => {
      const next = Number(slider.value) + 1;
      if (next > trace.maxStep) {
        stop();
        return;
      }
      render(next);
      if (next === trace.maxStep) stop();
    }, reducedMotion ? 90 : 42);
  });

  let resizeTimer;
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => renderTraceSvg(trace, currentState, currentPhase, currentChanges), 100);
  });
  render(0, true);
}

function renderTraceSvg(trace, state, phase, changed) {
  const container = $("#trace-visual");
  const narrow = container.clientWidth > 0 && container.clientWidth < 650;
  const half = trace.halfLength;
  const width = narrow ? 560 : 1000;
  const height = narrow ? 390 : 300;
  const cell = narrow ? 30 : 27;
  const startX = narrow ? 40 : 46;
  const gap = narrow ? 0 : 34;
  const partnerY = narrow ? 72 : 103;
  const focalY = narrow ? 226 : partnerY;
  const svg = svgEl("svg", {
    viewBox: "0 0 " + width + " " + height,
    class: "trace-svg",
    "aria-hidden": "true",
  });

  const defs = svgEl("defs");
  const arrowMarker = svgEl("marker", {
    id: "trace-arrowhead",
    markerWidth: 8,
    markerHeight: 6,
    refX: 7,
    refY: 3,
    orient: "auto",
  });
  arrowMarker.append(svgEl("path", { d: "M0,0 L8,3 L0,6 Z", fill: COLORS.partner }));
  defs.append(arrowMarker);
  svg.append(defs);

  function position(index) {
    if (narrow) {
      return {
        x: startX + (index % half) * cell,
        y: index < half ? partnerY : focalY,
      };
    }
    return {
      x: startX + (index < half ? index * cell : half * cell + gap + (index - half) * cell),
      y: partnerY,
    };
  }

  const partnerCenter = startX + half * cell / 2;
  const focalCenter = narrow
    ? partnerCenter
    : startX + half * cell + gap + half * cell / 2;
  svg.append(svgEl("text", {
    x: partnerCenter,
    y: partnerY - 38,
    class: "tape-name",
    fill: COLORS.partner,
  }, "PARTNER TAPE"));
  svg.append(svgEl("text", {
    x: focalCenter,
    y: narrow ? focalY - 14 : focalY - 38,
    class: "tape-name",
    fill: COLORS.focal,
  }, "TARGET TAPE"));

  if (narrow) {
    const endX = startX + half * cell;
    svg.append(svgEl("path", {
      d: "M" + endX + "," + (partnerY + 21) +
        " C" + (endX + 24) + "," + (partnerY + 55) +
        " " + (startX - 24) + "," + (focalY - 34) +
        " " + startX + "," + (focalY - 4),
      fill: "none",
      stroke: "#7d827b",
      "stroke-width": 1,
      "stroke-dasharray": "4 4",
    }));
    svg.append(svgEl("text", {
      x: width / 2,
      y: 151,
      class: "boundary-label",
    }, "interaction boundary wraps to target tape"));
  } else {
    const boundaryX = startX + half * cell + gap / 2;
    svg.append(svgEl("line", {
      x1: boundaryX,
      y1: 44,
      x2: boundaryX,
      y2: 241,
      stroke: "#7d827b",
      "stroke-width": 1,
      "stroke-dasharray": "4 4",
    }));
    svg.append(svgEl("text", {
      x: boundaryX,
      y: 257,
      class: "boundary-label",
    }, "interaction boundary"));
  }

  function drawSpan(start, end) {
    const segments = [];
    if (start < half) segments.push([start, Math.min(end, half - 1)]);
    if (end >= half) segments.push([Math.max(start, half), end]);
    segments.forEach(([segmentStart, segmentEnd]) => {
      const a = position(segmentStart);
      const b = position(segmentEnd);
      svg.append(svgEl("rect", {
        x: a.x - 3,
        y: a.y - 9,
        width: b.x + cell + 3 - (a.x - 3),
        height: 53,
        rx: 3,
        class: "loop-band",
      }));
    });
  }
  const overrideSpans = state.step < 123 ? null
    : state.step <= 205 ? [[3, 15], [16, 17]]
    : state.step <= 214 ? [[3, 20]]
    : [[16, 20]];
  const activeSpans = overrideSpans || normalizedLoopSpans(phase);
  const loopIsActive = phase.id === "loop-completion" || Boolean(overrideSpans) ||
    activeSpans.some(([start, end]) => state.pc >= start && state.pc <= end);
  if (loopIsActive) {
    activeSpans.forEach((span) => drawSpan(span[0], span[1]));
  }

  if (phase.id === "loop-completion") {
    drawCallout(
      svg,
      narrow ? width / 2 : startX + half * cell + gap / 2,
      narrow ? 180 : 76,
      "PRE-REPLICATOR LOOP COMPLETED ACROSS THE BOUNDARY"
    );
  }
  if (state.step === trace.maxStep) {
    const start = position(half);
    const end = position(half * 2 - 1);
    svg.append(svgEl("rect", {
      x: start.x - 6,
      y: start.y - 13,
      width: end.x + cell + 6 - (start.x - 6),
      height: 70,
      rx: 3,
      class: "replicator-band",
    }));
  }

  const tape = Array.from(state.partner + state.focal);
  tape.forEach((symbol, index) => {
    const pos = position(index);
    const isChanged = changed.includes(index);
    const isPartner = index < half;
    svg.append(svgEl("rect", {
      x: pos.x,
      y: pos.y,
      width: cell,
      height: 42,
      fill: isChanged ? COLORS.changed : (isPartner ? COLORS.partnerSoft : COLORS.focalSoft),
      stroke: isPartner ? COLORS.partner : COLORS.focal,
      "stroke-width": isChanged ? 2 : 1,
      class: "tape-cell" + (isChanged ? " changed-now" : ""),
    }));
    svg.append(svgEl("text", {
      x: pos.x + cell / 2,
      y: pos.y + 21,
      class: "cell-symbol" + (symbol === "x" ? " cell-neutral" : ""),
    }, displaySymbol(symbol)));
    svg.append(svgEl("text", {
      x: pos.x + cell / 2,
      y: pos.y + 54,
      class: "cell-index",
    }, String(index)));
  });

  const showCopyPath = phase.id !== "stabilization" && state.step >= phase.milestoneStep;
  if (showCopyPath) {
    (phase.copyPairs || []).forEach(([source, destination], pairIndex) => {
      const sourcePos = position(source);
      const destinationPos = position(destination);
      const x1 = sourcePos.x + cell / 2;
      const y1 = sourcePos.y - 3;
      const x2 = destinationPos.x + cell / 2;
      const y2 = destinationPos.y - 3;
      let path;
      if (!narrow || sourcePos.y === destinationPos.y) {
        const arc = 34 + (pairIndex % 4) * 7;
        path = "M" + x1 + "," + y1 + " Q" + ((x1 + x2) / 2) + "," +
          (Math.min(y1, y2) - arc) + " " + x2 + "," + y2;
      } else {
        const bendX = pairIndex % 2 ? width - 20 : 20;
        path = "M" + x1 + "," + (sourcePos.y + 43) + " Q" + bendX + "," +
          ((sourcePos.y + destinationPos.y) / 2) + " " + x2 + "," + y2;
      }
      svg.append(svgEl("path", {
        d: path,
        class: "copy-arrow",
        "marker-end": "url(#trace-arrowhead)",
      }));
    });
  }

  if (state.step === trace.maxStep) {
    const start = position(half);
    const end = position(half * 2 - 1);
    drawCallout(svg, (start.x + end.x + cell) / 2, start.y - 20, "FUNCTIONAL REVERSE-COPY REPLICATOR");
  }

  const h0Pos = position(state.h0);
  const h1Pos = position(state.h1);
  const headMarkers = [
    { pos: h0Pos, label: "H0", color: COLORS.h0 },
    { pos: h1Pos, label: "H1", color: COLORS.h1 },
  ];
  if (h0Pos.y === h1Pos.y && Math.abs(h0Pos.x - h1Pos.x) < cell * 1.75) {
    const same = h0Pos.x === h1Pos.x;
    const direction = h0Pos.x <= h1Pos.x ? 1 : -1;
    headMarkers[0].shift = same ? -19 : -16 * direction;
    headMarkers[1].shift = same ? 19 : 16 * direction;
  }

  headMarkers.forEach((marker) => {
    const actualX = marker.pos.x + cell / 2;
    const labelX = clamp(actualX + (marker.shift || 0), 18, width - 18);
    const labelY = marker.pos.y + 78;
    svg.append(svgEl("line", {
      x1: actualX,
      y1: marker.pos.y + 43,
      x2: labelX,
      y2: labelY - 9,
      stroke: marker.color,
      class: "head-connector",
    }));
    svg.append(svgEl("rect", {
      x: labelX - 15,
      y: labelY - 8,
      width: 30,
      height: 16,
      fill: "#fffefb",
      stroke: marker.color,
      class: "head-badge",
    }));
    svg.append(svgEl("text", {
      x: labelX,
      y: labelY + .5,
      fill: marker.color,
      class: "head-label",
    }, marker.label));
  });

  const ipPos = position(state.pc);
  const ipX = ipPos.x + cell / 2;
  const ipY = ipPos.y + 106;
  svg.append(svgEl("line", {
    x1: ipX,
    y1: ipPos.y + 43,
    x2: ipX,
    y2: ipY - 9,
    stroke: COLORS.ip,
    class: "head-connector",
  }));
  svg.append(svgEl("rect", {
    x: ipX - 14,
    y: ipY - 8,
    width: 28,
    height: 16,
    fill: "#fffefb",
    stroke: COLORS.ip,
    class: "head-badge",
  }));
  svg.append(svgEl("text", {
    x: ipX,
    y: ipY + .5,
    fill: COLORS.ip,
    class: "head-label",
  }, "IP"));

  svg.append(svgEl("text", {
    x: startX,
    y: height - 12,
    class: "boundary-label trace-note",
  }, "Inert symbols are shown as ·"));

  container.replaceChildren(svg);
}

function drawYAxisTitle(svg, margin, plotH, text) {
  const cx = 12;
  const cy = margin.top + plotH / 2;
  svg.append(svgEl("text", {
    x: cx,
    y: cy,
    class: "axis-label axis-title",
    "text-anchor": "middle",
    transform: "rotate(-90 " + cx + " " + cy + ")",
  }, text));
}

function drawXAxis(svg, width, height, margin, plotW, xEpoch, showTitle) {
  const tickEpochs = width < 480 ? [0, 8000, 16000] : [0, 4000, 8000, 12000, 16000];
  tickEpochs.forEach((epoch) => {
    svg.append(svgEl("text", {
      x: xEpoch(epoch),
      y: height - (showTitle ? 20 : 6),
      class: "axis-label",
      "text-anchor": epoch === 0 ? "start" : (epoch === 16000 ? "end" : "middle"),
    }, epoch === 0 ? "0" : (epoch / 1000) + "k"));
  });
  if (showTitle) {
    svg.append(svgEl("text", {
      x: margin.left + plotW / 2, y: height - 6, class: "axis-label axis-title", "text-anchor": "middle",
    }, "epoch"));
  }
}

function makeDiscoveryHistogram(container, mechanism) {
  const width = Math.max(320, Math.round(container.clientWidth || 760));
  const height = 190;
  const margin = { top: 12, right: 14, bottom: 34, left: 50 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const svg = svgEl("svg", { viewBox: "0 0 " + width + " " + height });
  const edges = mechanism.discovery.binEdges;
  const rates = mechanism.discovery.N.concat(mechanism.discovery.R);
  const yMax = Math.max(0.9, ...rates) * 1.08;
  const xEpoch = (epoch) => margin.left + (epoch / 16000) * plotW;
  const y = (value) => margin.top + plotH - (value / yMax) * plotH;

  for (let tick = 0; tick <= 3; tick += 1) {
    const value = (yMax * tick) / 3;
    const py = y(value);
    svg.append(svgEl("line", { x1: margin.left, y1: py, x2: width - margin.right, y2: py, class: "grid-line" }));
    svg.append(svgEl("text", { x: margin.left - 6, y: py + 3, class: "axis-label", "text-anchor": "end" }, value.toFixed(1)));
  }
  drawXAxis(svg, width, height, margin, plotW, xEpoch, true);
  drawYAxisTitle(svg, margin, plotH, "events / run");

  const baseline = margin.top + plotH;
  const groupGap = 6;
  edges.forEach((edge, index) => {
    if (index === 0 || index === edges.length - 1) return;
    const ex = xEpoch(edge);
    svg.append(svgEl("line", {
      x1: ex, y1: baseline, x2: ex, y2: baseline + 4, class: "bin-tick",
    }));
  });

  mechanism.discovery.N.forEach((_, index) => {
    const x1 = xEpoch(edges[index]);
    const x2 = xEpoch(edges[index + 1]);
    const groupWidth = (x2 - x1) - groupGap;
    const barWidth = groupWidth / 2 - 1;
    const groupStart = x1 + groupGap / 2;
    ["N", "R"].forEach((regime, regimeIndex) => {
      const value = mechanism.discovery[regime][index];
      svg.append(svgEl("rect", {
        x: groupStart + regimeIndex * (barWidth + 1),
        y: y(value),
        width: barWidth,
        height: baseline - y(value),
        class: regime === "N" ? "bar-n" : "bar-r",
      }));
    });
  });

  container.replaceChildren(svg);
}

function makeMetricLineChart(container, config) {
  const width = Math.max(320, Math.round(container.clientWidth || 760));
  const height = 190;
  const margin = { top: 12, right: 14, bottom: config.showEpochTitle ? 34 : 24, left: 50 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const svg = svgEl("svg", { viewBox: "0 0 " + width + " " + height });
  const xEpoch = (epoch) => margin.left + (epoch / 16000) * plotW;
  const x = (index) => xEpoch(config.epochs[index]);
  const yMax = Math.max(...config.series.flatMap((item) => item.values)) * 1.1;
  const y = (value) => margin.top + plotH - (value / yMax) * plotH;

  for (let tick = 0; tick <= 3; tick += 1) {
    const value = (yMax * tick) / 3;
    const py = y(value);
    svg.append(svgEl("line", { x1: margin.left, y1: py, x2: width - margin.right, y2: py, class: "grid-line" }));
    svg.append(svgEl("text", { x: margin.left - 6, y: py + 3, class: "axis-label", "text-anchor": "end" }, config.formatY(value)));
  }
  drawXAxis(svg, width, height, margin, plotW, xEpoch, config.showEpochTitle);
  drawYAxisTitle(svg, margin, plotH, config.yTitle);

  config.series.forEach((item) => {
    svg.append(svgEl("path", { d: seriesPath(item.values, x, y), class: "curve " + (item.css || "") }));
  });

  container.replaceChildren(svg);
}

function initMechanism(mechanism) {
  makeDiscoveryHistogram($("#first-discovery-histogram"), mechanism);
  makeMetricLineChart($("#motif-chart"), {
    epochs: mechanism.epochs,
    series: [
      { values: mechanism.N.preReplicator, css: "n" },
      { values: mechanism.R.preReplicator, css: "r" },
    ],
    formatY: (value) => value.toFixed(1),
    yTitle: "partners / run",
    showEpochTitle: false,
  });
  makeMetricLineChart($("#activation-chart"), {
    epochs: mechanism.epochs,
    series: [
      { values: mechanism.N.activation, css: "n" },
      { values: mechanism.R.activation, css: "r" },
    ],
    formatY: (value) => Math.round(value * 100) + "%",
    yTitle: "% convertible",
    showEpochTitle: true,
  });
}

function initReferenceCalculation(reference) {
  if (!reference) return;
  $("#fragment-one-in").textContent = "once per " +
    (reference.onePerPrograms / 1e6).toFixed(1) + " million";
}

function initCitation() {
  const bibtex = "@article{dusek2026replicator,\n" +
    "  title={Replicator Discovery Is Faster Without Population Coupling in a Self-Modifying Program Soup},\n" +
    "  author={Dušek, František and Papadopoulos, Vassilis and Hudcová, Barbora},\n" +
    "  year={2026},\n" +
    "  note={Manuscript}\n" +
    "}";
  $("#citation-button").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex);
      showToast("BibTeX copied");
    } catch {
      showToast("Clipboard unavailable");
    }
  });
}

let toastTimer;
function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 1800);
}

async function main() {
  initMedia();
  try {
    const data = await loadData();
    renderWaitingChart(data.hero);
    initTrace(data.trace);
    initReferenceCalculation(data.referenceCalculation);
    initMechanism(data.mechanism);
    initCitation();
  } catch (error) {
    console.error(error);
    document.body.insertAdjacentHTML(
      "afterbegin",
      '<div class="data-error">Research data could not be loaded. Serve this directory over HTTP; see web-demo/README.md.</div>'
    );
  }
}

main();
