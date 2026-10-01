// ============================================================
//  Lava lamp — start-up and main loop.
//
//  src/sim/       the SPH simulation (also run headless by tools/)
//  src/render/    the WebGL renderer; shaders live in shaders/
//  src/app/       the controls panel, mouse grab, benchmark mode
//  src/view.js    how the lamp is framed in the window
// ============================================================

import { SPH } from "./sim/sim.js";
import { MetaballRenderer } from "./render/renderer.js";
import { loadShaders } from "./render/shaders.js";
import { updateView } from "./view.js";
import { $, hexToRgb } from "./app/dom.js";
import { initControls } from "./app/controls.js";
import { initGrab } from "./app/grab.js";
import { initBench } from "./app/bench.js";

await loadShaders();

// ----------------- App glue ----------------------------------
const canvas = document.getElementById("lava");

// ---- Render resolution ----
// The backing store follows the canvas's *displayed* size × devicePixelRatio
// (capped at 2×) × a quality scale, so a phone showing a 320-px-wide lamp
// doesn't shade a 760×1400 buffer. MAX_PIXELS caps big fullscreen windows.
// The quality scale adapts to measured frame time unless ?scale=<0.3-1>
// pins it (handy for benchmarking).
const MAX_PIXELS = 2560 * 1600;   // wall pixels are cheap (no particle gather)
const MIN_SCALE = 0.45;
const pinnedScale = parseFloat(new URLSearchParams(location.search).get("scale"));
let renderScale = pinnedScale > 0 ? Math.min(1, Math.max(0.3, pinnedScale)) : 1.0;

function sizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return;
  updateView(rect.width, rect.height);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let w = rect.width * dpr * renderScale;
  let h = rect.height * dpr * renderScale;
  const over = Math.sqrt((w * h) / MAX_PIXELS);
  if (over > 1) { w /= over; h /= over; }
  w = Math.max(1, Math.round(w));
  h = Math.max(1, Math.round(h));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}
sizeCanvas();
new ResizeObserver(sizeCanvas).observe(canvas);

const MAX_P = 700;   // fluid (slider max 600) + wall budget (~40)
const sim = new SPH({ numParticles: 160 });
const renderer = new MetaballRenderer(canvas, MAX_P);


const params = {
  bg:   hexToRgb("#2a0d3e"),
  cold: hexToRgb("#9a0f1a"),
  hot:  hexToRgb("#ffcf3a"),
  compressColor: hexToRgb("#FFE0B0"),
  glow: 0.38,
  time: 0,
  volAbsorb: true,
  caustics: true,
  lens: true,
  bloom: true,
  wall: true,
};

// --- UI ---

const { updateRaysLabel } = initControls({ sim, params, renderer });
initGrab({ canvas, sim, params });

// --- Main loop ---
let last = performance.now();
let frames = 0;
let fpsTime = last;
let paused = false;

// Adaptive quality: every ~1 s look at the average frame interval. Too slow
// → shade fewer pixels; comfortably at 60 fps for a while → try more. After
// a downgrade we wait before probing upward again; if a probe fails right
// away the wait doubles (up to ~1 min), so a device sitting on the edge
// doesn't stutter every few seconds.
const FRAME_MS = 1000 / 60;
const quality = { sum: 0, count: 0, goodWindows: 0, holdOff: 0, warmup: 30,
                  backoff: 8, sinceUpgrade: 99 };
let nextFrameAt = 0;
function adaptQuality(frameMs) {
  if (pinnedScale > 0) return;
  if (quality.warmup > 0) { quality.warmup--; return; }   // shader compile, JIT
  quality.sum += frameMs;
  quality.count++;
  if (quality.sum < 1000) return;
  const avg = quality.sum / quality.count;
  quality.sum = 0; quality.count = 0;
  if (quality.holdOff > 0) quality.holdOff--;
  quality.sinceUpgrade++;
  if (avg > FRAME_MS * 1.1 && renderScale > MIN_SCALE) {        // under ~54 fps
    renderScale = Math.max(MIN_SCALE, renderScale * 0.85);
    quality.goodWindows = 0;
    quality.backoff = quality.sinceUpgrade <= 2 ? Math.min(64, quality.backoff * 2) : 8;
    quality.holdOff = quality.backoff;
    sizeCanvas();
  } else if (avg < FRAME_MS * 1.06) {
    quality.goodWindows++;
    if (quality.goodWindows >= 3 && quality.holdOff === 0 && renderScale < 1) {
      renderScale = Math.min(1, renderScale * 1.1);
      quality.goodWindows = 0;
      quality.sinceUpgrade = 0;
      sizeCanvas();
    }
  } else {
    quality.goodWindows = 0;
  }
}

// Pause simulation when tab is hidden to save CPU/GPU
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    paused = true;
  } else {
    paused = false;
    last = performance.now(); // reset dt so we don't get a huge jump
    nextFrameAt = 0;
    quality.sum = 0; quality.count = 0; quality.warmup = 10;
  }
});

// ---- Fixed physics timestep ----
// The sim always advances in identical frames of PHYS_DT (one regroup +
// 5 substeps). Letting the step follow the display frame rate made the
// fluid behave differently per device: at 30 fps a quarter of the wax
// ended up in 80+ particle blobs, at 60 fps none. Rendering interpolates
// positions between the last two physics frames, so motion stays smooth
// at any frame rate or Speed setting.
const PHYS_DT = 1 / 60;
const SUBSTEPS = 5;
const MAX_PHYS_FRAMES = 4;          // per render; beyond this we slow down
let physAcc = 0;
let interpX = null, interpY = null, prevX = null, prevY = null;
let interpN = -1, interpReset = -1;
function stepPhysics(rawDt) {
  const n = sim.n;
  if (!interpX || interpX.length < sim.cap) {
    interpX = new Float32Array(sim.cap); interpY = new Float32Array(sim.cap);
    prevX = new Float32Array(sim.cap);   prevY = new Float32Array(sim.cap);
  }
  if (n !== interpN || sim.resetCount !== interpReset) {
    prevX.set(sim.x.subarray(0, n)); prevY.set(sim.y.subarray(0, n));
    interpN = n; interpReset = sim.resetCount; physAcc = 0;
  }
  physAcc += Math.min(0.1, rawDt) * sim.timeScale;
  let steps = 0;
  while (physAcc >= PHYS_DT) {
    if (steps === MAX_PHYS_FRAMES) { physAcc = 0; break; }
    prevX.set(sim.x.subarray(0, n)); prevY.set(sim.y.subarray(0, n));
    sim.stepFrame(PHYS_DT);
    for (let i = 0; i < SUBSTEPS; i++) {
      sim.step(PHYS_DT / SUBSTEPS);
      sim.applyGrab(PHYS_DT / SUBSTEPS);
    }
    physAcc -= PHYS_DT;
    steps++;
  }
  const a = physAcc / PHYS_DT;
  for (let i = 0; i < n; i++) {
    interpX[i] = prevX[i] + (sim.x[i] - prevX[i]) * a;
    interpY[i] = prevY[i] + (sim.y[i] - prevY[i]) * a;
  }
  params.px = interpX; params.py = interpY;
}

function frame(now) {
  if (paused) { requestAnimationFrame(frame); return; }
  // Cap at ~60 fps: on 90/120 Hz screens rAF fires more often, which would
  // multiply the sim + shading work for no visible benefit. Pacing against
  // a schedule (rather than "skip if < 16 ms") averages 60 fps at 90 Hz too.
  if (now < nextFrameAt - 1) { requestAnimationFrame(frame); return; }
  nextFrameAt = Math.max(nextFrameAt + FRAME_MS, now - FRAME_MS);
  const rawDt = (now - last) / 1000;
  adaptQuality(now - last);
  last = now;
  stepPhysics(rawDt);

  params.time = now / 1000;
  params.quality = renderScale;
  renderer.render(sim, params);

  frames++;
  if (now - fpsTime > 500) {
    const fps = Math.round((frames * 1000) / (now - fpsTime));
    $("fpsText").textContent = fps + " fps · " + Math.round(renderScale * 100) + "% res";
    $("partText").textContent = sim.n + " particles";
    updateRaysLabel();
    frames = 0;
    fpsTime = now;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

initBench(sim);

// For poking at a running lamp from the console (module variables aren't
// globals): lavaLamp.sim, .renderer, .params, …
window.lavaLamp = {
  sim, renderer, params, quality, frame,
  get renderScale() { return renderScale; }, set renderScale(v) { renderScale = v; sizeCanvas(); },
  get paused() { return paused; }, set paused(v) { paused = v; },
  get nextFrameAt() { return nextFrameAt; }, set nextFrameAt(v) { nextFrameAt = v; },
};
