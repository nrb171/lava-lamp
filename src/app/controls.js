// ============================================================
//  The controls panel: sliders and toggles → sim / params, presets,
//  panel and fullscreen buttons, keyboard shortcuts.
// ============================================================

import { $, hexToRgb } from "./dom.js";

export function initControls({ sim, params, renderer }) {

  $("bgColor").oninput   = (e) => params.bg   = hexToRgb(e.target.value);
  $("coldColor").oninput = (e) => params.cold = hexToRgb(e.target.value);
  $("hotColor").oninput  = (e) => params.hot  = hexToRgb(e.target.value);
  $("compColor").oninput = (e) => params.compressColor = hexToRgb(e.target.value);
  $("volAbsorb").onchange = (e) => params.volAbsorb = e.target.checked;
  $("caustics").onchange = (e) => params.caustics = e.target.checked;
  $("glassLens").onchange = (e) => params.lens = e.target.checked;
  $("bloom").onchange = (e) => params.bloom = e.target.checked;
  $("wallLight").onchange = (e) => params.wall = e.target.checked;

  // Light rays: scales the GPU light trace's ray grids (both axes), wall
  // and liquid; the label shows how many rays are traced
  params.rayScale = 1;
  $("rays").oninput = (e) => { params.rayScale = parseFloat(e.target.value); };
  $("taa").onchange = (e) => params.taa = e.target.checked;
  $("denoise").onchange = (e) => params.denoise = e.target.checked;
  $("wallFull").onchange = (e) => params.wallFull = e.target.checked;
  function updateRaysLabel() {
    const r = renderer;
    if (!r.wallGpu) { $("raysVal").textContent = "CPU"; return; }
    const n = (params.wall !== false ? (r.wallRays || 0) : 0) + (r.causRays || 0);
    $("raysVal").textContent = "×" + params.rayScale.toFixed(2) + " · " + (n >= 1000 ? Math.round(n / 1000) + "k" : n);
  }

  $("glow").oninput = (e) => {
    params.glow = parseFloat(e.target.value);
    $("glowVal").textContent = params.glow.toFixed(2);
  };
  $("speed").oninput = (e) => {
    sim.timeScale = parseFloat(e.target.value);
    $("speedVal").textContent = sim.timeScale.toFixed(2);
  };
  $("sg").oninput = (e) => {
    sim.gravityScale = parseFloat(e.target.value);
    $("sgVal").textContent = sim.gravityScale.toFixed(2);
  };
  $("heat").oninput = (e) => {
    sim.heatScale = parseFloat(e.target.value);
    $("heatVal").textContent = sim.heatScale.toFixed(2);
  };
  $("bulb").oninput = (e) => {
    sim.bulbHeight = parseFloat(e.target.value);
    $("bulbVal").textContent = sim.bulbHeight.toFixed(0);
  };
  $("edge").oninput = (e) => {
    sim.edgeFactor = parseFloat(e.target.value);
    $("edgeVal").textContent = sim.edgeFactor.toFixed(2);
  };
  $("diff").oninput = (e) => {
    sim.heatDiffScale = parseFloat(e.target.value);
    $("diffVal").textContent = sim.heatDiffScale.toFixed(2);
  };
  $("acool").oninput = (e) => {
    sim.ambientCoolScale = parseFloat(e.target.value);
    $("acoolVal").textContent = sim.ambientCoolScale.toFixed(2);
  };
  $("noise").oninput = (e) => {
    sim.heatNoise = parseFloat(e.target.value);
    $("noiseVal").textContent = sim.heatNoise.toFixed(2);
  };
  $("st").oninput = (e) => {
    sim.surfaceTension = parseFloat(e.target.value);
    $("stVal").textContent = sim.surfaceTension.toFixed(2);
  };
  $("tr").oninput = (e) => {
    sim.tempRepelMult = parseFloat(e.target.value);
    $("trVal").textContent = sim.tempRepelMult.toFixed(1);
  };
  $("sten").oninput = (e) => {
    sim.surfaceTensionScale = parseFloat(e.target.value);
    $("stenVal").textContent = sim.surfaceTensionScale.toFixed(2);
  };
  $("spring").oninput = (e) => {
    // Slider unit (1.00) corresponds to springScale = 0.0005 internally —
    // rebased so the previous "sweet spot" of slider 0.05 is now the
    // displayed default of 1.00. Slider 0 → no spring; slider 4 → 4× the
    // unit force (0.002 internal).
    const v = parseFloat(e.target.value);
    sim.springScale = v / 2000;
    $("springVal").textContent = v.toFixed(2);
  };
  // ---- Pool spring zone visualization ----
  function drawPoolRamp() {
    const cvs = $("poolRampViz");
    const ctx = cvs.getContext("2d");
    const W = cvs.width, H = cvs.height;
    ctx.clearRect(0, 0, W, H);

    const lo = sim.poolSpringLo;
    const hi = sim.poolSpringHi;
    const atten = sim.poolSpringAtten;
    // Map y-space [200..650] → canvas x [0..W]
    const Y_MIN = 200, Y_MAX = 650;
    const toX = (y) => ((y - Y_MIN) / (Y_MAX - Y_MIN)) * W;

    // Background gradient showing bulb regions
    const bg = ctx.createLinearGradient(0, 0, W, 0);
    bg.addColorStop(0, "rgba(255,200,50,0.08)");   // hot zone (top)
    bg.addColorStop(0.6, "rgba(150,20,30,0.12)");  // mid
    bg.addColorStop(1, "rgba(150,20,30,0.20)");    // pool zone (bottom)
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Draw the cosine ramp
    ctx.beginPath();
    const padY = 6;
    for (let px = 0; px < W; px++) {
      const y = Y_MIN + (px / W) * (Y_MAX - Y_MIN);
      let raw = (y - lo) / (hi - lo);
      raw = raw < 0 ? 0 : raw > 1 ? 1 : raw;
      const pool = 0.5 * (1 - Math.cos(Math.PI * raw));
      const springMult = 1.0 - atten * pool;
      const canvasY = padY + (1 - springMult) * (H - 2 * padY);
      if (px === 0) ctx.moveTo(px, canvasY);
      else ctx.lineTo(px, canvasY);
    }
    ctx.strokeStyle = "#FFE0B0";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Labels
    ctx.font = "9px monospace";
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.textAlign = "center";
    ctx.fillText("1.0×", 14, padY + 3);
    ctx.fillText((1 - atten).toFixed(1) + "×", 14, H - padY + 2);

    // Zone markers
    const xLo = toX(lo), xHi = toX(hi);
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(xLo, 0); ctx.lineTo(xLo, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xHi, 0); ctx.lineTo(xHi, H); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.font = "8px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Lo", xLo + 2, H - 2);
    ctx.textAlign = "right";
    ctx.fillText("Hi", xHi - 2, H - 2);
  }

  $("poolLo").oninput = (e) => {
    const v = parseInt(e.target.value);
    sim.poolSpringLo = v;
    $("poolLoVal").textContent = v;
    $("poolZoneVal").textContent = v + "–" + sim.poolSpringHi;
    drawPoolRamp();
  };
  $("poolHi").oninput = (e) => {
    const v = parseInt(e.target.value);
    sim.poolSpringHi = v;
    $("poolHiVal").textContent = v;
    $("poolZoneVal").textContent = sim.poolSpringLo + "–" + v;
    drawPoolRamp();
  };
  $("poolAtten").oninput = (e) => {
    const v = parseFloat(e.target.value);
    sim.poolSpringAtten = v;
    $("poolAttenVal").textContent = v.toFixed(2);
    drawPoolRamp();
  };
  // Initial draw
  requestAnimationFrame(drawPoolRamp);

  $("sticky").oninput = (e) => {
    sim.stickyStrength = parseFloat(e.target.value);
    $("stickyVal").textContent = sim.stickyStrength.toFixed(2);
  };
  $("size").oninput = (e) => {
    sim.renderScale = parseFloat(e.target.value);
    $("sizeVal").textContent = sim.renderScale.toFixed(2);
  };
  $("mass").oninput = (e) => {
    const newMass = parseFloat(e.target.value);
    // restDensity was calibrated to the OLD mass at reset time. Rescale
    // it proportionally so equilibrium pressure stays balanced when the
    // user drags the slider mid-simulation.
    if (sim.mass > 0) {
      sim.restDensity *= newMass / sim.mass;
    }
    sim.mass = newMass;
    $("massVal").textContent = sim.mass.toFixed(2);
  };
  $("pressure").oninput = (e) => {
    // Slider scales the SPH pressure stiffness (gasK). 1.00 ≙ baseline
    // gasK of 2400; 0 disables internal pressure entirely; 3 triples it.
    const v = parseFloat(e.target.value);
    sim.gasK = 2400 * v;
    $("pressureVal").textContent = v.toFixed(2);
  };
  $("visc").oninput = (e) => {
    sim.viscScale = parseFloat(e.target.value);
    $("viscVal").textContent = sim.viscScale.toFixed(2);
  };
  $("count").oninput = (e) => {
    const n = parseInt(e.target.value, 10);
    $("countVal").textContent = n;
    sim.setNumParticles(n);
  };
  $("reset").onclick = () => sim.reset();

  // --- Fullscreen + panel toggle ---
  const controlsEl = document.querySelector(".controls");
  if (window.innerWidth < 760) controlsEl.classList.add("hidden");
  $("togglePanel").onclick = () => controlsEl.classList.toggle("hidden");
  $("toggleFs").onclick = async () => {
    const inFs = document.fullscreenElement || document.webkitFullscreenElement;
    try {
      if (!inFs) {
        const el = document.documentElement;
        await (el.requestFullscreen ? el.requestFullscreen() : el.webkitRequestFullscreen?.());
      } else {
        await (document.exitFullscreen ? document.exitFullscreen() : document.webkitExitFullscreen?.());
      }
    } catch (_) { /* user-cancelled or unsupported */ }
  };
  // Keyboard shortcuts: F = fullscreen, C = toggle controls (ignored while typing in inputs)
  window.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
    if (e.key === "f" || e.key === "F") $("toggleFs").click();
    else if (e.key === "c" || e.key === "C") $("togglePanel").click();
  });

  // Presets
  const PRESETS = {
    classic: { bg: "#2a0d3e", cold: "#9a0f1a", hot: "#ffcf3a", comp: "#FFE0B0" },
    cosmic:  { bg: "#0d0a2e", cold: "#5e1ab8", hot: "#f068ff", comp: "#E8C0FF" },
    ocean:   { bg: "#031421", cold: "#0a4f8a", hot: "#5be3ff", comp: "#A0D8EF" },
    forest:  { bg: "#0d1f12", cold: "#1f5f2a", hot: "#c9ff5b", comp: "#E0F0A0" },
  };
  document.querySelectorAll(".preset").forEach((b) => {
    b.onclick = () => {
      const p = PRESETS[b.dataset.preset];
      if (!p) return;
      $("bgColor").value = p.bg;     params.bg   = hexToRgb(p.bg);
      $("coldColor").value = p.cold; params.cold = hexToRgb(p.cold);
      $("hotColor").value = p.hot;   params.hot  = hexToRgb(p.hot);
      if (p.comp) { $("compColor").value = p.comp; params.compressColor = hexToRgb(p.comp); }
    };
  });

  return { updateRaysLabel };
}
