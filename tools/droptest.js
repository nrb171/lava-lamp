#!/usr/bin/env node
// Isolated-drop test for the surface-tension model: one elliptical blob,
// zero gravity, no heating. A correct surface-tension model pulls it
// toward a circle (Laplace pressure), stays in one piece, and — when
// viscosity is low relative to σ — overshoots and oscillates.
// Usage: node droptest.js [key=value ...]
import { pathToFileURL } from 'node:url';
import { SPH } from '../src/sim/sim.js';
function aspectOf(s) {
  let cx = 0, cy = 0, N = 0;
  for (let i = s.nFixed; i < s.n; i++) { cx += s.x[i]; cy += s.y[i]; N++; }
  cx /= N; cy /= N;
  let a = 0, b = 0, c = 0;
  for (let i = s.nFixed; i < s.n; i++) {
    const dx = s.x[i] - cx, dy = s.y[i] - cy; a += dx*dx; b += dy*dy; c += dx*dy;
  }
  a /= N; b /= N; c /= N;
  const d = Math.sqrt(0.25*(a-b)*(a-b) + c*c);
  // signed: > 1 when wider than tall, < 1 when taller than wide
  const r = Math.sqrt((0.5*(a+b)+d) / Math.max(1e-6, 0.5*(a+b)-d));
  return a >= b ? r : 1 / r;
}
function run(over, N = 40, ax = 2.2) {
  const s = new SPH({ numParticles: 160 });
  Object.assign(s, { gravity: 0, heatScale: 0, ambientCool: 0, heatNoise: 0,
                     stickyStrength: 0 }, over);
  // replace fluid with a hex-packed ellipse centred mid-bulb
  s.n = s.nFixed;
  const sp = s.h * 0.55, cx = 190, cy = 330;
  const R = Math.sqrt(N * sp * sp * 0.866 / Math.PI);
  const rx = R * Math.sqrt(ax), ry = R / Math.sqrt(ax);
  for (let row = -20; row <= 20; row++) for (let col = -20; col <= 20; col++) {
    const x = (col + (row & 1) * 0.5) * sp, y = row * sp * 0.866;
    if ((x/rx)**2 + (y/ry)**2 > 1) continue;
    const i = s.n++;
    s.x[i] = cx + x; s.y[i] = cy + y; s.vx[i] = 0; s.vy[i] = 0; s.temp[i] = s.tAmbient;
  }
  s.rebuildGrid(); s.prevGroupId.fill(0); s.recomputeGroups(); s.computeCentroids();
  const dt = 1/60, trace = [];
  let maxGroups = 1;
  for (let f = 0; f < 360; f++) {
    s.stepFrame(dt);
    for (let k = 0; k < 5; k++) s.step(dt / 5);
    // pieces of >= 3 particles (initial hex packing can leave 1-2 tip
    // particles outside connect range; those aren't a tear)
    const cnt = {}; for (let i = s.nFixed; i < s.n; i++) cnt[s.groupId[i]] = (cnt[s.groupId[i]] || 0) + 1;
    const pieces = Object.values(cnt).filter(c => c >= 3).length;
    maxGroups = Math.max(maxGroups, pieces);
    if (f % 15 === 14) trace.push(aspectOf(s).toFixed(2));
  }
  return { n: s.n - s.nFixed, maxGroups, trace: trace.join(' ') };
}
if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const over = {};
  for (const kv of process.argv.slice(2).join(' ').split(/\s+/).filter(Boolean)) { const [k, v] = kv.split('='); over[k] = parseFloat(v); }
  const r = run(over);
  console.log(`particles=${r.n} maxPieces=${r.maxGroups}\naspect every 0.25s: ${r.trace}`);
}
export { run };
