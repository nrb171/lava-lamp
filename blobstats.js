#!/usr/bin/env node
// ============================================================
//  Headless blob-behaviour harness.
//
//  Runs the sim exactly as index.html does (60 fps, 5 substeps) with a
//  seeded RNG and reports:
//    tears/min   — a free blob (>= 8 particles) splitting into 2+ pieces
//                  of >= 3 particles each (what we want to suppress)
//    detach/min  — a new blob (>= 5 particles) leaving the pool
//                  (healthy lava-lamp behaviour — must stay > 0)
//    frags       — mean # of free fragments with 1-4 particles
//    aspect      — mean major/minor axis ratio of free blobs (1 = round)
//    jiggle      — mean internal (non-rigid) speed² of free blobs
//    aloft       — mean fraction of fluid particles in the upper 60%
//    ms/frame    — sim cost per rendered frame
//
//  Usage: node blobstats.js [seconds=90] [seeds=3] [key=value ...]
//  key=value pairs are assigned onto the SPH instance before running,
//  e.g. `node blobstats.js 60 2 capSigma=0 ruptureDist=0`.
// ============================================================

const { SPH, SIM_H } = require('./sim.js');

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function runOnce(seed, seconds, overrides) {
  Math.random = mulberry32(seed);
  const sim = new SPH({ numParticles: 160 });
  Object.assign(sim, overrides);
  const dt = 1 / 60, substeps = 5, sub = dt / substeps;
  const frames = Math.round(seconds / dt);
  const warm = Math.round(15 / dt);           // let the first blobs form
  const K = sim.MAX_BLOBS;
  const prev = new Int32Array(sim.cap);
  const tally = new Int32Array(K * K);
  const size = new Int32Array(K), prevSize = new Int32Array(K);

  let tears = 0, detaches = 0, fragSum = 0, aspSum = 0, aspN = 0;
  let jigSum = 0, jigN = 0, aloftSum = 0, samples = 0, simMs = 0;

  for (let f = 0; f < frames; f++) {
    prev.set(sim.groupId);
    const poolPrev = sim.groupId[0];
    const t0 = performance.now();
    sim.stepFrame(dt);
    for (let s = 0; s < substeps; s++) { sim.step(sub); sim.applyGrab(sub); }
    simMs += performance.now() - t0;
    if (f < warm) continue;

    // stepFrame regrouped at its start, so compare that grouping to the
    // one from the previous frame.
    const n = sim.n, nFixed = sim.nFixed, gid = sim.groupId;
    const pool = gid[0];
    tally.fill(0); size.fill(0); prevSize.fill(0);
    for (let i = nFixed; i < n; i++) {
      tally[prev[i] * K + gid[i]]++;
      size[gid[i]]++;
      prevSize[prev[i]]++;
    }
    for (let p = 1; p < K; p++) {
      if (prevSize[p] < 8) continue;
      let pieces = 0;
      for (let c = 0; c < K; c++) if (tally[p * K + c] >= 3) pieces++;
      if (p === poolPrev) {
        // pool shedding a blob: a non-pool group with >= 5 ex-pool particles
        for (let c = 1; c < K; c++) {
          if (c !== pool && tally[p * K + c] >= 5 && size[c] >= 5) detaches++;
        }
      } else if (pieces >= 2) {
        tears++;
      }
    }

    if (f % 6) continue;
    samples++;
    const cmx = new Float64Array(K), cmy = new Float64Array(K);
    const cvx = new Float64Array(K), cvy = new Float64Array(K);
    let aloft = 0;
    for (let i = nFixed; i < n; i++) {
      const g = gid[i];
      cmx[g] += sim.x[i]; cmy[g] += sim.y[i];
      cvx[g] += sim.vx[i]; cvy[g] += sim.vy[i];
      if (sim.y[i] < SIM_H * 0.6) aloft++;
    }
    aloftSum += aloft / (n - nFixed);
    const sxx = new Float64Array(K), syy = new Float64Array(K), sxy = new Float64Array(K);
    const jig = new Float64Array(K);
    for (let k = 0; k < K; k++) if (size[k]) {
      cmx[k] /= size[k]; cmy[k] /= size[k]; cvx[k] /= size[k]; cvy[k] /= size[k];
    }
    for (let i = nFixed; i < n; i++) {
      const g = gid[i];
      const dx = sim.x[i] - cmx[g], dy = sim.y[i] - cmy[g];
      sxx[g] += dx * dx; syy[g] += dy * dy; sxy[g] += dx * dy;
      const ux = sim.vx[i] - cvx[g], uy = sim.vy[i] - cvy[g];
      jig[g] += ux * ux + uy * uy;
    }
    for (let k = 1; k < K; k++) {
      if (k === pool || size[k] === 0) continue;
      if (size[k] <= 4) { fragSum++; continue; }
      if (size[k] < 8) continue;
      const a = sxx[k] / size[k], b = syy[k] / size[k], c = sxy[k] / size[k];
      const tr = a + b, det = a * b - c * c;
      const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
      const l1 = tr / 2 + disc, l2 = Math.max(1e-6, tr / 2 - disc);
      aspSum += Math.sqrt(l1 / l2); aspN++;
      jigSum += jig[k] / size[k]; jigN++;
    }
  }
  const minutes = (frames - warm) * dt / 60;
  return {
    tears: tears / minutes,
    detach: detaches / minutes,
    frags: fragSum / samples,
    aspect: aspN ? aspSum / aspN : NaN,
    jiggle: jigN ? jigSum / jigN : NaN,
    aloft: aloftSum / samples,
    ms: simMs / frames,
  };
}

if (require.main === module) {
  const args = process.argv.slice(2);
  const seconds = parseFloat(args[0]) || 90;
  const seeds = parseInt(args[1], 10) || 3;
  const overrides = {};
  for (const kv of args.slice(2).join(' ').split(/\s+/).filter(Boolean)) {
    const [k, v] = kv.split('=');
    overrides[k] = parseFloat(v);
  }
  const rows = [];
  for (let s = 1; s <= seeds; s++) rows.push(runOnce(s * 7919, seconds, overrides));
  const keys = Object.keys(rows[0]);
  const mean = {};
  for (const k of keys) mean[k] = rows.reduce((a, r) => a + r[k], 0) / rows.length;
  const fmt = (r) => keys.map(k => `${k}=${r[k].toFixed(2)}`).join('  ');
  rows.forEach((r, i) => console.log(`seed ${i + 1}: ${fmt(r)}`));
  console.log(`MEAN:   ${fmt(mean)}`);
}

module.exports = { runOnce };
