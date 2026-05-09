#!/usr/bin/env node
// ============================================================
//  Generalized Kalman parameter optimizer for SPH lava lamp
//
//  Usage:
//    node optimize.js                          # optimize all default kernels
//    node optimize.js --kernel breakup         # optimize one kernel
//    node optimize.js --kernel breakup,thermal # optimize a combination
//    node optimize.js --samples 80 --time 10   # custom sweep settings
//    node optimize.js --list                   # list available kernels
//
//  Each "kernel" is a named group of related parameters with its own
//  cost function. Kernels can be optimized individually or jointly
//  (combined cost). New kernels are added by registering them in the
//  KERNELS object below.
// ============================================================

const { SPH, SIM_H, SIM_W } = require('./sim.js');

// ============================================================
//  Kernel registry
//
//  Each kernel defines:
//    params:   { name: [value1, value2, ...] }  — parameter search grid
//    apply:    (sim, params) => void             — applies params to a sim instance
//    measure:  (sim, dt) => number               — samples cost at the current frame
//                                                  (called every few frames; lower = better)
//    warmup:   seconds of sim time before measuring
//    weight:   relative weight when combining kernels (default 1.0)
//    describe: human-readable description
// ============================================================

const KERNELS = {

  // ---- Blob breakup / cohesion ----
  breakup: {
    describe: 'Blob fragmentation — orphan particles, small blobs, breakup events',
    weight: 1.0,
    warmup: 5,
    params: {
      localFraction:  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
      springK:        [200, 300, 400, 500, 600],
      springScale_x2k:[1.5, 2.0, 2.5, 3.0, 3.7, 4.0],  // displayed as /2000, max 4
      ceilDrag:       [0.05, 0.12, 0.20, 0.30, 0.40],
      ceilPush:       [1.0, 2.0, 4.0, 6.0, 8.0],
      closingSpeedK:  [0.005, 0.012, 0.020, 0.030],
    },
    apply(sim, p) {
      // localFraction is a const inside step() — we patch it via a
      // property that the sim reads (requires sim.js to honour it).
      sim._optLocalFraction = p.localFraction;
      sim.springK = p.springK;
      sim.springScale = p.springScale_x2k / 2000;
      sim._optCeilDrag = p.ceilDrag;
      sim._optCeilPush = p.ceilPush;
      sim._optClosingSpeedK = p.closingSpeedK;
    },
    measure(sim) {
      const K = sim.MAX_BLOBS;
      const gc = new Int32Array(K);
      for (let i = sim.nFixed; i < sim.n; i++) gc[sim.groupId[i]]++;
      let orphans = 0, small = 0;
      for (let k = 1; k < K; k++) {
        if (gc[k] === 1) orphans++;
        else if (gc[k] > 0 && gc[k] <= 3) small += gc[k];
      }
      return orphans * 10 + small * 3;
    },
  },

  // ---- Thermal dynamics ----
  thermal: {
    describe: 'Heat transfer — time to first rise, temperature distribution uniformity',
    weight: 1.0,
    warmup: 2,
    params: {
      heatScale:       [0.50, 0.70, 0.90, 1.10, 1.40],
      heatDiffScale:   [1.5, 2.5, 3.2, 4.0, 5.0],
      ambientCoolScale:[0.5, 0.8, 1.15, 1.6, 2.2],
      heatNoise:       [0.5, 1.0, 1.5, 2.0],
      bulbHeight:      [50, 70, 85, 110, 140],
      edgeFactor:      [0.10, 0.21, 0.35, 0.50],
    },
    apply(sim, p) {
      sim.heatScale = p.heatScale;
      sim.heatDiffScale = p.heatDiffScale;
      sim.ambientCoolScale = p.ambientCoolScale;
      sim.heatNoise = p.heatNoise;
      sim.bulbHeight = p.bulbHeight;
      sim.edgeFactor = p.edgeFactor;
    },
    measure(sim) {
      // Penalize: no blobs rising (all cold), or everything too hot
      let hotCount = 0, coldCount = 0, totalTemp = 0;
      for (let i = sim.nFixed; i < sim.n; i++) {
        totalTemp += sim.temp[i];
        if (sim.temp[i] > 0.5) hotCount++;
        else if (sim.temp[i] < 0.25) coldCount++;
      }
      const n = sim.n - sim.nFixed;
      if (n === 0) return 100;
      const avgT = totalTemp / n;
      // Ideal: ~30% hot, ~50% cold, average temp ~0.3-0.4
      const hotRatio = hotCount / n;
      const coldRatio = coldCount / n;
      const hotPenalty = Math.abs(hotRatio - 0.30) * 20;
      const coldPenalty = Math.abs(coldRatio - 0.50) * 10;
      const tempPenalty = Math.abs(avgT - 0.35) * 15;
      return hotPenalty + coldPenalty + tempPenalty;
    },
  },

  // ---- Buoyancy / gravity ----
  buoyancy: {
    describe: 'Rise/sink dynamics — blob velocity, time aloft, cycling frequency',
    weight: 1.0,
    warmup: 6,
    params: {
      gravityScale:    [0.4, 0.6, 0.80, 1.0, 1.4],
      coolDensityExcess:[0.010, 0.015, 0.020, 0.030],
      hotDensityDeficit:[0.030, 0.045, 0.060, 0.080],
      buoyancyExp:     [2.0, 3.0, 4.0, 5.0, 6.0],
      stickyStrength:  [0.05, 0.10, 0.15, 0.25, 0.40],
    },
    apply(sim, p) {
      sim.gravityScale = p.gravityScale;
      sim.coolDensityExcess = p.coolDensityExcess;
      sim.hotDensityDeficit = p.hotDensityDeficit;
      sim.buoyancyExp = p.buoyancyExp;
      sim.stickyStrength = p.stickyStrength;
    },
    measure(sim) {
      // Count blobs in top, middle, bottom thirds — want activity in all
      const topY = SIM_H * 0.33, botY = SIM_H * 0.66;
      let top = 0, mid = 0, bot = 0;
      for (let i = sim.nFixed; i < sim.n; i++) {
        if (sim.y[i] < topY) top++;
        else if (sim.y[i] < botY) mid++;
        else bot++;
      }
      const n = sim.n - sim.nFixed;
      if (n === 0) return 100;
      // Penalize everything stuck at bottom or top
      const topR = top / n, midR = mid / n, botR = bot / n;
      const spread = Math.abs(topR - 0.15) + Math.abs(midR - 0.25) + Math.abs(botR - 0.60);
      // Also penalize zero velocity (nothing moving)
      let avgSpeed = 0;
      for (let i = sim.nFixed; i < sim.n; i++) {
        avgSpeed += Math.sqrt(sim.vx[i]*sim.vx[i] + sim.vy[i]*sim.vy[i]);
      }
      avgSpeed /= n;
      const speedPenalty = avgSpeed < 5 ? (5 - avgSpeed) * 2 : 0;
      return spread * 30 + speedPenalty;
    },
  },

  // ---- Surface tension / blob shape ----
  surface: {
    describe: 'Surface tension — blob roundness, cohesion, inter-blob repulsion',
    weight: 1.0,
    warmup: 5,
    params: {
      surfaceTension:       [0.8, 1.2, 1.65, 2.2, 3.0],
      surfaceTensionScale:  [0.5, 0.75, 1.0, 1.3, 1.6],
      cohesion:             [0.15, 0.25, 0.30, 0.40, 0.55],
      interRepel:           [40, 60, 80, 120, 160],
      tempRepelMult:        [0.3, 0.5, 0.8, 1.2, 1.6],
      cushionStrength:      [1.0, 1.5, 2.5, 3.5, 5.0],
    },
    apply(sim, p) {
      sim.surfaceTension = p.surfaceTension;
      sim.surfaceTensionScale = p.surfaceTensionScale;
      sim.cohesion = p.cohesion;
      sim.interRepel = p.interRepel;
      sim.tempRepelMult = p.tempRepelMult;
      sim.cushionStrength = p.cushionStrength;
    },
    measure(sim) {
      // Penalize: too many tiny blobs (poor cohesion) or everything merged
      const K = sim.MAX_BLOBS;
      const gc = new Int32Array(K);
      for (let i = sim.nFixed; i < sim.n; i++) gc[sim.groupId[i]]++;
      let blobCount = 0, maxBlob = 0, tinyBlobs = 0;
      for (let k = 1; k < K; k++) {
        if (gc[k] > 0) {
          blobCount++;
          if (gc[k] > maxBlob) maxBlob = gc[k];
          if (gc[k] <= 2) tinyBlobs++;
        }
      }
      const n = sim.n - sim.nFixed;
      if (n === 0) return 100;
      // Ideal: 2-6 distinct blobs, no tiny fragments, largest isn't everything
      const countPenalty = blobCount < 2 ? (2 - blobCount) * 10 :
                           blobCount > 8 ? (blobCount - 8) * 3 : 0;
      const tinyPenalty = tinyBlobs * 5;
      const monopoly = maxBlob / n > 0.85 ? 10 : 0;
      return countPenalty + tinyPenalty + monopoly;
    },
  },

  // ---- Viscosity / fluid feel ----
  viscosity: {
    describe: 'Fluid viscosity — flow smoothness, damping, velocity distribution',
    weight: 0.8,
    warmup: 4,
    params: {
      viscScale:   [0.3, 0.5, 0.7, 0.90, 1.2, 1.6],
      mass:        [0.7, 0.9, 1.1, 1.4, 1.8],
      springDamp:  [0.02, 0.05, 0.08, 0.12],
    },
    apply(sim, p) {
      sim.viscScale = p.viscScale;
      const oldMass = sim.mass;
      if (oldMass > 0) sim.restDensity *= p.mass / oldMass;
      sim.mass = p.mass;
      sim.springDamp = p.springDamp;
    },
    measure(sim) {
      // Penalize: too fast (chaotic) or too slow (frozen)
      let totalSpeed = 0, maxSpeed = 0;
      const n = sim.n - sim.nFixed;
      for (let i = sim.nFixed; i < sim.n; i++) {
        const s = Math.sqrt(sim.vx[i]*sim.vx[i] + sim.vy[i]*sim.vy[i]);
        totalSpeed += s;
        if (s > maxSpeed) maxSpeed = s;
      }
      if (n === 0) return 100;
      const avg = totalSpeed / n;
      // Ideal: avg speed 15-40 px/s, max < 200
      const avgPenalty = avg < 10 ? (10 - avg) * 1.5 :
                         avg > 60 ? (avg - 60) * 0.5 : 0;
      const maxPenalty = maxSpeed > 200 ? (maxSpeed - 200) * 0.1 : 0;
      return avgPenalty + maxPenalty;
    },
  },

  // ---- Pool merging ----
  poolMerge: {
    describe: 'Pool absorption — how well sinking blobs merge back into the resting pool',
    weight: 1.0,
    warmup: 8,
    params: {
      poolDwellTau:     [1.0, 2.0, 3.0, 5.0, 8.0],
      poolBarrierFloor: [0.01, 0.03, 0.05, 0.10, 0.20],
      stickyStrength:   [0.05, 0.10, 0.15, 0.25, 0.40],
      stickyPull:       [2.0, 3.0, 4.32, 6.0, 8.0],
    },
    apply(sim, p) {
      sim.poolDwellTau = p.poolDwellTau;
      sim.poolBarrierFloor = p.poolBarrierFloor;
      sim.stickyStrength = p.stickyStrength;
      sim.stickyPull = p.stickyPull;
    },
    measure(sim) {
      // Count small blobs lingering in the pool zone
      const poolTop = sim.poolZoneTop;
      const K = sim.MAX_BLOBS;
      const poolGC = new Int32Array(K);
      for (let i = sim.nFixed; i < sim.n; i++) {
        if (sim.y[i] > poolTop) poolGC[sim.groupId[i]]++;
      }
      // Find the main pool group (largest in the pool zone)
      let mainPool = 0, mainSize = 0;
      for (let k = 0; k < K; k++) {
        if (poolGC[k] > mainSize) { mainSize = poolGC[k]; mainPool = k; }
      }
      // Count stray small blobs in pool zone that aren't the main pool
      let strays = 0;
      for (let k = 1; k < K; k++) {
        if (k !== mainPool && poolGC[k] > 0 && poolGC[k] <= 5) {
          strays += poolGC[k];
        }
      }
      return strays * 8;
    },
  },

  // ---- Moderate blob population ----
  blobPop: {
    describe: 'Blob population — reward 3-5 moderate-sized blobs, penalize monopolies and fragments',
    weight: 1.2,
    warmup: 8,
    params: {
      surfaceTension:       [1.5, 2.2, 3.0, 4.0, 5.0],
      surfaceTensionScale:  [0.6, 0.9, 1.3, 1.7, 2.2],
      cohesion:             [0.20, 0.35, 0.55, 0.75, 1.0],
      interRepel:           [40, 80, 120, 180, 250],
      tempRepelMult:        [0.5, 1.0, 1.6, 2.5, 4.0],
      closingSpeedK:        [0.005, 0.015, 0.025, 0.040, 0.060],
      localFraction:        [0.08, 0.15, 0.20, 0.28, 0.38],
      springK:              [250, 400, 550, 700],
      connectDist_frac:     [0.65, 0.75, 0.82, 0.90, 1.0], // multiplier on h
    },
    apply(sim, p) {
      sim.surfaceTension = p.surfaceTension;
      sim.surfaceTensionScale = p.surfaceTensionScale;
      sim.cohesion = p.cohesion;
      sim.interRepel = p.interRepel;
      sim.tempRepelMult = p.tempRepelMult;
      sim._optClosingSpeedK = p.closingSpeedK;
      sim._optLocalFraction = p.localFraction;
      sim.springK = p.springK;
      sim.connectDist = sim.h * p.connectDist_frac;
    },
    measure(sim) {
      const K = sim.MAX_BLOBS;
      const n = sim.n - sim.nFixed;
      if (n === 0) return 200;

      // Only measure blobs ABOVE the pool zone — the resting pool at
      // the bottom is one giant blob by design and shouldn't count.
      const poolY = sim.poolZoneTop || SIM_H * 0.83;

      // Per-group: count + centroid-y (only for non-pool particles)
      const gc = new Int32Array(K);
      const gy = new Float64Array(K);   // sum of y positions
      for (let i = sim.nFixed; i < sim.n; i++) {
        const g = sim.groupId[i];
        gc[g]++;
        gy[g] += sim.y[i];
      }

      // Active particles: those in blobs whose centroid is above pool
      const activeGC = new Int32Array(K);  // active particle count per group
      let activeN = 0;
      for (let k = 1; k < K; k++) {
        if (gc[k] === 0) continue;
        const centroidY = gy[k] / gc[k];
        if (centroidY < poolY) {
          // Blob centroid is above pool — it's an active blob
          activeGC[k] = gc[k];
          activeN += gc[k];
        }
      }

      // If nothing is active yet (everything pooled), modest penalty
      if (activeN === 0) return 30;

      // Classify active blobs by size
      let moderate = 0;     // 8-50 particles — the sweet spot
      let large = 0;        // >50 — monopolising
      let small = 0;        // 4-7 — borderline
      let tiny = 0;         // 1-3 — fragments
      let largeTotal = 0;
      let moderateTotal = 0;
      const sizes = [];

      for (let k = 1; k < K; k++) {
        if (activeGC[k] === 0) continue;
        sizes.push(activeGC[k]);
        if (activeGC[k] <= 3)       tiny++;
        else if (activeGC[k] <= 7)  small++;
        else if (activeGC[k] <= 50) { moderate++; moderateTotal += activeGC[k]; }
        else                        { large++;    largeTotal += activeGC[k]; }
      }

      // Ideal: 3-5 moderate blobs
      const modCountPenalty = moderate < 3 ? (3 - moderate) * 12 :
                              moderate > 5 ? (moderate - 5) * 6 : 0;

      // Penalize fragments heavily
      const tinyPenalty = tiny * 8;

      // Penalize large monopolies among active blobs
      const monopolyPenalty = large * 15 + (activeN > 0 ? (largeTotal / activeN) * 20 : 0);

      // Reward moderate blobs holding a healthy share of active particles
      // Ideal: moderate blobs account for 60-90% of active particles
      const modShare = activeN > 0 ? moderateTotal / activeN : 0;
      const sharePenalty = modShare < 0.5 ? (0.5 - modShare) * 30 :
                           modShare > 0.90 ? (modShare - 0.90) * 15 : 0;

      // Penalize small borderline blobs
      const smallPenalty = small * 3;

      // Evenness: stddev of moderate blob sizes (prefer similar sizes)
      let evenPenalty = 0;
      if (moderate >= 2) {
        const modSizes = sizes.filter(s => s >= 8 && s <= 50);
        const mean = modSizes.reduce((a, b) => a + b, 0) / modSizes.length;
        const variance = modSizes.reduce((a, s) => a + (s - mean) * (s - mean), 0) / modSizes.length;
        evenPenalty = Math.sqrt(variance) * 0.5;
      }

      return modCountPenalty + tinyPenalty + monopolyPenalty + sharePenalty + smallPenalty + evenPenalty;
    },
  },
};


// ============================================================
//  Latin Hypercube Sampling
// ============================================================
function latinHypercube(n, paramDef) {
  const keys = Object.keys(paramDef);
  const perms = keys.map(k => {
    const vals = paramDef[k];
    const perm = [];
    for (let i = 0; i < n; i++) perm.push(i % vals.length);
    for (let i = perm.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [perm[i], perm[j]] = [perm[j], perm[i]];
    }
    return perm;
  });
  const samples = [];
  for (let i = 0; i < n; i++) {
    const sample = {};
    keys.forEach((k, di) => { sample[k] = paramDef[k][perms[di][i]]; });
    samples.push(sample);
  }
  return samples;
}


// ============================================================
//  Kalman Filter (multi-parameter, scalar measurement)
// ============================================================
class ParamKalman {
  constructor(init, diagCov, Q, R) {
    this.n = init.length;
    this.x = init.slice();
    this.P = Array.from({ length: this.n }, (_, i) =>
      Array.from({ length: this.n }, (_, j) => i === j ? diagCov : 0));
    this.Q = Q;
    this.R = R;
  }

  update(z, H) {
    const n = this.n;
    const Ph = new Array(n).fill(0);
    let hPh = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) Ph[i] += this.P[i][j] * H[j];
      hPh += H[i] * Ph[i];
    }
    const S = hPh + this.R;
    if (Math.abs(S) < 1e-12) return;
    const K = Ph.map(v => v / S);
    let pred = 0;
    for (let i = 0; i < n; i++) pred += H[i] * this.x[i];
    const innov = z - pred;
    for (let i = 0; i < n; i++) this.x[i] += K[i] * innov;
    const newP = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => this.P[i][j] - K[i] * Ph[j]));
    this.P = newP;
    for (let i = 0; i < n; i++) this.P[i][i] += this.Q;
  }
}


// ============================================================
//  Trial runner
// ============================================================
function runTrial(kernelNames, paramSets, warmupS, measureS) {
  const sim = new SPH({ numParticles: 160 });

  // Apply all kernel params
  for (const name of kernelNames) {
    KERNELS[name].apply(sim, paramSets[name]);
  }

  const dt = 1 / 60;
  const substeps = 2;
  const subDt = dt / substeps;

  // Warmup with boosted heat
  const maxWarmup = Math.max(...kernelNames.map(n => KERNELS[n].warmup));
  const warmupFrames = Math.floor(Math.max(warmupS, maxWarmup) / dt);
  const origHeat = sim.heatScale;
  sim.heatScale = origHeat * 4;
  for (let f = 0; f < warmupFrames; f++) {
    sim.stepFrame(dt);
    for (let s = 0; s < substeps; s++) sim.step(subDt);
  }
  sim.heatScale = origHeat;

  // Measure
  const measureFrames = Math.floor(measureS / dt);
  const costs = {};
  const counts = {};
  for (const name of kernelNames) { costs[name] = 0; counts[name] = 0; }

  for (let f = 0; f < measureFrames; f++) {
    sim.stepFrame(dt);
    for (let s = 0; s < substeps; s++) sim.step(subDt);
    if (f % 4 === 0) {
      for (const name of kernelNames) {
        costs[name] += KERNELS[name].measure(sim, dt);
        counts[name]++;
      }
    }
  }

  // Weighted combined cost
  let totalCost = 0;
  const perKernel = {};
  for (const name of kernelNames) {
    const avg = counts[name] > 0 ? costs[name] / counts[name] : 0;
    perKernel[name] = avg;
    totalCost += avg * (KERNELS[name].weight || 1.0);
  }

  return { totalCost, perKernel };
}


// ============================================================
//  Main optimizer
// ============================================================
function optimize(kernelNames, opts = {}) {
  const numSamples = opts.samples || 50;
  const measureTime = opts.time || 10;
  const warmupTime = opts.warmup || 5;
  const kalmanIters = opts.kalmanIters || 3;

  // Merge param grids from all selected kernels
  const allParams = {};
  for (const name of kernelNames) {
    const k = KERNELS[name];
    if (!k) { console.error(`Unknown kernel: ${name}`); process.exit(1); }
    for (const [pName, vals] of Object.entries(k.params)) {
      allParams[pName] = vals;
    }
  }
  const paramNames = Object.keys(allParams);

  console.log(`\n${'='.repeat(70)}`);
  console.log(`  Optimizing kernels: ${kernelNames.join(', ')}`);
  console.log(`  Parameters (${paramNames.length}): ${paramNames.join(', ')}`);
  console.log(`  Samples: ${numSamples}, Measure: ${measureTime}s, Warmup: ${warmupTime}s`);
  console.log(`${'='.repeat(70)}\n`);

  // LHS
  console.log('Phase 1: Latin Hypercube Sampling...');
  const samples = latinHypercube(numSamples, allParams);
  const results = [];

  for (let i = 0; i < samples.length; i++) {
    // Split combined sample into per-kernel param sets
    const paramSets = {};
    for (const name of kernelNames) {
      paramSets[name] = {};
      for (const pName of Object.keys(KERNELS[name].params)) {
        paramSets[name][pName] = samples[i][pName];
      }
    }
    const result = runTrial(kernelNames, paramSets, warmupTime, measureTime);
    results.push({ sample: samples[i], paramSets, ...result });
    if ((i + 1) % 10 === 0) {
      const best = Math.min(...results.map(r => r.totalCost));
      console.log(`  ${i + 1}/${numSamples} (best: ${best.toFixed(2)})`);
    }
  }
  results.sort((a, b) => a.totalCost - b.totalCost);

  // Report top results
  console.log('\nTop 5 parameter sets:');
  for (let i = 0; i < Math.min(5, results.length); i++) {
    const r = results[i];
    const kernelCosts = kernelNames.map(n => `${n}=${r.perKernel[n].toFixed(2)}`).join(' ');
    console.log(`  #${i + 1} total=${r.totalCost.toFixed(2)} [${kernelCosts}]`);
    const pStr = paramNames.map(n => `${n}=${r.sample[n]}`).join(' ');
    console.log(`     ${pStr}`);
  }

  // Phase 2: Kalman refinement
  console.log(`\nPhase 2: Kalman refinement (${kalmanIters} iterations)...`);
  const paramMin = paramNames.map(n => Math.min(...allParams[n]));
  const paramMax = paramNames.map(n => Math.max(...allParams[n]));
  const paramSpan = paramMin.map((mn, i) => paramMax[i] - mn);

  function normalize(sample) {
    return paramNames.map((n, i) => paramSpan[i] > 0 ? (sample[n] - paramMin[i]) / paramSpan[i] : 0.5);
  }
  function denormalize(state) {
    const s = {};
    paramNames.forEach((n, i) => {
      s[n] = paramMin[i] + state[i] * paramSpan[i];
      s[n] = Math.max(paramMin[i], Math.min(paramMax[i], s[n]));
    });
    return s;
  }
  function sampleToParamSets(sample) {
    const ps = {};
    for (const name of kernelNames) {
      ps[name] = {};
      for (const pName of Object.keys(KERNELS[name].params)) {
        ps[name][pName] = sample[pName];
      }
    }
    return ps;
  }

  const best = results[0];
  const normInit = normalize(best.sample);
  const kalman = new ParamKalman(normInit, 0.05, 0.002, 3.0);

  for (let iter = 0; iter < kalmanIters; iter++) {
    const cp = denormalize(kalman.x);
    const cps = sampleToParamSets(cp);
    const br = runTrial(kernelNames, cps, warmupTime, measureTime);

    // Numerical Jacobian
    const jac = [];
    for (let i = 0; i < paramNames.length; i++) {
      const ps = kalman.x.slice();
      ps[i] += 0.08;
      const pr = runTrial(kernelNames, sampleToParamSets(denormalize(ps)), warmupTime, measureTime);
      jac.push((pr.totalCost - br.totalCost) / 0.08);
    }

    kalman.update(br.totalCost, jac);
    console.log(`  iter ${iter + 1}: cost=${br.totalCost.toFixed(2)}`);

    // Feed top LHS results
    for (let s = 0; s < Math.min(15, results.length); s++) {
      kalman.update(results[s].totalCost, jac.map(j => j * 0.5));
    }
  }

  // Final result
  const finalSample = denormalize(kalman.x);
  const finalPS = sampleToParamSets(finalSample);

  // Validate
  console.log('\nPhase 3: Validation...');
  let totalVal = 0;
  const valRuns = 3;
  for (let v = 0; v < valRuns; v++) {
    totalVal += runTrial(kernelNames, finalPS, warmupTime, measureTime).totalCost;
  }
  const avgVal = totalVal / valRuns;

  // Baseline
  let totalBase = 0;
  const baselinePS = {};
  for (const name of kernelNames) {
    baselinePS[name] = {};
    // Use SPH defaults as baseline
    const defaultSim = new SPH({ numParticles: 160 });
    for (const pName of Object.keys(KERNELS[name].params)) {
      baselinePS[name][pName] = defaultSim[pName] !== undefined ? defaultSim[pName] :
        pName === 'springScale_x2k' ? defaultSim.springScale * 2000 :
        pName === 'closingSpeedK' ? 0.020 :
        pName === 'ceilDrag' ? 0.30 :
        pName === 'ceilPush' ? 4.0 :
        pName === 'localFraction' ? 0.20 :
        KERNELS[name].params[pName][Math.floor(KERNELS[name].params[pName].length / 2)];
    }
  }
  for (let v = 0; v < valRuns; v++) {
    totalBase += runTrial(kernelNames, baselinePS, warmupTime, measureTime).totalCost;
  }
  const avgBase = totalBase / valRuns;

  console.log(`\n${'='.repeat(70)}`);
  console.log('OPTIMAL PARAMETERS:');
  console.log(`${'='.repeat(70)}`);
  for (const n of paramNames) {
    console.log(`  ${n}: ${typeof finalSample[n] === 'number' ? finalSample[n].toFixed(4) : finalSample[n]}`);
  }
  console.log(`\n  Optimized cost: ${avgVal.toFixed(2)}`);
  console.log(`  Baseline cost:  ${avgBase.toFixed(2)}`);
  const improvement = avgBase > 0 ? ((1 - avgVal / avgBase) * 100) : 0;
  console.log(`  Improvement:    ${improvement.toFixed(1)}%`);

  // JSON output
  console.log('\n--- JSON ---');
  console.log(JSON.stringify({
    kernels: kernelNames,
    optimal: finalSample,
    optimalCost: avgVal,
    baselineCost: avgBase,
    improvement,
    topLHS: results.slice(0, 3).map(r => ({ params: r.sample, cost: r.totalCost })),
  }, null, 2));

  return { optimal: finalSample, cost: avgVal, improvement };
}


// ============================================================
//  CLI
// ============================================================
function main() {
  const args = process.argv.slice(2);

  if (args.includes('--list')) {
    console.log('\nAvailable kernels:\n');
    for (const [name, k] of Object.entries(KERNELS)) {
      const pCount = Object.keys(k.params).length;
      console.log(`  ${name.padEnd(12)} (${pCount} params, weight ${k.weight}) — ${k.describe}`);
      for (const [pName, vals] of Object.entries(k.params)) {
        console.log(`    ${pName}: [${vals.join(', ')}]`);
      }
      console.log();
    }
    process.exit(0);
  }

  let kernelNames = Object.keys(KERNELS); // default: all
  let numSamples = 50;
  let measureTime = 10;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--kernel' && args[i + 1]) {
      kernelNames = args[++i].split(',').map(s => s.trim());
    } else if (args[i] === '--samples' && args[i + 1]) {
      numSamples = parseInt(args[++i], 10);
    } else if (args[i] === '--time' && args[i + 1]) {
      measureTime = parseFloat(args[++i]);
    }
  }

  // Validate kernel names
  for (const name of kernelNames) {
    if (!KERNELS[name]) {
      console.error(`Unknown kernel "${name}". Use --list to see available kernels.`);
      process.exit(1);
    }
  }

  optimize(kernelNames, { samples: numSamples, time: measureTime });
}

if (require.main === module) {
  main();
}

module.exports = { KERNELS, optimize, ParamKalman, latinHypercube };
