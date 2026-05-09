#!/usr/bin/env node
// Batch runner for blobPop optimization — runs in phases that each complete
// within ~40 seconds so we can call them from a short-timeout shell.
//
// Usage:
//   node batch_blobpop.js lhs <startIdx> <endIdx>   — run LHS samples [start,end)
//   node batch_blobpop.js kalman                      — Kalman refinement using LHS results
//   node batch_blobpop.js validate                    — validation + baseline + final report

const fs = require('fs');
const path = require('path');
const { KERNELS, latinHypercube, ParamKalman } = require('./optimize.js');
const { SPH, SIM_H, SIM_W } = require('./sim.js');

const RESULTS_FILE = path.join(__dirname, '..', 'outputs', 'blobpop_lhs_results.json');
const FINAL_FILE = path.join(__dirname, '..', 'outputs', 'blobpop_final.json');
const SAMPLES_FILE = path.join(__dirname, '..', 'outputs', 'blobpop_samples.json');

const KERNEL_NAMES = ['blobPop'];
const TOTAL_SAMPLES = 200;
const MEASURE_TIME = 14;
const WARMUP_TIME = 8;
const KALMAN_ITERS = 5;

// ---- Trial runner (copied from optimize.js since it's not exported) ----
function runTrial(kernelNames, paramSets, warmupS, measureS) {
  const sim = new SPH({ numParticles: 160 });
  for (const name of kernelNames) {
    KERNELS[name].apply(sim, paramSets[name]);
  }
  const dt = 1 / 60;
  const substeps = 2;
  const subDt = dt / substeps;
  const maxWarmup = Math.max(...kernelNames.map(n => KERNELS[n].warmup));
  const warmupFrames = Math.floor(Math.max(warmupS, maxWarmup) / dt);
  const origHeat = sim.heatScale;
  sim.heatScale = origHeat * 4;
  for (let f = 0; f < warmupFrames; f++) {
    sim.stepFrame(dt);
    for (let s = 0; s < substeps; s++) sim.step(subDt);
  }
  sim.heatScale = origHeat;
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
  let totalCost = 0;
  const perKernel = {};
  for (const name of kernelNames) {
    const avg = counts[name] > 0 ? costs[name] / counts[name] : 0;
    perKernel[name] = avg;
    totalCost += avg * (KERNELS[name].weight || 1.0);
  }
  return { totalCost, perKernel };
}

// ---- Helpers ----
function getAllParams() {
  const allParams = {};
  for (const name of KERNEL_NAMES) {
    for (const [pName, vals] of Object.entries(KERNELS[name].params)) {
      allParams[pName] = vals;
    }
  }
  return allParams;
}

function sampleToParamSets(sample) {
  const ps = {};
  for (const name of KERNEL_NAMES) {
    ps[name] = {};
    for (const pName of Object.keys(KERNELS[name].params)) {
      ps[name][pName] = sample[pName];
    }
  }
  return ps;
}

function loadResults() {
  if (fs.existsSync(RESULTS_FILE)) {
    return JSON.parse(fs.readFileSync(RESULTS_FILE, 'utf8'));
  }
  return [];
}

function saveResults(results) {
  fs.writeFileSync(RESULTS_FILE, JSON.stringify(results, null, 2));
}

// ---- Phase: generate all samples once ----
function generateSamples() {
  if (fs.existsSync(SAMPLES_FILE)) {
    return JSON.parse(fs.readFileSync(SAMPLES_FILE, 'utf8'));
  }
  const allParams = getAllParams();
  const samples = latinHypercube(TOTAL_SAMPLES, allParams);
  fs.writeFileSync(SAMPLES_FILE, JSON.stringify(samples, null, 2));
  console.log(`Generated ${samples.length} LHS samples -> ${SAMPLES_FILE}`);
  return samples;
}

// ---- Phase: LHS batch ----
function runLHSBatch(startIdx, endIdx) {
  const samples = generateSamples();
  const existing = loadResults();
  const end = Math.min(endIdx, samples.length);

  console.log(`LHS batch: samples ${startIdx}-${end - 1} of ${samples.length}`);
  const t0 = Date.now();

  for (let i = startIdx; i < end; i++) {
    const paramSets = sampleToParamSets(samples[i]);
    const result = runTrial(KERNEL_NAMES, paramSets, WARMUP_TIME, MEASURE_TIME);
    existing.push({ idx: i, sample: samples[i], paramSets, ...result });

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const bestSoFar = Math.min(...existing.map(r => r.totalCost));
    console.log(`  ${i + 1}/${samples.length} cost=${result.totalCost.toFixed(2)} best=${bestSoFar.toFixed(2)} (${elapsed}s)`);
  }

  saveResults(existing);
  console.log(`Saved ${existing.length} total results to ${RESULTS_FILE}`);
}

// ---- Phase: Kalman refinement ----
function runKalman() {
  const results = loadResults();
  if (results.length === 0) { console.error('No LHS results found!'); process.exit(1); }

  results.sort((a, b) => a.totalCost - b.totalCost);
  const allParams = getAllParams();
  const paramNames = Object.keys(allParams);

  console.log(`\nKalman refinement: ${KALMAN_ITERS} iterations on ${results.length} LHS results`);
  console.log(`Best LHS cost: ${results[0].totalCost.toFixed(2)}`);

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

  const best = results[0];
  const normInit = normalize(best.sample);
  const kalman = new ParamKalman(normInit, 0.05, 0.002, 3.0);

  for (let iter = 0; iter < KALMAN_ITERS; iter++) {
    const cp = denormalize(kalman.x);
    const cps = sampleToParamSets(cp);
    const br = runTrial(KERNEL_NAMES, cps, WARMUP_TIME, MEASURE_TIME);

    const jac = [];
    for (let i = 0; i < paramNames.length; i++) {
      const ps = kalman.x.slice();
      ps[i] += 0.08;
      const pr = runTrial(KERNEL_NAMES, sampleToParamSets(denormalize(ps)), WARMUP_TIME, MEASURE_TIME);
      jac.push((pr.totalCost - br.totalCost) / 0.08);
    }

    kalman.update(br.totalCost, jac);
    console.log(`  iter ${iter + 1}: cost=${br.totalCost.toFixed(2)}`);

    for (let s = 0; s < Math.min(15, results.length); s++) {
      kalman.update(results[s].totalCost, jac.map(j => j * 0.5));
    }
  }

  const finalSample = denormalize(kalman.x);
  // Save for validation phase
  fs.writeFileSync(FINAL_FILE, JSON.stringify({ finalSample, paramNames, paramMin, paramMax }, null, 2));
  console.log(`Kalman done. Final params saved to ${FINAL_FILE}`);

  // Print top 5 from LHS
  console.log('\nTop 5 LHS parameter sets:');
  for (let i = 0; i < Math.min(5, results.length); i++) {
    const r = results[i];
    const kernelCosts = KERNEL_NAMES.map(n => `${n}=${r.perKernel[n].toFixed(2)}`).join(' ');
    console.log(`  #${i + 1} total=${r.totalCost.toFixed(2)} [${kernelCosts}]`);
    const pStr = paramNames.map(n => `${n}=${r.sample[n]}`).join(' ');
    console.log(`     ${pStr}`);
  }
}

// ---- Phase: Validation ----
function runValidation() {
  const finalData = JSON.parse(fs.readFileSync(FINAL_FILE, 'utf8'));
  const { finalSample, paramNames } = finalData;
  const results = loadResults();
  results.sort((a, b) => a.totalCost - b.totalCost);

  const finalPS = sampleToParamSets(finalSample);

  console.log('\nPhase 3: Validation (3 runs)...');
  let totalVal = 0;
  const valRuns = 3;
  for (let v = 0; v < valRuns; v++) {
    const c = runTrial(KERNEL_NAMES, finalPS, WARMUP_TIME, MEASURE_TIME).totalCost;
    totalVal += c;
    console.log(`  validation run ${v + 1}: cost=${c.toFixed(2)}`);
  }
  const avgVal = totalVal / valRuns;

  // Baseline
  console.log('\nBaseline (3 runs)...');
  const baselinePS = {};
  for (const name of KERNEL_NAMES) {
    baselinePS[name] = {};
    const defaultSim = new SPH({ numParticles: 160 });
    for (const pName of Object.keys(KERNELS[name].params)) {
      baselinePS[name][pName] = defaultSim[pName] !== undefined ? defaultSim[pName] :
        pName === 'springScale_x2k' ? defaultSim.springScale * 2000 :
        pName === 'closingSpeedK' ? 0.020 :
        pName === 'ceilDrag' ? 0.30 :
        pName === 'ceilPush' ? 4.0 :
        pName === 'localFraction' ? 0.20 :
        pName === 'connectDist_frac' ? (defaultSim.connectDist / defaultSim.h) :
        KERNELS[name].params[pName][Math.floor(KERNELS[name].params[pName].length / 2)];
    }
  }
  let totalBase = 0;
  for (let v = 0; v < valRuns; v++) {
    const c = runTrial(KERNEL_NAMES, baselinePS, WARMUP_TIME, MEASURE_TIME).totalCost;
    totalBase += c;
    console.log(`  baseline run ${v + 1}: cost=${c.toFixed(2)}`);
  }
  const avgBase = totalBase / valRuns;

  const improvement = avgBase > 0 ? ((1 - avgVal / avgBase) * 100) : 0;

  // Top 5
  console.log(`\n${'='.repeat(70)}`);
  console.log('TOP 5 LHS PARAMETER SETS:');
  console.log(`${'='.repeat(70)}`);
  for (let i = 0; i < Math.min(5, results.length); i++) {
    const r = results[i];
    const kernelCosts = KERNEL_NAMES.map(n => `${n}=${r.perKernel[n].toFixed(2)}`).join(' ');
    console.log(`  #${i + 1} total=${r.totalCost.toFixed(2)} [${kernelCosts}]`);
    const pStr = paramNames.map(n => `${n}=${r.sample[n]}`).join(' ');
    console.log(`     ${pStr}`);
  }

  console.log(`\n${'='.repeat(70)}`);
  console.log('OPTIMAL PARAMETERS:');
  console.log(`${'='.repeat(70)}`);
  for (const n of paramNames) {
    console.log(`  ${n}: ${typeof finalSample[n] === 'number' ? finalSample[n].toFixed(4) : finalSample[n]}`);
  }
  console.log(`\n  Optimized cost: ${avgVal.toFixed(2)}`);
  console.log(`  Baseline cost:  ${avgBase.toFixed(2)}`);
  console.log(`  Improvement:    ${improvement.toFixed(1)}%`);

  console.log('\n--- JSON ---');
  console.log(JSON.stringify({
    kernels: KERNEL_NAMES,
    optimal: finalSample,
    optimalCost: avgVal,
    baselineCost: avgBase,
    improvement,
    topLHS: results.slice(0, 5).map(r => ({ params: r.sample, cost: r.totalCost })),
  }, null, 2));

  console.log('\nDONE');
}

// ---- CLI ----
const cmd = process.argv[2];
if (cmd === 'lhs') {
  const start = parseInt(process.argv[3], 10) || 0;
  const end = parseInt(process.argv[4], 10) || (start + 25);
  runLHSBatch(start, end);
} else if (cmd === 'kalman') {
  runKalman();
} else if (cmd === 'validate') {
  runValidation();
} else {
  console.log('Usage: node batch_blobpop.js lhs <start> <end>');
  console.log('       node batch_blobpop.js kalman');
  console.log('       node batch_blobpop.js validate');
}
