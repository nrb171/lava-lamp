// ============================================================
//  Lava-lamp SPH simulation — physics module
//  Separated from index.html for modularity and headless use.
// ============================================================

const SIM_W = 380;
const SIM_H = 700;

function bottleHalfFrac(t) {
  if (t < 0.03) return 0.0;
  if (t < 0.06) {
    const u = (t - 0.03) / 0.03;
    return 0.247 * (u * u * (3 - 2 * u));
  }
  if (t < 0.82) {
    const u = (t - 0.06) / 0.76;
    const e = u * u * (3 - 2 * u);
    return 0.247 + (0.50 - 0.247) * e;
  }
  if (t < 0.93) {
    const u = (t - 0.82) / 0.11;
    const e = u * u * (3 - 2 * u);
    return 0.50 + (0.40 - 0.50) * e;
  }
  if (t < 0.96) {
    const u = (t - 0.93) / 0.03;
    const e = u * u * (3 - 2 * u);
    return 0.40 + (0.32 - 0.40) * e;
  }
  if (t < 0.99) {
    const u = (t - 0.96) / 0.03;
    return 0.32 * (1 - u);
  }
  return 0.0;
}

function bottleHalfWidth(y) {
  return bottleHalfFrac(y / SIM_H) * SIM_W;
}

class SPH {
  constructor(opts = {}) {
    this.numParticles = opts.numParticles || 160;

    // Kernel and core fluid params
    this.h = 26;                  // smoothing radius (px)
    this.mass = 0.9;
    this.gasK = 2400 * 3.00;      // pressure stiffness
    this.viscosity = 0.2 / 26;    // base kinematic viscosity
    this.viscScale = 1.00;        // user multiplier
    this.cohesion = 0.55;
    this.surfaceTensionScale = 1.60;

    // Distinct masses ("blobs")
    this.MAX_BLOBS = 32;
    this.surfaceTension = 1.00;    // user multiplier for inter-blob repulsion
    this.interRepel = 60;
    this.tempRepelMult = 0.05;
    this.connectDist = this.h * 0.82;

    // Pair-wise spring binding
    this.springRest = this.h * 0.55;
    this.springK = 400;
    this.springDamp = 0.08;
    this.springMaxStretch = this.h * 0.4;
    this.springMin = this.h * 0.4;
    this.springReach = this.h * 1.3;
    this.springScale = 6.00 / 2000;

    // Pool-zone spring attenuation band (y-space)
    this.poolSpringLo = 480;   // spring starts fading here
    this.poolSpringHi = 580;   // spring is at minimum here
    this.poolSpringAtten = 0.6; // multiply spring by this in pool (0.6 = 60%)

    // Sticky bottom layer
    this.stickyHeight = 90;
    this.stickyStrength = 0.15;
    this.stickyPull = 2.00;

    // Pool-zone barrier attenuation
    this.poolZoneTop = SIM_H * 0.83;
    this.poolZoneFloor = SIM_H * 0.92;
    this.poolDwellTau = 3.0;
    this.poolBarrierFloor = 0.05;

    // Pinned wall particles
    this.MAX_FIXED = 60;

    // Cushion zone
    this.cushionRange = this.h * 1.35;
    this.cushionStrength = 1.50;

    // Time scale
    this.timeScale = 1.0;

    // Visual particle size
    this.renderScale = 1.30;

    // Physics
    this.gravity = 240;
    this.coolDensityExcess = 0.030;
    this.hotDensityDeficit = 0.045;
    this.gravityScale = 1.00;
    this.buoyancyExp = 4.0;
    this.coolMassRef = 8;

    // Heat
    this.tAmbient = 0.18;
    this.heatScale = 3.00;
    this.heatRate = 0.05;
    this.heatDiff = 0.05;
    this.heatDiffScale = 5.00;
    this.interDiffRatio = 0.03;
    this.ambientCool = 0.01;
    this.ambientCoolScale = 1.10;
    this.heatNoise = 0.50;
    this.simTime = 0;
    this.bulbHeight = 50;
    this.edgeFactor = 0.80;

    this.restDensity = 1;

    // Spatial hash grid
    this.cellSize = this.h;
    this.gridW = Math.ceil(SIM_W / this.cellSize) + 4;
    this.gridH = Math.ceil(SIM_H / this.cellSize) + 4;
    this.gridCount = this.gridW * this.gridH;
    this.cellHead = new Int32Array(this.gridCount);
    this.cellNext = new Int32Array(0);

    this.allocate(this.numParticles + this.MAX_FIXED);
    this.reset();
  }

  allocate(n) {
    this.n = 0;
    this.cap = n;
    this.x  = new Float32Array(n);
    this.y  = new Float32Array(n);
    this.vx = new Float32Array(n);
    this.vy = new Float32Array(n);
    this.fx = new Float32Array(n);
    this.fy = new Float32Array(n);
    this.density  = new Float32Array(n);
    this.compression = new Float32Array(n);  // per-particle compression ratio (0 = rest, >0 = squeezed)
    this.pressure = new Float32Array(n);
    this.temp = new Float32Array(n);
    this.dT   = new Float32Array(n);
    this.zoneDwell = new Float32Array(n);
    this.cellNext = new Int32Array(n);
    this.groupId = new Int32Array(n);
    this.prevGroupId = new Int32Array(n);
    this.uf = new Int32Array(n);

    const K = this.MAX_BLOBS || 8;
    this.cmx = new Float32Array(K);
    this.cmy = new Float32Array(K);
    this.cmn = new Int32Array(K);
    this.sumRoundX = new Float32Array(K);
    this.sumRoundY = new Float32Array(K);
    this.sumFrx = new Float32Array(K);
    this.sumFry = new Float32Array(K);
    this.blobZ = new Float32Array(K);
    this.blobSizeSmooth = new Float32Array(K);

    this._rootSizeKeys = new Int32Array(n);
    this._rootSizeVals = new Int32Array(n);
    this._rootSorted   = new Int32Array(n);
    this._rootIdMap    = new Int32Array(n);

    this._groupSize   = new Int32Array(K);
    this._majPrev     = new Int32Array(K);
    this._newBlobZ    = new Float32Array(K);
    this._handled     = new Uint8Array(K);
    this._prevTally   = new Int32Array(K * K);
    this._claimBuckets = new Int32Array(K * K);
    this._claimCounts  = new Int32Array(K);

    // Precomputed SPH kernel constants (depend only on h)
    const h = this.h;
    this.POLY6      =  4 / (Math.PI * Math.pow(h, 8));
    this.SPIKY_GRAD = -30 / (Math.PI * Math.pow(h, 5));
    this.VISC_LAP   =  40 / (Math.PI * Math.pow(h, 4));
  }

  // ---- Union-find helpers ----
  _ufFind(i) {
    let r = i;
    while (this.uf[r] !== r) r = this.uf[r];
    while (this.uf[i] !== r) { const next = this.uf[i]; this.uf[i] = r; i = next; }
    return r;
  }
  _ufUnion(a, b) {
    const ra = this._ufFind(a), rb = this._ufFind(b);
    if (ra !== rb) this.uf[ra] = rb;
  }

  computeCentroids() {
    const K = this.MAX_BLOBS;
    for (let k = 0; k < K; k++) { this.cmx[k] = 0; this.cmy[k] = 0; this.cmn[k] = 0; }
    for (let i = 0; i < this.n; i++) {
      const g = this.groupId[i];
      this.cmx[g] += this.x[i];
      this.cmy[g] += this.y[i];
      this.cmn[g]++;
    }
    for (let k = 0; k < K; k++) {
      if (this.cmn[k] > 0) {
        this.cmx[k] /= this.cmn[k];
        this.cmy[k] /= this.cmn[k];
      }
    }
  }

  setNumParticles(n) {
    const total = n + this.MAX_FIXED;
    if (total !== this.cap) this.allocate(total);
    this.numParticles = n;
    this.reset();
  }

  reset() {
    this.n = 0;
    // 1) Permanent pool: pinned wall particles
    const wallSpacingX = this.h * 0.55;
    const wallSpacingY = this.h * 0.48;
    const wallBaseY = SIM_H * 0.93;
    const numRows = 2;
    const perRow = Math.floor(this.MAX_FIXED / numRows);
    for (let row = 0; row < numRows; row++) {
      const wy = wallBaseY - row * wallSpacingY;
      const halfW = bottleHalfWidth(wy) - 6;
      if (halfW < 8) continue;
      const offsetX = (row % 2 === 1) ? wallSpacingX * 0.5 : 0;
      const fitsInRow = Math.floor((2 * halfW) / wallSpacingX);
      const numCol = Math.min(perRow, fitsInRow);
      if (numCol < 2) continue;
      const totalSpan = (numCol - 1) * wallSpacingX;
      const startX = SIM_W * 0.5 - totalSpan * 0.5;
      for (let i = 0; i < numCol; i++) {
        if (this.n >= this.MAX_FIXED) break;
        const wx = startX + i * wallSpacingX + offsetX;
        if (Math.abs(wx - SIM_W * 0.5) > halfW) continue;
        this.x[this.n] = wx;
        this.y[this.n] = wy;
        this.vx[this.n] = 0;
        this.vy[this.n] = 0;
        this.temp[this.n] = this.tAmbient;
        this.groupId[this.n] = 0;
        this.n++;
      }
    }
    this.nFixed = this.n;

    // 2) Place fluid particles above the pool
    const targetSpacing = this.h * 0.55;
    const minSpacing2 = (targetSpacing * 0.7) * (targetSpacing * 0.7);
    let placed = 0;
    let attempts = 0;
    const maxAttempts = this.numParticles * 400;
    const wallTopY = wallBaseY - wallSpacingY * (numRows - 1);
    while (placed < this.numParticles && attempts < maxAttempts) {
      attempts++;
      const y = wallTopY - 4 - Math.random() * 110;
      if (y < SIM_H * 0.10) continue;
      const halfW = bottleHalfWidth(y);
      if (halfW < 8) continue;
      const cx = SIM_W * 0.5;
      const x = cx + (Math.random() * 2 - 1) * (halfW - 6);
      let ok = true;
      for (let i = 0; i < this.n; i++) {
        const dx = this.x[i] - x;
        const dy = this.y[i] - y;
        if (dx*dx + dy*dy < minSpacing2) {
          ok = false; break;
        }
      }
      if (!ok) continue;
      this.x[this.n] = x;
      this.y[this.n] = y;
      this.vx[this.n] = (Math.random()-0.5) * 2;
      this.vy[this.n] = (Math.random()-0.5) * 2;
      this.temp[this.n] = this.tAmbient + (Math.random()-0.5) * 0.05;
      this.n++;
      placed++;
    }

    this.rebuildGrid();
    this.prevGroupId.fill(0);
    this.blobZ.fill(0);
    this.blobSizeSmooth.fill(0);
    this.recomputeGroups();
    this._updateBlobZ();
    this.computeCentroids();
    this.computeDensities();
    const densSlice = this.density.slice(0, this.n);
    densSlice.sort();
    const median = densSlice[Math.floor(densSlice.length / 2)] || 1;
    this.restDensity = median * 0.92;
  }

  cellIndex(x, y) {
    let cx = (x / this.cellSize) | 0;
    let cy = (y / this.cellSize) | 0;
    cx = Math.max(0, Math.min(this.gridW - 1, cx + 2));
    cy = Math.max(0, Math.min(this.gridH - 1, cy + 2));
    return cy * this.gridW + cx;
  }

  rebuildGrid() {
    this.cellHead.fill(-1);
    for (let i = 0; i < this.n; i++) {
      const idx = this.cellIndex(this.x[i], this.y[i]);
      this.cellNext[i] = this.cellHead[idx];
      this.cellHead[idx] = i;
    }
  }

  forNeighbors(i, fn) {
    const cx = Math.max(0, Math.min(this.gridW - 1, ((this.x[i] / this.cellSize) | 0) + 2));
    const cy = Math.max(0, Math.min(this.gridH - 1, ((this.y[i] / this.cellSize) | 0) + 2));
    for (let dy = -1; dy <= 1; dy++) {
      const ny = cy + dy;
      if (ny < 0 || ny >= this.gridH) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const nx = cx + dx;
        if (nx < 0 || nx >= this.gridW) continue;
        let j = this.cellHead[ny * this.gridW + nx];
        while (j !== -1) {
          fn(j);
          j = this.cellNext[j];
        }
      }
    }
  }

  computeDensities() {
    const n = this.n;
    const h = this.h, h2 = h * h;
    const POLY6 = this.POLY6;
    const m = this.mass;
    const nFixed = this.nFixed;
    const cellSize = this.cellSize;
    const gridW = this.gridW, gridH = this.gridH;
    const cellHead = this.cellHead, cellNext = this.cellNext;
    const px = this.x, py = this.y;
    const gid = this.groupId;
    const bz = this.blobZ;
    const cmn = this.cmn;

    for (let i = 0; i < n; i++) {
      let rho = 0;
      const xi = px[i], yi = py[i];
      const gi = gid[i];
      const isFluidI = i >= nFixed;

      const cxI = Math.max(0, Math.min(gridW - 1, ((xi / cellSize) | 0) + 2));
      const cyI = Math.max(0, Math.min(gridH - 1, ((yi / cellSize) | 0) + 2));
      for (let dy = -1; dy <= 1; dy++) {
        const ny = cyI + dy;
        if (ny < 0 || ny >= gridH) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cxI + dx;
          if (nx < 0 || nx >= gridW) continue;
          let j = cellHead[ny * gridW + nx];
          while (j !== -1) {
            const ddx = xi - px[j];
            const ddy = yi - py[j];
            const r2 = ddx*ddx + ddy*ddy;
            if (r2 < h2) {
              const w = h2 - r2;
              const term = m * POLY6 * w * w * w;
              const gj = gid[j];
              if (gj === gi) {
                rho += term;
              } else if (isFluidI && j >= nFixed) {
                const zi = bz[gi];
                const zj = bz[gj];
                const absZ = Math.abs(zi - zj);
                const sI = Math.max(1, cmn[gi]);
                const sJ = Math.max(1, cmn[gj]);
                const zReach = Math.min(0.55, (Math.sqrt(sI) + Math.sqrt(sJ)) * 0.07);
                if (absZ < zReach) {
                  rho += 0.3 * (1 - absZ / zReach) * term;
                }
              }
            }
            j = cellNext[j];
          }
        }
      }
      this.density[i] = rho;
    }
  }

  recomputeGroups() {
    const n = this.n;
    if (n === 0) return;
    for (let i = 0; i < n; i++) this.uf[i] = i;
    const cd2 = this.connectDist * this.connectDist;
    // Tighter connect distance for fluid-fluid pairs in the mid-bulb
    const fluidCD = this.connectDist * 0.82;
    const fluidCD2 = fluidCD * fluidCD;
    const poolZoneTop = this.poolZoneTop;
    const nFixed = this.nFixed;
    const cellSize = this.cellSize;
    const gridW = this.gridW, gridH = this.gridH;
    const cellHead = this.cellHead, cellNext = this.cellNext;
    const px = this.x, py = this.y;

    for (let i = 0; i < n; i++) {
      const xi = px[i], yi = py[i];
      const gi = this.groupId[i];
      const zi = this.blobZ[gi];
      const sizeI = Math.max(1, this.cmn[gi]);
      const isWallI = i < nFixed;

      const cxI = Math.max(0, Math.min(gridW - 1, ((xi / cellSize) | 0) + 2));
      const cyI = Math.max(0, Math.min(gridH - 1, ((yi / cellSize) | 0) + 2));
      for (let dy = -1; dy <= 1; dy++) {
        const ny = cyI + dy;
        if (ny < 0 || ny >= gridH) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cxI + dx;
          if (nx < 0 || nx >= gridW) continue;
          let j = cellHead[ny * gridW + nx];
          while (j !== -1) {
            if (j > i) {
              const ddx = xi - px[j];
              const ddy = yi - py[j];
              const isWallJ = j < nFixed;
              // Near the pool, use full connect distance so blobs absorb;
              // in the mid-bulb, use tighter threshold to prevent merging.
              const nearPool = (yi > poolZoneTop || py[j] > poolZoneTop);
              const effCD2 = (isWallI || isWallJ || nearPool) ? cd2 : fluidCD2;
              if (ddx*ddx + ddy*ddy < effCD2) {
                if (isWallI || isWallJ) {
                  this._ufUnion(i, j);
                } else {
                  const gj = this.groupId[j];
                  const zj = this.blobZ[gj];
                  const sizeJ = Math.max(1, this.cmn[gj]);
                  const sizeMin = Math.min(sizeI, sizeJ);
                  const zMerge = Math.max(0.10, 0.25 / Math.sqrt(sizeMin));
                  if (Math.abs(zi - zj) < zMerge) {
                    this._ufUnion(i, j);
                  }
                }
              }
            }
            j = cellNext[j];
          }
        }
      }
    }

    // Count component sizes by root
    const rootSizeVals = this._rootSizeVals;
    const rootIdMap = this._rootIdMap;
    rootSizeVals.fill(0);
    let numRoots = 0;
    const rootSizeKeys = this._rootSizeKeys;
    rootIdMap.fill(-1);

    for (let i = 0; i < n; i++) {
      const r = this._ufFind(i);
      if (rootIdMap[r] === -1) {
        rootIdMap[r] = numRoots;
        rootSizeKeys[numRoots] = r;
        numRoots++;
      }
      rootSizeVals[rootIdMap[r]]++;
    }

    // Sort roots by descending size
    const sorted = this._rootSorted;
    for (let i = 0; i < numRoots; i++) sorted[i] = i;
    for (let i = 1; i < numRoots; i++) {
      const key = sorted[i];
      const keyVal = rootSizeVals[key];
      let j = i - 1;
      while (j >= 0 && rootSizeVals[sorted[j]] < keyVal) {
        sorted[j + 1] = sorted[j];
        j--;
      }
      sorted[j + 1] = key;
    }

    const cap = this.MAX_BLOBS;
    rootIdMap.fill(0);
    for (let k = 0; k < numRoots; k++) {
      const rootIdx = sorted[k];
      rootIdMap[rootSizeKeys[rootIdx]] = Math.min(k + 1, cap - 1);
    }
    for (let i = 0; i < n; i++) {
      this.groupId[i] = rootIdMap[this._ufFind(i)];
    }
  }

  _updateBlobZ() {
    const n = this.n;
    const K = this.MAX_BLOBS;

    const groupSize = this._groupSize;
    const prevTally = this._prevTally;
    groupSize.fill(0);
    prevTally.fill(0);
    for (let i = 0; i < n; i++) {
      const cur = this.groupId[i];
      const prev = this.prevGroupId[i];
      groupSize[cur]++;
      prevTally[cur * K + prev]++;
    }

    const majPrev = this._majPrev;
    for (let c = 0; c < K; c++) {
      if (groupSize[c] === 0) { majPrev[c] = -1; continue; }
      let bestP = -1, bestCnt = 0;
      const base = c * K;
      for (let p = 1; p < K; p++) {
        const cnt = prevTally[base + p];
        if (cnt > bestCnt) { bestCnt = cnt; bestP = p; }
      }
      majPrev[c] = bestP;
    }

    const claimBuckets = this._claimBuckets;
    const claimCounts = this._claimCounts;
    claimCounts.fill(0);
    for (let c = 1; c < K; c++) {
      if (groupSize[c] === 0) continue;
      const p = majPrev[c];
      if (p < 0) continue;
      claimBuckets[p * K + claimCounts[p]] = c;
      claimCounts[p]++;
    }

    const newBlobZ = this._newBlobZ;
    const handled = this._handled;
    newBlobZ.fill(0);
    handled.fill(0);
    for (let p = 0; p < K; p++) {
      const cnt = claimCounts[p];
      if (cnt === 0) continue;
      const base = p * K;
      for (let a = 1; a < cnt; a++) {
        const key = claimBuckets[base + a];
        const keySize = groupSize[key];
        let b = a - 1;
        while (b >= 0 && groupSize[claimBuckets[base + b]] < keySize) {
          claimBuckets[base + b + 1] = claimBuckets[base + b];
          b--;
        }
        claimBuckets[base + b + 1] = key;
      }
      newBlobZ[claimBuckets[base]] = this.blobZ[p];
      handled[claimBuckets[base]] = 1;
      for (let k = 1; k < cnt; k++) {
        newBlobZ[claimBuckets[base + k]] = Math.random();
        handled[claimBuckets[base + k]] = 1;
      }
    }
    for (let c = 1; c < K; c++) {
      if (groupSize[c] > 0 && !handled[c]) {
        newBlobZ[c] = Math.random();
      }
    }

    for (let k = 0; k < K; k++) this.blobZ[k] = newBlobZ[k];
    for (let i = 0; i < this.nFixed; i++) {
      this.blobZ[this.groupId[i]] = 0.5;
    }

    // Smooth blob sizes
    const prevSmooth = this.blobSizeSmooth;
    const cmn = this.cmn;
    let maxN = 1;
    for (let k = 0; k < K; k++) if (cmn[k] > maxN) maxN = cmn[k];

    const newSmooth = this._newBlobZ;
    newSmooth.fill(0);
    for (let c = 1; c < K; c++) {
      if (groupSize[c] === 0) continue;
      const target = cmn[c] / maxN;
      const p = majPrev[c];
      if (p >= 0 && prevSmooth[p] > 0.001) {
        const inherited = prevSmooth[p];
        const rate = (target < inherited) ? 0.04 : 0.12;
        newSmooth[c] = inherited + (target - inherited) * rate;
      } else {
        newSmooth[c] = cmn[c] / maxN;
      }
    }
    for (let k = 0; k < K; k++) this.blobSizeSmooth[k] = newSmooth[k];
  }

  // Called once per frame — handles group computation, blob z, centroids,
  // and heat noise pre-computation. These don't need to run every substep.
  stepFrame(totalDt) {
    if (this.n === 0) return;
    this.rebuildGrid();
    this.prevGroupId.set(this.groupId);
    this.recomputeGroups();
    this._updateBlobZ();
    this.computeCentroids();

    // Identify the pool blob — largest group with centroid below poolZoneFloor
    let poolBlobId = -1, poolBlobSize = 0;
    for (let k = 1; k < this.MAX_BLOBS; k++) {
      if (this.cmn[k] > poolBlobSize && this.cmy[k] > this.poolZoneFloor) {
        poolBlobSize = this.cmn[k];
        poolBlobId = k;
      }
    }
    this._poolBlobId = poolBlobId;

    const T = this.simTime + totalDt;
    const d1 = T * 0.071, d2 = T * 0.103, d3 = T * 0.047;
    this._ph1 = T * 0.13;
    this._ph2 = T * 0.09;
    this._ph3 = T * 0.17;
    this._a1 = 0.30 + 0.30 * Math.sin(d1 * 1.3);
    this._a2 = 0.25 + 0.25 * Math.sin(d2 * 0.8 + 0.4);
    this._a3 = 0.20 + 0.20 * Math.sin(d3 * 1.5 + 1.2);
    this._k1 = 0.045 + 0.012 * Math.sin(d2 * 0.7);
    this._k2 = 0.030 + 0.010 * Math.sin(d3 * 0.9 + 2.1);
    this._k3 = 0.060 + 0.018 * Math.sin(d1 * 0.5 + 0.8);

    this._visc = this.viscosity * this.viscScale;
    this._repelPeak = this.interRepel * this.surfaceTension * this.cushionStrength;
    this._barrierWidth = this.cushionRange - this.connectDist;
    const cohBoost = 1.0 + 0.6 * Math.max(0, 1.0 - this.viscScale);
    this._cohScale = this.cohesion * this.surfaceTensionScale * cohBoost;
    this._buoDenom = Math.exp(this.buoyancyExp) - 1;
    this._buoTotal = this.coolDensityExcess + this.hotDensityDeficit;
    this._springK = this.springK * this.springScale;
    this._springDamp = this.springDamp * this.springScale;
  }

  step(dt) {
    if (this.n === 0) return;
    this.rebuildGrid();

    const n = this.n;
    const h = this.h, h2 = h * h;
    const POLY6      = this.POLY6;
    const SPIKY_GRAD = this.SPIKY_GRAD;
    const VISC_LAP   = this.VISC_LAP;
    const m = this.mass;
    const nFixed = this.nFixed;
    const cellSize = this.cellSize;
    const gridW = this.gridW, gridH = this.gridH;
    const cellHead = this.cellHead, cellNextArr = this.cellNext;
    const px = this.x, py = this.y;
    const pvx = this.vx, pvy = this.vy;
    const ptemp = this.temp;
    const gid = this.groupId;
    const bz = this.blobZ;
    const cmn = this.cmn;
    const pden = this.density;
    const ppres = this.pressure;

    // 1) Density & pressure (inlined neighbor walk)
    for (let i = 0; i < n; i++) {
      let rho = 0;
      const xi = px[i], yi = py[i];
      const gi = gid[i];
      const isFluidI = i >= nFixed;

      const cxI = Math.max(0, Math.min(gridW - 1, ((xi / cellSize) | 0) + 2));
      const cyI = Math.max(0, Math.min(gridH - 1, ((yi / cellSize) | 0) + 2));
      for (let ddy = -1; ddy <= 1; ddy++) {
        const ny = cyI + ddy;
        if (ny < 0 || ny >= gridH) continue;
        for (let ddx = -1; ddx <= 1; ddx++) {
          const nx = cxI + ddx;
          if (nx < 0 || nx >= gridW) continue;
          let j = cellHead[ny * gridW + nx];
          while (j !== -1) {
            const dx = xi - px[j];
            const dy = yi - py[j];
            const r2 = dx*dx + dy*dy;
            if (r2 < h2) {
              const w = h2 - r2;
              const term = m * POLY6 * w * w * w;
              const gj = gid[j];
              if (gj === gi) {
                rho += term;
              } else if (isFluidI && j >= nFixed) {
                const zi = bz[gi];
                const zj = bz[gj];
                const absZ = Math.abs(zi - zj);
                const sI = Math.max(1, cmn[gi]);
                const sJ = Math.max(1, cmn[gj]);
                const zReach = Math.min(0.55, (Math.sqrt(sI) + Math.sqrt(sJ)) * 0.07);
                if (absZ < zReach) {
                  rho += 0.3 * (1 - absZ / zReach) * term;
                }
              }
            }
            j = cellNextArr[j];
          }
        }
      }
      if (rho < this.restDensity) rho = this.restDensity;
      pden[i] = rho;
      ppres[i] = this.gasK * (rho - this.restDensity);
      // Compression ratio: 0 at rest density, rises as particle is squeezed.
      // Capped at 3.0 to keep shader values sane.
      const cRaw = rho / this.restDensity - 1.0;
      this.compression[i] = cRaw > 3.0 ? 3.0 : (cRaw > 0 ? cRaw : 0);
    }

    // Reset per-blob inter-blob force accumulators
    this.sumFrx.fill(0);
    this.sumFry.fill(0);

    // 2) Forces + heat exchange (inlined neighbor walk)
    const visc = this._visc;
    const repelPeak = this._repelPeak;
    const repelOuter = this.cushionRange;
    const repelOuter2 = repelOuter * repelOuter;
    const barrierWidth = this._barrierWidth;
    const cohScale = this._cohScale;
    const tempRepel = this.tempRepelMult;
    const poolDwellTau = this.poolDwellTau;
    const poolBarrierFloor = this.poolBarrierFloor;
    const buoK = this.buoyancyExp;
    const buoDenom = this._buoDenom;
    const buoTotal = this._buoTotal;
    const localFraction = this._optLocalFraction !== undefined ? this._optLocalFraction : 0.33;
    const springRest = this.springRest;
    const springK = this._springK;
    const springDamp = this._springDamp;
    const springMaxStretch = this.springMaxStretch;
    const springMin = this.springMin;
    const springReach2 = this.springReach * this.springReach;
    const interRatio = this.interDiffRatio;
    const sumFrx = this.sumFrx, sumFry = this.sumFry;
    const zoneDwell = this.zoneDwell;
    const POOL_LO = this.poolSpringLo, POOL_HI = this.poolSpringHi, POOL_INV = 1.0 / (POOL_HI - POOL_LO);
    const poolSpringAtten = this.poolSpringAtten;

    for (let i = 0; i < n; i++) {
      let fpx = 0, fpy = 0;
      let fvx = 0, fvy = 0;
      let fcx = 0, fcy = 0;
      let frx = 0, fry = 0;
      let dTsum = 0, neighCount = 0;
      const xi = px[i], yi = py[i];
      const vxi = pvx[i], vyi = pvy[i];
      const Pi = ppres[i];
      const rhoi = pden[i];
      const Ti = ptemp[i];
      const gi = gid[i];
      // Per-particle pool ramp for spring attenuation
      const _siRaw = (yi - POOL_LO) * POOL_INV;
      const _si = _siRaw < 0 ? 0 : _siRaw > 1 ? 1 : _siRaw;
      const poolI = 0.5 * (1 - Math.cos(Math.PI * _si));

      const cxI = Math.max(0, Math.min(gridW - 1, ((xi / cellSize) | 0) + 2));
      const cyI = Math.max(0, Math.min(gridH - 1, ((yi / cellSize) | 0) + 2));
      for (let ddy = -1; ddy <= 1; ddy++) {
        const ny = cyI + ddy;
        if (ny < 0 || ny >= gridH) continue;
        for (let ddx = -1; ddx <= 1; ddx++) {
          const nx = cxI + ddx;
          if (nx < 0 || nx >= gridW) continue;
          let j = cellHead[ny * gridW + nx];
          while (j !== -1) {
            if (j !== i) {
              const dx = xi - px[j];
              const dy = yi - py[j];
              const r2 = dx*dx + dy*dy;
              if (r2 < repelOuter2 && r2 > 1e-6) {
                const r = Math.sqrt(r2);
                const sameGroupJ = (gid[j] === gi);

                // Inter-blob repulsion
                if (!sameGroupJ) {
                  const tDelta = Math.abs(Ti - ptemp[j]);
                  const tBoost = 1.0 + tempRepel * tDelta;
                  const u = (repelOuter - r) / barrierWidth;
                  const dwell = (zoneDwell[i] + zoneDwell[j]) * 0.5;
                  const dwellFactor = poolBarrierFloor + (1.0 - poolBarrierFloor) * Math.exp(-dwell / poolDwellTau);
                  let zFactor = 1.0;
                  if (i >= nFixed && j >= nFixed) {
                    const zi = bz[gi];
                    const gj = gid[j];
                    const zj = bz[gj];
                    const zDiff = zi - zj;
                    const absZ = Math.abs(zDiff);
                    const sI = Math.max(1, cmn[gi]);
                    const sJ = Math.max(1, cmn[gj]);
                    const zReach = Math.min(0.55, (Math.sqrt(sI) + Math.sqrt(sJ)) * 0.07);
                    if (absZ >= zReach) {
                      zFactor = 0;
                    } else {
                      const fade = 1 - absZ / zReach;
                      zFactor = fade * (1 - 0.5 * zDiff / zReach);
                    }
                  }
                  // Closing-speed boost
                  const dvxR = vxi - pvx[j], dvyR = vyi - pvy[j];
                  const closingSpeed = -(dvxR * dx + dvyR * dy) / r;
                  const speedBoost = 1.0 + Math.max(0, closingSpeed) * 0.030;
                  const forceMag = repelPeak * tBoost * dwellFactor * zFactor * speedBoost * u * u;
                  const force = forceMag / r;
                  const fdx = force * dx, fdy = force * dy;
                  frx += localFraction * fdx;
                  fry += localFraction * fdy;
                  sumFrx[gi] += (1 - localFraction) * fdx;
                  sumFry[gi] += (1 - localFraction) * fdy;
                }

                // Standard SPH at r < h
                if (r2 < h2) {
                  let diffWeight;
                  if (sameGroupJ) {
                    const rhoj = pden[j];
                    const pTerm = -m * (Pi + ppres[j]) / (2 * rhoj) * SPIKY_GRAD * (h - r) * (h - r) / r;
                    fpx += pTerm * dx;
                    fpy += pTerm * dy;
                    const vTerm = visc * m / rhoj * VISC_LAP * (h - r);
                    fvx += vTerm * (pvx[j] - vxi);
                    fvy += vTerm * (pvy[j] - vyi);
                    const q = r / h;
                    let C;
                    if (q < 0.5) C = 2 * (1-q)*(1-q)*(1-q) * q*q*q - 1.0/64.0;
                    else         C = (1-q)*(1-q)*(1-q) * q*q*q;
                    const cTerm = -cohScale * m * 380 * C / r;
                    fcx += cTerm * dx;
                    fcy += cTerm * dy;
                    diffWeight = 1.0;
                  } else {
                    diffWeight = interRatio;
                  }
                  const wT = (h - r) / h;
                  dTsum += (ptemp[j] - Ti) * wT * diffWeight;
                  neighCount++;
                }

                // Long-reach spring (same-group only)
                if (sameGroupJ && r2 < springReach2) {
                  const dvx = vxi - pvx[j];
                  const dvy = vyi - pvy[j];
                  const vAxial = (dvx * dx + dvy * dy) / r;
                  // Per-particle cosine ramp: full spring above POOL_LO, half at POOL_HI
                  const _sjRaw = (py[j] - POOL_LO) * POOL_INV;
                  const _sj = _sjRaw < 0 ? 0 : _sjRaw > 1 ? 1 : _sjRaw;
                  const poolJ = 0.5 * (1 - Math.cos(Math.PI * _sj));
                  const effSpringK = springK * (1 - poolSpringAtten * poolI * poolJ);
                  if (r > springRest) {
                    const rawStretch = r - springRest;
                    const stretch = rawStretch < springMaxStretch ? rawStretch : springMaxStretch;
                    const sdMag = -effSpringK * stretch * stretch / springMaxStretch - springDamp * vAxial;
                    fcx += sdMag * dx / r;
                    fcy += sdMag * dy / r;
                  } else if (r < springMin) {
                    const compress = springMin - r;
                    const sdMag = effSpringK * compress * compress / springMaxStretch - springDamp * vAxial;
                    fcx += sdMag * dx / r;
                    fcy += sdMag * dy / r;
                  }
                }
              }
            }
            j = cellNextArr[j];
          }
        }
      }

      // Buoyancy
      const tNorm = Math.max(0, Math.min(1, (Ti - this.tAmbient) / (1.0 - this.tAmbient)));
      const riseFactor = (Math.exp(buoK * tNorm) - 1) / buoDenom;
      const densRatio = this.coolDensityExcess - buoTotal * riseFactor;
      const effG = this.gravity * densRatio * this.gravityScale;

      this.fx[i] = (fpx + fvx) / rhoi + fcx + frx;
      this.fy[i] = (fpy + fvy) / rhoi + fcy + effG + fry;
      this.dT[i] = (neighCount > 0 ? dTsum / neighCount : 0) * (this.heatDiff * this.heatDiffScale);
    }

    // Per-blob redistribution
    {
      const K = this.MAX_BLOBS;
      const fX = this.sumRoundX, fY = this.sumRoundY;
      for (let k = 0; k < K; k++) {
        const cnt = cmn[k];
        if (cnt > 0) {
          fX[k] = sumFrx[k] / cnt;
          fY[k] = sumFry[k] / cnt;
        } else {
          fX[k] = 0; fY[k] = 0;
        }
      }
      for (let i = nFixed; i < n; i++) {
        const g = gid[i];
        this.fx[i] += fX[g];
        this.fy[i] += fY[g];
      }
    }

    // 3) Integrate (fluid only — walls are pinned)
    const fluidBotY = SIM_H * 0.92;
    const stickyTop = fluidBotY - this.stickyHeight;
    const stickyDt = this.stickyStrength * dt;
    const stickyPullDt = this.stickyPull * this.stickyStrength * dt;
    // Ceiling cushion — back-pressure from incompressible fluid at top
    const ceilHeight = 60;
    const ceilBot = SIM_H * 0.07 + ceilHeight;
    const ceilDrag = 0.30;
    const ceilPush = 8.0;
    const ceilDragDt = ceilDrag * dt;
    const ceilPushDt = ceilPush * dt;
    const invMass = 1.0 / this.mass;
    for (let i = nFixed; i < n; i++) {
      pvx[i] += this.fx[i] * dt * invMass;
      pvy[i] += this.fy[i] * dt * invMass;
      pvx[i] *= 0.9995;
      pvy[i] *= 0.9995;
      if (py[i] > stickyTop) {
        const t = Math.min(1, (py[i] - stickyTop) / this.stickyHeight);
        const drag = Math.max(0, 1 - stickyDt * t);
        pvx[i] *= drag;
        pvy[i] *= drag;
        pvy[i] += stickyPullDt * t;
      }
      // Ceiling cushion: y-only drag + downward nudge
      if (py[i] < ceilBot) {
        const t = Math.min(1, (ceilBot - py[i]) / ceilHeight);
        const drag = Math.max(0, 1 - ceilDragDt * t);
        pvy[i] *= drag;
        pvy[i] += ceilPushDt * t;
      }
      const vmax = 600;
      const vlen2 = pvx[i]*pvx[i] + pvy[i]*pvy[i];
      if (vlen2 > vmax*vmax) {
        const s = vmax / Math.sqrt(vlen2);
        pvx[i] *= s; pvy[i] *= s;
      }
      px[i] += pvx[i] * dt;
      py[i] += pvy[i] * dt;
    }

    // 4) Heat sources & diffusion
    const hr = this.heatRate * this.heatScale;
    const bulbY = SIM_H * 0.93;
    const bH = this.bulbHeight;
    const cxFluid = SIM_W * 0.5;
    const edgeF = this.edgeFactor;
    this.simTime += dt;
    const nAmt = this.heatNoise;
    const ph1 = this._ph1, ph2 = this._ph2, ph3 = this._ph3;
    const a1 = this._a1, a2 = this._a2, a3 = this._a3;
    const k1 = this._k1, k2 = this._k2, k3 = this._k3;

    for (let i = 0; i < n; i++) {
      const distAbove = bulbY - py[i];
      if (distAbove > -10 && distAbove < bH) {
        const v = 1 - Math.max(0, distAbove) / bH;
        const halfW = Math.max(20, bottleHalfWidth(py[i]));
        const xNorm = Math.min(1, Math.abs(px[i] - cxFluid) / halfW);
        const hFactor = 1 - (1 - edgeF) * xNorm * xNorm;
        let noiseMul = 1.0;
        if (nAmt > 0) {
          const sn = (
            a1 * Math.sin(px[i] * k1 + ph1) +
            a2 * Math.sin(px[i] * k2 - py[i] * 0.012 + ph2) +
            a3 * Math.sin(px[i] * k3 + py[i] * 0.018 + ph3)
          );
          noiseMul = Math.max(0, 1 + nAmt * sn);
        }
        ptemp[i] += hr * v * v * hFactor * noiseMul * dt;
      }
      const blobN = Math.max(1, cmn[gid[i]]);
      const massFactor = Math.max(0.5, Math.sqrt(blobN / this.coolMassRef));
      const ambEff = this.ambientCool * this.ambientCoolScale;
      ptemp[i] -= (ambEff / massFactor) * (ptemp[i] - this.tAmbient) * dt;
      ptemp[i] += this.dT[i] * dt;
      if (ptemp[i] < 0) ptemp[i] = 0;
      if (ptemp[i] > 1.5) ptemp[i] = 1.5;

      if (py[i] > this.poolZoneTop) {
        const z = Math.min(1, (py[i] - this.poolZoneTop) / (this.poolZoneFloor - this.poolZoneTop));
        zoneDwell[i] += z * dt;
      } else {
        zoneDwell[i] = 0;
      }
    }

    // 5) Boundary handling
    const fluidTop = SIM_H * 0.07;
    const fluidBot = SIM_H * 0.92;
    for (let i = nFixed; i < n; i++) {
      if (py[i] < fluidTop) {
        py[i] = fluidTop;
        pvy[i] = Math.abs(pvy[i]) * 0.3;
      } else if (py[i] > fluidBot) {
        py[i] = fluidBot;
        pvy[i] = -Math.abs(pvy[i]) * 0.3;
      }
      const halfW = bottleHalfWidth(py[i]);
      const cx = SIM_W * 0.5;
      const limit = halfW - 4;
      if (limit < 4) continue;
      const off = px[i] - cx;
      if (off > limit) {
        px[i] = cx + limit;
        pvx[i] = -Math.abs(pvx[i]) * 0.3;
      } else if (off < -limit) {
        px[i] = cx - limit;
        pvx[i] = Math.abs(pvx[i]) * 0.3;
      }
    }
  }
}

// UMD export — works as both Node module and browser script
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { SPH, SIM_W, SIM_H, bottleHalfFrac, bottleHalfWidth };
}
