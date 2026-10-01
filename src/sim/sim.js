// ============================================================
//  Lava-lamp SPH simulation — physics module (an ES module: used by the
//  page, src/main.js, and headless by the tools in tools/)
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
    this.gasK = 2400 * 5.00;      // pressure stiffness
    // Near pressure (Clavet, Beaudoin & Poulin 2005): a short-range
    // repulsion between particles of the same body, from a "near density"
    // Σ(1 − r/h)³ that grows without bound as particles crowd. Ordinary
    // pressure per unit mass, P/ρ, levels off at gasK however hard a body is
    // squeezed, while the pull of its springs and cohesion grows with its
    // number of neighbours — so a big, cohesive blob could crush itself to
    // a point. This keeps particles apart whatever the cohesion.
    this.nearK = 40;
    this.viscosity = 0.2 / 26;    // base kinematic viscosity
    this.viscScale = 0.50;        // user multiplier (lower = bouncier: Ohnesorge ∝ ν/√(σR))
    this.cohesion = 0.55;
    this.surfaceTensionScale = 5.00;   // UI "Surface tension": scales cohesion AND capSigma

    // Capillary (surface-tension) restoring force on each free blob's
    // lowest shape mode — see _applyCapillaryMode() for the derivation.
    // capSigma is the kinematic surface tension σ/ρ (sim px³/s²), before
    // surfaceTensionScale. At the default scale σ/ρ = 33000, a Bond number
    // Bo = g'R²/(σ/ρ) ≈ 0.3 for a typical 12-particle blob (R ≈ 30 px,
    // g' = gravity·hotDensityDeficit ≈ 11 px/s²): surface tension beats
    // buoyancy, so rising blobs stay round. (Real lamps sit near Bo ≈ 1-2.)
    this.capSigma = 30000;
    this.capMinN = 5;        // fewer particles than this: no resolvable shape
    // Viscous damping of each free blob's shape change (1/s): the
    // traceless part of its best-fit linear velocity field — the rate it
    // stretches or shears — is damped, leaving its motion, spin and the
    // pool alone. Without it the capillary mode rings: blobs wobble between
    // ellipses rather than settling round (stiffer surface tension only
    // made them ring faster).
    this.shapeDamp = 3;

    // Distinct masses ("blobs")
    this.MAX_BLOBS = 32;
    this.surfaceTension = 1.00;    // user multiplier for inter-blob repulsion
    this.interRepel = 60;
    this.tempRepelMult = 0.05;
    this.connectDist = this.h * 0.82;
    // Rupture distance. Coalescence and pinch-off are different physical
    // events: two blobs merge only once the film between them drains (the
    // tight connectDist), but a continuous body only splits once its neck
    // thins past rupture. So fluid particles already in the same blob stay
    // linked out to ruptureDist. Without this hysteresis, a wobbling blob
    // flickers between one group and two — swapping cohesion for
    // inter-blob repulsion — and tears itself apart.
    this.ruptureDist = this.h * 0.95;   // must stay < h (3×3 cell search)


    // Pair-wise spring binding
    this.springRest = this.h * 0.55;
    this.springK = 400;
    this.springDamp = 0.08;
    this.springMaxStretch = this.h * 0.4;
    this.springMin = this.h * 0.4;
    this.springReach = this.h * 1.3;
    this.springScale = 12.00 / 2000;

    // Pool-zone spring attenuation band (y-space)
    this.poolSpringLo = 480;   // spring starts fading here
    this.poolSpringHi = 580;   // spring is at minimum here
    this.poolSpringAtten = 0.6; // multiply spring by this in pool (0.6 = 60%)

    // Sticky bottom layer
    this.stickyHeight = 90;
    this.stickyStrength = 0.80;
    this.stickyPull = 2.00;

    // Pool-zone barrier attenuation
    this.poolZoneTop = SIM_H * 0.83;
    this.poolZoneFloor = SIM_H * 0.92;
    this.poolDwellTau = 3.0;
    this.poolBarrierFloor = 0.05;

    // Pool/wall particles — anchored by a heavily-damped harmonic oscillator
    // (no longer fully pinned). Each particle is pulled back to its home
    // position (set in reset()) by a stiff spring; per-substep velocity is
    // multiplied by wallDampFactor so motion stays small and well-controlled.
    this.MAX_FIXED = 60;
    this.wallSpringK    = 1500;   // restoring stiffness toward home
    this.wallDampFactor = 0.30;   // velocity retained per substep (heavy)
    this.wallMaxDisplace = 4.0;   // hard clamp on displacement (sim px)

    // Cushion zone
    this.cushionRange = this.h * 1.35;
    this.cushionStrength = 1.50;

    // Mouse "grab" — when the user clicks and holds, fluid particles
    // near the cursor are pulled toward an (offset-preserved) target
    // with a per-particle Gaussian-weighted spring + damping. The
    // weight falls off smoothly with distance from the click point,
    // so the centre of the cluster is tugged firmly while the fringes
    // are only nudged. A separately-smoothed "soft target" lags a bit
    // behind the raw cursor, giving the whole interaction a draggy
    // feel rather than snapping rigidly to the mouse.
    this.grab = {
      active: false,
      tx: 0,  ty: 0,         // raw target (cursor) position, sim coords
      stx: 0, sty: 0,        // smoothed target — what the spring pulls to
      tvx: 0, tvy: 0,        // low-pass cursor velocity, sim px/s
      springK: 90,           // spring stiffness (acc / sim px), per-particle scaled by w
      damp: 14,              // PURE friction toward zero (1/s) — higher = more drag
      vCouple: 0.30,         // 0..1 — how much cursor velocity transfers to grabbed particles
      targetSmoothK: 9,      // how fast stx,sty chase tx,ty (1/s); lower = more drag
      radius: 40,            // outer pickup cutoff (sim px) — hard zero past this
      sigma: 22,             // Gaussian std-dev for the weight falloff
      weightThresh: 0.02,    // ignore particles whose weight is below this
      maxParticles: 100,     // safety cap on how many to grab at once
      particles: [],         // [{idx, offX, offY, w}]
    };

    // Drag of the surrounding liquid on the wax: velocity kept per substep
    this.velDamp = 0.9998;

    // Time scale
    this.timeScale = 1.0;

    // Visual particle size
    this.renderScale = 1.30;

    // Physics
    this.gravity = 240;
    this.coolDensityExcess = 0.030;
    this.hotDensityDeficit = 0.045;
    this.gravityScale = 1.50;
    this.buoyancyExp = 3.0;
    // Buoyancy of a free blob comes from its density as a whole — its mean
    // temperature — not each particle's own: per-particle buoyancy makes a
    // blob with a hot side and a cool side pull itself apart, and turning
    // buoyancy up then tears blobs instead of lifting them. This is the
    // share taken from the blob's mean (the rest per particle). The pool —
    // any group centred in the pool zone, joined to the base or not — stays
    // per particle, so hot plumes rise out of it rather than the whole pool
    // lifting off as one blob.
    this.blobBuoyancy = 1.0;
    this.coolMassRef = 8;

    // Heat
    this.tAmbient = 0.18;
    this.heatScale = 4.50;
    this.heatRate = 0.05;
    this.heatDiff = 0.05;
    this.heatDiffScale = 5.00;
    this.interDiffRatio = 0.03;
    this.ambientCool = 0.01;
    this.ambientCoolScale = 3.50;
    this.heatNoise = 0.50;
    this.simTime = 0;
    this.bulbHeight = 58;
    this.edgeFactor = 0.30;

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
    // Home (anchor) position for pool/wall particles; fluid particles ignore.
    this.homeX = new Float32Array(n);
    this.homeY = new Float32Array(n);
    this.vx = new Float32Array(n);
    this.vy = new Float32Array(n);
    this.fx = new Float32Array(n);
    this.fy = new Float32Array(n);
    this.density  = new Float32Array(n);
    this.compression = new Float32Array(n);  // per-particle compression ratio
    this.nearDensity = new Float32Array(n);  // Σ (1 − r/h)³ over the same body (near pressure)
    this.pressure = new Float32Array(n);
    this.temp = new Float32Array(n);
    this.dT   = new Float32Array(n);
    this.zoneDwell = new Float32Array(n);
    // Per-step, per-particle factors hoisted out of the pair loop
    this._dwellExp = new Float32Array(n);   // exp(-dwell / (2·poolDwellTau))
    this._poolRamp = new Float32Array(n);   // cosine pool-zone ramp on y
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
    this._capAcc = new Float64Array(K * 8);
    this._sqrtSize = new Float32Array(K);   // √max(1, blob size), per frame

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
    this.resetCount = (this.resetCount || 0) + 1;   // lets the renderer drop stale per-particle state
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
    // Snapshot anchor positions for the pool oscillator.
    for (let i = 0; i < this.nFixed; i++) {
      this.homeX[i] = this.x[i];
      this.homeY[i] = this.y[i];
    }

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
    const ruptureD2 = this.ruptureDist * this.ruptureDist;
    const prevGid = this.prevGroupId;
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
              let effCD2 = (isWallI || isWallJ || nearPool) ? cd2 : fluidCD2;
              // Same blob last frame → stays linked until the neck ruptures.
              // (prevGid 0 only occurs right after reset.)
              if (!isWallI && !isWallJ && prevGid[i] === prevGid[j] &&
                  prevGid[i] !== 0 && ruptureD2 > effCD2) {
                effCD2 = ruptureD2;
              }
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

    for (let k = 0; k < this.MAX_BLOBS; k++) this._sqrtSize[k] = Math.sqrt(Math.max(1, this.cmn[k]));

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
    const sqrtSize = this._sqrtSize;
    const nearK = this.nearK;

    // 1) Density & pressure (inlined neighbor walk)
    const pnear = this.nearDensity;
    for (let i = 0; i < n; i++) {
      let rho = 0, rhoNear = 0;
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
                const qn = 1 - Math.sqrt(r2) / h;
                rhoNear += qn * qn * qn;
              } else if (isFluidI && j >= nFixed) {
                const absZ = Math.abs(bz[gi] - bz[gj]);
                const zReach = Math.min(0.55, (sqrtSize[gi] + sqrtSize[gj]) * 0.07);
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
      pnear[i] = rhoNear;
      pden[i] = rho;
      ppres[i] = this.gasK * (rho - this.restDensity);
      // Compression ratio: 0 at rest density, rises as particle is squeezed.
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
    // per-group mean temperature (for blobBuoyancy)
    const K = this.MAX_BLOBS;
    const gT = this._groupT || (this._groupT = new Float64Array(K));
    const gTn = this._groupTn || (this._groupTn = new Float64Array(K));
    const blobBuo = this.blobBuoyancy;
    if (blobBuo > 0) {
      gT.fill(0); gTn.fill(0);
      for (let i = nFixed; i < n; i++) { gT[gid[i]] += ptemp[i]; gTn[gid[i]]++; }
      for (let k = 0; k < K; k++) gT[k] = gTn[k] > 0 ? gT[k] / gTn[k] : 0;
    }
    const poolG = this._poolBlobId, wallG = nFixed > 0 ? gid[0] : -1;
    const cmyG = this.cmy, poolTopY = this.poolZoneTop;

    // Pool-spring cosine ramp: spring stiffness blends from 1.0×
    // above poolSpringLo to (1-atten)× below poolSpringHi.
    const POOL_LO = this.poolSpringLo, POOL_HI = this.poolSpringHi, POOL_INV = 1.0 / (POOL_HI - POOL_LO);
    const poolSpringAtten = this.poolSpringAtten;
    // exp(-(d_i+d_j)/(2τ)) = e_i·e_j with e = exp(-d/(2τ)): one exp per
    // particle instead of one per pair. Same for the pool cosine ramp.
    const dwellExp = this._dwellExp, poolRamp = this._poolRamp;
    const halfInvTau = 0.5 / poolDwellTau;
    for (let i = 0; i < n; i++) {
      dwellExp[i] = Math.exp(-zoneDwell[i] * halfInvTau);
      const sr = (py[i] - POOL_LO) * POOL_INV;
      const sc = sr < 0 ? 0 : sr > 1 ? 1 : sr;
      poolRamp[i] = 0.5 * (1 - Math.cos(Math.PI * sc));
    }
    for (let i = 0; i < n; i++) {
      let fpx = 0, fpy = 0;
      let fvx = 0, fvy = 0;
      let fcx = 0, fcy = 0;
      let frx = 0, fry = 0;
      let fnx = 0, fny = 0;
      const nearI = pnear[i];
      let dTsum = 0, neighCount = 0;
      const xi = px[i], yi = py[i];
      const poolI = poolRamp[i];
      const dwellI = dwellExp[i];
      const vxi = pvx[i], vyi = pvy[i];
      const Pi = ppres[i];
      const rhoi = pden[i];
      const Ti = ptemp[i];
      const gi = gid[i];

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
                  const dwellFactor = poolBarrierFloor + (1.0 - poolBarrierFloor) * dwellI * dwellExp[j];
                  let zFactor = 1.0;
                  if (i >= nFixed && j >= nFixed) {
                    const gj = gid[j];
                    const zDiff = bz[gi] - bz[gj];
                    const absZ = Math.abs(zDiff);
                    const zReach = Math.min(0.55, (sqrtSize[gi] + sqrtSize[gj]) * 0.07);
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
                    if (nearK > 0) {
                      const qn = 1 - r / h;
                      const nTerm = nearK * (nearI + pnear[j]) * qn * qn / r;
                      fnx += nTerm * dx;
                      fny += nTerm * dy;
                    }
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
                  // Smoothly attenuate spring force as BOTH particles enter the
                  // pool zone. effSpringK glides 1.0× → 0.5× as poolI·poolJ → 1.
                  const effSpringK = springK * (1 - poolSpringAtten * poolI * poolRamp[j]);
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
      const Tb = (blobBuo > 0 && gi !== poolG && gi !== wallG && cmyG[gi] < poolTopY) ? Ti + blobBuo * (gT[gi] - Ti) : Ti;
      const tNorm = Math.max(0, Math.min(1, (Tb - this.tAmbient) / (1.0 - this.tAmbient)));
      const riseFactor = (Math.exp(buoK * tNorm) - 1) / buoDenom;
      const densRatio = this.coolDensityExcess - buoTotal * riseFactor;
      const effG = this.gravity * densRatio * this.gravityScale;

      this.fx[i] = (fpx + fvx) / rhoi + fcx + frx + fnx;
      this.fy[i] = (fpy + fvy) / rhoi + fcy + effG + fry + fny;
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

    // Surface tension on each blob's elliptical (n = 2) capillary mode.
    if (this.capSigma > 0) this._applyCapillaryMode();

    // 3a) Integrate pool/wall particles via heavily-damped harmonic
    // oscillator anchored at each particle's home position. Forces from the
    // fluid are honored, but the spring + heavy velocity damping + hard
    // clamp keep displacement to a few pixels.
    {
      const wallK    = this.wallSpringK;
      const wallDamp = this.wallDampFactor;
      const wallMaxD = this.wallMaxDisplace;
      const wallMaxD2 = wallMaxD * wallMaxD;
      const homeX = this.homeX, homeY = this.homeY;
      const invMassW = 1.0 / this.mass;
      for (let i = 0; i < nFixed; i++) {
        const dxh = px[i] - homeX[i];
        const dyh = py[i] - homeY[i];
        const fxw = this.fx[i] - wallK * dxh;
        const fyw = this.fy[i] - wallK * dyh;
        pvx[i] = (pvx[i] + fxw * dt * invMassW) * wallDamp;
        pvy[i] = (pvy[i] + fyw * dt * invMassW) * wallDamp;
        px[i] += pvx[i] * dt;
        py[i] += pvy[i] * dt;
        // Hard clamp: never let a pool particle drift far from home.
        const ndx = px[i] - homeX[i];
        const ndy = py[i] - homeY[i];
        const nd2 = ndx * ndx + ndy * ndy;
        if (nd2 > wallMaxD2) {
          const s = wallMaxD / Math.sqrt(nd2);
          px[i] = homeX[i] + ndx * s;
          py[i] = homeY[i] + ndy * s;
          pvx[i] = 0;
          pvy[i] = 0;
        }
      }
    }

    // 3b) Integrate fluid particles
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
    const velDamp = this.velDamp;
    for (let i = nFixed; i < n; i++) {
      pvx[i] += this.fx[i] * dt * invMass;
      pvy[i] += this.fy[i] * dt * invMass;
      pvx[i] *= velDamp;
      pvy[i] *= velDamp;
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

  // ---- Capillary restoring force on the n = 2 shape mode ----------
  //
  // Particle-level SPH surface tension (e.g. Akinci et al. 2013) needs the
  // interface to span many kernel widths; our blobs are 5-40 particles,
  // under two kernels across, and its curvature estimate is noise that
  // shatters them. So surface tension is applied where the shape IS
  // resolved: each free blob's lowest deformation mode.
  //
  // A 2-D blob of area A = πR² deformed into an ellipse a = R·eˢ, b = R·e⁻ˢ
  // (area-preserving) has surface energy E(s) = σ·P(s), with Ramanujan's
  // perimeter P = πR·f(s),  f(s) = 6·cosh s − √(6·cosh 2s + 10).
  // The potential-flow mode shape of that deformation is the linear field
  // u = ṡ·(x', −y') in the principal frame, so the mode's inertia is
  // M_s = Σ m|d|² and Lagrange's equation gives
  //     s̈ = −σ π R f'(s) / (ρ Σ V|d|²),   ρ·V = m.
  // For small s, f'(s) → 3s and a uniform disc has Σ V|d|² = πR⁴/2, which
  // recovers Rayleigh's capillary frequency ω₂² = 6σ/(ρR³) exactly. Large
  // blobs are softer (∝ R⁻³), just like real drops.
  //
  // The force is applied as that modal acceleration on each particle,
  // a_i = s̈·((d·e₁)e₁ − (d·e₂)e₂). Because Σd = 0 and the field is
  // symmetric, it exerts no net force or torque. Damping is left to the
  // SPH viscosity — the ratio of the two (Ohnesorge number) decides
  // whether a disturbed blob wobbles or oozes back.
  _applyCapillaryMode() {
    const K = this.MAX_BLOBS;
    const n = this.n, nFixed = this.nFixed;
    const px = this.x, py = this.y, gid = this.groupId;
    const acc = this._capAcc;     // per group: N, Σx, Σy, Σxx, Σyy, Σxy, -, -
    acc.fill(0);
    const pvx = this.vx, pvy = this.vy;
    // velocity moments for shapeDamp: Σvx, Σvy, Σ vx·x, Σ vx·y, Σ vy·x, Σ vy·y
    const vm = this._capVel || (this._capVel = new Float64Array(K * 8));
    const shapeDamp = this.shapeDamp;
    if (shapeDamp > 0) vm.fill(0);
    for (let i = nFixed; i < n; i++) {
      const o = gid[i] << 3;
      const x = px[i], y = py[i];
      acc[o] += 1; acc[o + 1] += x; acc[o + 2] += y;
      acc[o + 3] += x * x; acc[o + 4] += y * y; acc[o + 5] += x * y;
      if (shapeDamp > 0) {
        const vx = pvx[i], vy = pvy[i];
        vm[o] += vx; vm[o + 1] += vy;
        vm[o + 2] += vx * x; vm[o + 3] += vx * y; vm[o + 4] += vy * x; vm[o + 5] += vy * y;
      }
    }
    // Walls belong to the pool body, which is not a free drop.
    const skipA = nFixed > 0 ? gid[0] : -1;
    const skipB = this._poolBlobId;
    const V0 = this.mass / this.restDensity;       // area per particle
    const sig = this.capSigma * this.surfaceTensionScale;
    // shapeDamp: fit v ≈ v̄ + G·d per free blob (G = C_vd · C_dd⁻¹ from the
    // moments), keep the traceless symmetric part S (stretch and shear
    // rates) and push back against it: a_i = −shapeDamp · S · d_i. No net
    // force (Σd = 0), no torque (S symmetric), no change of area (trace 0).
    const sd = this._capS || (this._capS = new Float64Array(K * 6));   // cx, cy, Sxx, Sxy, -, on
    if (shapeDamp > 0) {
      for (let g = 0; g < K; g++) {
        const o = g << 3, N = acc[o], q = g * 6;
        sd[q + 5] = 0;
        if (N < this.capMinN || g === skipA || g === skipB) continue;
        const cx = acc[o + 1] / N, cy = acc[o + 2] / N;
        if (cy > this.poolZoneTop) continue;        // plumes rising out of the pool: leave be
        const dxx = acc[o + 3] / N - cx * cx, dyy = acc[o + 4] / N - cy * cy, dxy = acc[o + 5] / N - cx * cy;
        const det = dxx * dyy - dxy * dxy;
        if (det < 1e-6) continue;
        const mvx = vm[o] / N, mvy = vm[o + 1] / N;
        // covariances of velocity with position
        const cvxx = vm[o + 2] / N - mvx * cx, cvxy = vm[o + 3] / N - mvx * cy;
        const cvyx = vm[o + 4] / N - mvy * cx, cvyy = vm[o + 5] / N - mvy * cy;
        // G = Cvd · Cdd⁻¹
        const ixx = dyy / det, iyy = dxx / det, ixy = -dxy / det;
        const gxx = cvxx * ixx + cvxy * ixy, gxy = cvxx * ixy + cvxy * iyy;
        const gyx = cvyx * ixx + cvyy * ixy, gyy = cvyx * ixy + cvyy * iyy;
        const tr = 0.5 * (gxx + gyy);
        sd[q] = cx; sd[q + 1] = cy;
        sd[q + 2] = gxx - tr; sd[q + 3] = 0.5 * (gxy + gyx);
        sd[q + 5] = 1;
      }
    }
    for (let g = 0; g < K; g++) {
      const o = g << 3;
      const N = acc[o];
      acc[o + 6] = 0;
      if (N < this.capMinN || g === skipA || g === skipB) continue;
      const cx = acc[o + 1] / N, cy = acc[o + 2] / N;
      const a = acc[o + 3] / N - cx * cx;
      const b = acc[o + 4] / N - cy * cy;
      const c = acc[o + 5] / N - cx * cy;
      const disc = Math.sqrt(0.25 * (a - b) * (a - b) + c * c);
      const l1 = 0.5 * (a + b) + disc;
      const l2 = Math.max(0.5 * (a + b) - disc, l1 * 1e-4);
      if (l1 <= 1e-6 || disc < 1e-9) continue;     // already round
      // semi-axis ratio a/b = √(λ1/λ2) = e^{2s}
      const sMode = 0.25 * Math.log(l1 / l2);
      const R = Math.sqrt(N * V0 / Math.PI);
      const ch2 = Math.cosh(2 * sMode);
      const fPrime = 6 * Math.sinh(sMode) - 6 * Math.sinh(2 * sMode) / Math.sqrt(6 * ch2 + 10);
      const sumD2 = N * (a + b);
      const sdd = -sig * Math.PI * R * fPrime / (V0 * sumD2);
      // principal axis e1
      let ex = l1 - b, ey = c;
      if (Math.abs(ex) + Math.abs(ey) < 1e-12) { ex = c; ey = l1 - a; }
      const inv = 1 / Math.hypot(ex, ey);
      acc[o + 1] = cx; acc[o + 2] = cy;
      acc[o + 3] = ex * inv; acc[o + 4] = ey * inv;
      acc[o + 6] = sdd * this.mass;               // force = m·a
    }
    const fx = this.fx, fy = this.fy;
    if (shapeDamp > 0) {
      const mD = shapeDamp * this.mass;
      for (let i = nFixed; i < n; i++) {
        const q = gid[i] * 6;
        if (sd[q + 5] === 0) continue;
        const dx = px[i] - sd[q], dy = py[i] - sd[q + 1];
        const sxx = sd[q + 2], sxy = sd[q + 3];
        fx[i] -= mD * (sxx * dx + sxy * dy);
        fy[i] -= mD * (sxy * dx - sxx * dy);
      }
    }
    for (let i = nFixed; i < n; i++) {
      const o = gid[i] << 3;
      const k = acc[o + 6];
      if (k === 0) continue;
      const dx = px[i] - acc[o + 1], dy = py[i] - acc[o + 2];
      const e1x = acc[o + 3], e1y = acc[o + 4];
      const p1 = dx * e1x + dy * e1y;             // along e1
      const p2 = -dx * e1y + dy * e1x;            // along e2 = (-e1y, e1x)
      fx[i] += k * (p1 * e1x + p2 * e1y);
      fy[i] += k * (p1 * e1y - p2 * e1x);
    }
  }

  // --- Mouse "grab" interaction ---------------------------------
  // Pull captured particles toward a *smoothed* target (which lags
  // behind the raw cursor for drag-feel). Each particle carries a
  // Gaussian weight `w` baked in at pickup time — far-edge particles
  // are gently nudged; central ones are tugged firmly. Called once
  // per sub-step after step(), so the impulse stacks with the regular
  // SPH integration and the result is clamped by the next step()'s
  // boundary pass.
  applyGrab(dt) {
    const g = this.grab;
    if (!g.active || g.particles.length === 0) return;
    // 1) Drive the smoothed target toward the raw cursor. The longer
    //    the time constant, the more lag the user feels.
    const sBlend = 1.0 - Math.exp(-g.targetSmoothK * dt);
    g.stx += (g.tx - g.stx) * sBlend;
    g.sty += (g.ty - g.sty) * sBlend;

    const stx = g.stx, sty = g.sty;
    const tvx = g.tvx, tvy = g.tvy;
    const springK = g.springK;
    const damp = g.damp;
    const vCouple = g.vCouple;
    const nFixed = this.nFixed;
    const px = this.x, py = this.y;
    const pvx = this.vx, pvy = this.vy;
    const list = g.particles;
    for (let p = 0; p < list.length; p++) {
      const rec = list[p];
      const i = rec.idx;
      if (i < nFixed || i >= this.n) continue;
      const w = rec.w;
      const dx = (stx + rec.offX) - px[i];
      const dy = (sty + rec.offY) - py[i];
      // Weighted spring pull toward the smoothed target.
      const k = w * springK;
      pvx[i] += k * dx * dt;
      pvy[i] += k * dy * dt;
      // Pure viscous damping toward zero — scaled by weight so far-edge
      // particles barely feel it. This is what makes the grab "draggy":
      // grabbed particles lose their own SPH momentum as they're held.
      const decay = Math.exp(-damp * dt * w);
      pvx[i] *= decay;
      pvy[i] *= decay;
      // Add a small fraction of the cursor's velocity back — so a quick
      // flick still imparts momentum, but the dominant feel is drag.
      const inject = (1.0 - decay) * vCouple;
      pvx[i] += tvx * inject;
      pvy[i] += tvy * inject;
    }
  }

  // Pick up fluid particles near (sx, sy). Each captured particle keeps
  // its offset from the click point (so the cluster keeps its shape
  // while being dragged) and a Gaussian weight that smoothly fades the
  // grab effect from centre to edge.
  beginGrab(sx, sy) {
    const g = this.grab;
    const r2 = g.radius * g.radius;
    const inv2s2 = 1.0 / (2.0 * g.sigma * g.sigma);
    const thresh = g.weightThresh;
    const list = [];
    const cap = g.maxParticles;
    for (let i = this.nFixed; i < this.n && list.length < cap; i++) {
      const dx = this.x[i] - sx;
      const dy = this.y[i] - sy;
      const d2 = dx * dx + dy * dy;
      if (d2 >= r2) continue;
      const w = Math.exp(-d2 * inv2s2);
      if (w < thresh) continue;
      list.push({ idx: i, offX: dx, offY: dy, w });
    }
    g.particles = list;
    g.tx = sx;  g.ty = sy;
    g.stx = sx; g.sty = sy;        // start smoothed target at the click point
    g.tvx = 0;  g.tvy = 0;
    g.active = list.length > 0;
    return g.active;
  }

  // Update the cursor target; newVx/newVy are the raw cursor velocity
  // in sim px/s (caller supplies — easy to compute from dt). We
  // additionally low-pass filter that here so jittery hand motion
  // doesn't translate into pixel-scale velocity spikes.
  updateGrab(sx, sy, newVx, newVy) {
    const g = this.grab;
    if (!g.active) return;
    g.tx = sx; g.ty = sy;
    const blend = 0.3;             // a touch heavier filter than before
    g.tvx = g.tvx * (1 - blend) + newVx * blend;
    g.tvy = g.tvy * (1 - blend) + newVy * blend;
  }

  endGrab() {
    this.grab.active = false;
    this.grab.particles.length = 0;
    this.grab.tvx = 0; this.grab.tvy = 0;
  }
}

export { SPH, SIM_W, SIM_H, bottleHalfFrac, bottleHalfWidth };
