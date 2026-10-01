// ============================================================
//  Geometry for the light tracers: the blobs as ellipsoids, the pool's
//  base surface as a height field and its bumps as ellipsoids.
//  Methods of MetaballRenderer (installed by renderer.js).
// ============================================================

import { SIM_W, SIM_H, bottleHalfFrac } from "../sim/sim.js";

export class LightGeometry {
  // Blobs as the wall light sees them: into this._wBlobs (per blob: centre
  // x from the axis, y, z; semi-axes) and this._wBlobS (lens strength).
  // Returns the blob count. Shared by the GPU and CPU wall traces.
  buildWallBlobs(sim, px, py) {
    const cx = SIM_W * 0.5;
    // Wax ellipsoids from the particles (axis-aligned).
    // Slots 0..K-1 are sim blobs; K..K+3 are pool-merge ghosts (particles
    // already in the pool group but still fading in on screen).
    // Each blob also gets a lens strength s ∈ [0, 1]: the mean of its
    // particles' fade-in since leaving the pool (or 1 − the ghost's merge
    // mix). s scales wax's refractive contrast and absorption, so a blob
    // breaking from the pool — or sinking back into it — changes the light
    // on the wall smoothly instead of switching on or off.
    const K = sim.MAX_BLOBS, KG = K + 4;
    const acc = this._wAcc || (this._wAcc = new Float64Array(KG * 6));
    acc.fill(0);
    const ghostOf = this.ghostOf, freeAge = this.freeAge;
    for (let i = 0; i < sim.n; i++) {
      const gh = i >= sim.nFixed ? ghostOf[i] : -1;
      const g = gh >= 0 ? K + gh : sim.groupId[i], o = g * 6, x = px[i], y = py[i];
      const a = Math.min(1, freeAge[i] / 0.8), fade = a * a * (3 - 2 * a);
      const str = gh >= 0 ? 1 - this.ghostMix[gh] : fade;
      acc[o] += 1; acc[o + 1] += x; acc[o + 2] += y; acc[o + 3] += x * x; acc[o + 4] += y * y; acc[o + 5] += str;
    }
    const V0 = sim.mass / sim.restDensity, pr2 = V0 / Math.PI;
    const gPool = sim.nFixed > 0 ? sim.groupId[0] : -1;
    const B = this._wBlobs || (this._wBlobs = new Float32Array(KG * 6));
    const BS = this._wBlobS || (this._wBlobS = new Float32Array(KG));
    let nb = 0;
    for (let g = 0; g < KG; g++) {
      const o = g * 6, N = acc[o];
      if (N < 3 || g === gPool || g === sim._poolBlobId) continue;   // the pool is the source
      const str = acc[o + 5] / N;
      if (str < 0.01) continue;
      const mx = acc[o + 1] / N, my = acc[o + 2] / N;
      const vx = Math.max(0, acc[o + 3] / N - mx * mx), vy = Math.max(0, acc[o + 4] / N - my * my);
      // uniform disc: variance = r²/4  →  r = 2σ (plus a particle radius)
      const ax = Math.sqrt(4 * vx + pr2), ay = Math.sqrt(4 * vy + pr2);
      const Rl = bottleHalfFrac(Math.min(1, Math.max(0, my / SIM_H))) * SIM_W;
      let az = Math.min(Math.sqrt(ax * ay), 0.9 * Rl);
      const zN = g < K ? sim.blobZ[g] - 0.5 : (this.blobZ36[g - K + 32] - 0.5);
      let cz = zN * 2 * 0.8 * Rl;
      const lim = Math.max(0, Rl - az);
      if (cz > lim) cz = lim; else if (cz < -lim) cz = -lim;
      const q = nb * 6;
      B[q] = mx - cx; B[q + 1] = my; B[q + 2] = cz; B[q + 3] = ax; B[q + 4] = ay; B[q + 5] = az;
      BS[nb] = str;
      nb++;
    }
    return nb;
  }

  // The pool, for LIGHT_TRACE_FS: a smooth base surface (a height field)
  // plus its bumps and columns as wax ellipsoids (addPoolBumps), like the
  // free blobs. (Bumps in the height field made a rising column a steep
  // mound that bent light to extremes, then switched to a blob's ellipsoid
  // when it pinched off: a pop.)
  //
  // Each particle adds its area (2-D sim) to the wax thickness over its x,
  // weighted by how much it belongs to the pool: 1 in the pool; a blob
  // breaking free fades out over the same 0.8 s its own lens fades in; a
  // merging blob (ghost) fades in as its lens fades out. Pass 1 finds the
  // base; particles more than ~10 px above it are bumps; pass 2 rebuilds
  // the base without them. The base is made round about the lamp's axis;
  // row 0 of the texture is its top y vs radius.
  buildPoolTex(sim, px, py, dt) {
    const gl = this.gl, N = 128, half = SIM_W / 2, dxs = SIM_W / (N - 1);
    if (!this._poolFloor) {
      // glass bottom (2-D) under each x
      this._poolFloor = new Float32Array(N);
      // (solved exactly: a floor found in fixed steps is a staircase, and
      // revolved, each step is a ring of sharp curvature that a point
      // bulb draws on the wall as a line)
      for (let j = 0; j < N; j++) {
        const ax = Math.abs(j * dxs - half);
        let lo = 0.8 * SIM_H, hi = 0.99 * SIM_H;      // radius ≥ ax at lo
        if (bottleHalfFrac(hi / SIM_H) * SIM_W >= ax) { this._poolFloor[j] = hi; continue; }
        for (let k = 0; k < 30; k++) {
          const m = 0.5 * (lo + hi);
          if (bottleHalfFrac(m / SIM_H) * SIM_W >= ax) lo = m; else hi = m;
        }
        this._poolFloor[j] = lo;
      }
      this._topS = new Float64Array(N);
      this._h2 = new Float64Array(N); this._hb = new Float64Array(N);
      this._hbSlow = new Float64Array(N); this._hbInit = false;
      this._poolData = new Float32Array(2 * N);
    }
    if (!this._pw || this._pw.length < sim.n) {
      this._pw = new Float32Array(sim.cap); this._bm = new Float32Array(sim.cap); this._bump = new Int32Array(sim.cap);
    }
    const h2 = this._h2, hb = this._hb, floor = this._poolFloor, pw = this._pw;
    const a0 = sim.mass / sim.restDensity, sig = 0.5 * sim.h;
    const inv = 1 / (2 * sig * sig), norm = a0 / (Math.sqrt(2 * Math.PI) * sig);
    const gPool = sim.nFixed > 0 ? sim.groupId[0] : -1;
    const ghostOf = this.ghostOf, freeAge = this.freeAge;
    for (let i = 0; i < sim.n; i++) {
      const gh = i >= sim.nFixed ? ghostOf[i] : -1;
      let w;
      if (gh >= 0) w = this.ghostMix[gh];
      else if (sim.groupId[i] === gPool) w = 1;
      else { const a = Math.min(1, freeAge[i] / 0.8); w = 1 - a * a * (3 - 2 * a); }
      pw[i] = w > 1e-3 ? w : 0;
    }
    const bm = this._bm;
    const thickness = (skipBumps) => {
      h2.fill(0);
      for (let i = 0; i < sim.n; i++) {
        const w = skipBumps ? pw[i] * (1 - bm[i]) : pw[i];
        if (w === 0) continue;
        const xi = px[i];
        const j0 = Math.max(0, Math.floor((xi - 3 * sig) / dxs)), j1 = Math.min(N - 1, Math.ceil((xi + 3 * sig) / dxs));
        for (let j = j0; j <= j1; j++) { const d = j * dxs - xi; h2[j] += w * norm * Math.exp(-d * d * inv); }
      }
      // broad: Gaussian, σ = 40 px
      const sj = 40 / dxs, R = Math.ceil(3 * sj);
      for (let j = 0; j < N; j++) {
        let s = 0, ws = 0;
        for (let k = Math.max(0, j - R); k <= Math.min(N - 1, j + R); k++) {
          const w = Math.exp(-0.5 * ((k - j) / sj) ** 2); s += w * h2[k]; ws += w;
        }
        hb[j] = s / ws;
      }
    };
    // pass 1: base with everything. A particle belongs to a bump by how
    // far it stands above the base — gradually, from 0.2 h to 0.6 h, so
    // nothing switches between the two models in one frame.
    bm.fill(0);
    thickness(false);
    let nBump = 0;
    const bump = this._bump, h = sim.h;
    for (let i = sim.nFixed; i < sim.n; i++) {
      if (pw[i] === 0) continue;
      const u = Math.min(N - 1, Math.max(0, px[i] / dxs)), j = Math.min(N - 2, u | 0), f = u - j;
      const base = (floor[j] - hb[j]) * (1 - f) + (floor[j + 1] - hb[j + 1]) * f;
      const e = Math.min(1, Math.max(0, (base - py[i] - 0.2 * h) / (0.4 * h)));
      bm[i] = e * e * (3 - 2 * e);
      if (bm[i] > 0) bump[nBump++] = i;
    }
    this._nBump = nBump;
    // pass 2: the base without them (by their bump share)
    thickness(true);
    // The base level follows a running average (the wall baseline's time
    // constant): the pool slowly filling or draining moves the whole light
    // cone, which the wall would otherwise show as broad glows.
    const hs = this._hbSlow;
    if (!this._hbInit || sim.resetCount !== this._hbReset) {
      hs.set(hb); this._hbInit = true; this._hbReset = sim.resetCount;
    } else {
      const k = Math.min(1, dt / (this._bdAge < 20 ? 2 : this.WALL_AVG_TAU));
      for (let j = 0; j < N; j++) hs[j] += (hb[j] - hs[j]) * k;
    }
    // top, smoothed again (σ = 20 px) and evaluated directly at each x
    // (interpolating between samples would leave kinks): the glass floor's
    // own curvature would otherwise carry into the surface
    const topS = this._topS;
    for (let j = 0; j < N; j++) topS[j] = floor[j] - hs[j];
    const s2 = 20 / dxs, R2 = Math.ceil(4.5 * s2);   // (wide window: no step as it slides)
    const D = this._poolData;
    const topAt = this._poolTopAt = (x) => {
      const u = x / dxs, c = Math.round(u);
      let sum = 0, ws = 0;
      for (let k = Math.max(0, c - R2); k <= Math.min(N - 1, c + R2); k++) {
        const w = Math.exp(-0.5 * ((k - u) / s2) ** 2); sum += w * topS[k]; ws += w;
      }
      return sum / ws;
    };
    let yMin = 1e9;
    for (let j = 0; j < N; j++) {
      const rho = half * j / (N - 1);
      D[j] = 0.5 * (topAt(half + rho) + topAt(half - rho));
      D[N + j] = 0;
      yMin = Math.min(yMin, D[j]);
    }
    // highest the surface gets (smallest y), so the tracer only looks the
    // pool up near it
    this.poolYMin = yMin - 2;
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.poolTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, N, 2, 0, gl.RED, gl.FLOAT, D);
    return N;
  }

  // The pool's bumps and columns (buildPoolTex) as ellipsoids, appended to
  // the blob list: bump particles grouped along x (a gap of more than h
  // splits them), each group's weighted moments giving an ellipsoid that
  // reaches down into the base. Particles count by their pool weight ×
  // bump share. Lens strength = how much its particles belong to the pool,
  // so as a blob pinches off, its share of the column's lens fades out
  // while its own fades in. Returns the new blob count.
  addPoolBumps(sim, px, py, nb, maxB) {
    const n = this._nBump, idx = this._bump, pw = this._pw, bm = this._bm;
    if (!n) return nb;
    const order = Array.from(idx.subarray(0, n)).sort((a, b) => px[a] - px[b]);
    const B = this._wBlobs, BS = this._wBlobS, cx = SIM_W * 0.5;
    const pr2 = sim.mass / sim.restDensity / Math.PI, gap = sim.h;
    let start = 0;
    for (let e = 1; e <= n && nb < maxB; e++) {
      if (e < n && px[order[e]] - px[order[e - 1]] <= gap) continue;
      // cluster order[start .. e-1]
      let sw = 0, sm = 0, sx = 0, sy = 0, sxx = 0, syy = 0;
      for (let k = start; k < e; k++) {
        const i = order[k], w = pw[i] * bm[i];
        sw += w; sm += bm[i]; sx += w * px[i]; sy += w * py[i]; sxx += w * px[i] * px[i]; syy += w * py[i] * py[i];
      }
      start = e;
      if (sw < 0.5) continue;
      const mx = sx / sw, my = sy / sw;
      const vx = Math.max(0, sxx / sw - mx * mx), vy = Math.max(0, syy / sw - my * my);
      const ax = Math.sqrt(4 * vx + pr2);
      const top = my - Math.sqrt(4 * vy + pr2);
      const bottom = this._poolTopAt(mx) + 0.25 * sim.h;          // into the base
      if (bottom - top < 4) continue;
      const cy = 0.5 * (top + bottom), ay = 0.5 * (bottom - top);
      const Rl = bottleHalfFrac(Math.min(1, Math.max(0, cy / SIM_H))) * SIM_W;
      const az = Math.min(ax, 0.9 * Rl);
      const q = nb * 6;
      B[q] = mx - cx; B[q + 1] = cy; B[q + 2] = 0; B[q + 3] = ax; B[q + 4] = ay; B[q + 5] = az;
      // strength: how much its particles belong to the pool, faded in with
      // its size (1 → 3 particles' worth) instead of appearing at a count
      const g = Math.min(1, Math.max(0, (sw - 1) / 2));
      BS[nb] = (sw / sm) * g * g * (3 - 2 * g);
      if (BS[nb] < 0.01) continue;
      nb++;
    }
    return nb;
  }
}
