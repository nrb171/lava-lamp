// ============================================================
//  GPU light: the ray tracer's targets and passes — wall photons, caustic
//  photons in the liquid, TAA accumulation and denoising (shaders/light/).
//  Methods of MetaballRenderer (installed by renderer.js).
// ============================================================

import { SIM_W, SIM_H, bottleHalfFrac } from "../sim/sim.js";
import { VIEW_M, VIEW_T, VIEW_W, VIEW_H } from "../view.js";
import { LIGHT_SRC_T, BULB_SIGMA, WALL_PHOTON_SIGMA, CAUS_PHOTON_SIGMA } from "./constants.js";
import { makeTriBlur } from "./util.js";

export class GpuLight {
  // Wall grid over the whole view: ~4 sim-px square cells (coarser only on
  // very large windows, capped at ~60k cells). Rebuilt when the view
  // changes size; the running-average baseline restarts then.
  ensureWallGrid() {
    // GPU: ~2 sim-px cells, up to ~300k of them. CPU: ~4 px, up to ~60k.
    const cell = this.wallGpu ? Math.max(2, Math.sqrt(VIEW_W * VIEW_H / 300000))
                              : Math.max(4, Math.sqrt(VIEW_W * VIEW_H / 60000));
    const W = Math.max(8, Math.round(VIEW_W / cell)), H = Math.max(8, Math.round(VIEW_H / cell));
    if (W === this.WG_W && H === this.WG_H && this._wgVW === VIEW_W && this._wgVH === VIEW_H) return;
    this._wgVW = VIEW_W; this._wgVH = VIEW_H;
    if (W !== this.WG_W || H !== this.WG_H) {
      this.WG_W = W; this.WG_H = H;
      const GW = W * H;
      this.wlL = new Float32Array(GW); this.wlW = new Float32Array(GW);
      this.bdL = new Float32Array(GW);     this.bdW = new Float32Array(GW);   // smoothed difference
      this.bdAvgL = new Float32Array(GW);  this.bdAvgW = new Float32Array(GW);
      this.bdData = new Float32Array(GW * 2);
      this.wgRowBlur = makeTriBlur(W, 4);
      this.wgColBlur = makeTriBlur(H, 4);
      this.wgTmp = new Float32Array(GW);
      const gl = this.gl;
      const zeros = new Float32Array(GW * 2);
      for (const tx of this.bdTex) {
        gl.bindTexture(gl.TEXTURE_2D, tx);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, W, H, 0, gl.RG, gl.FLOAT, zeros);
      }
      if (this.wallGpu && !this.allocWallGpu(W, H)) {
        // driver refused the targets: fall back to the CPU trace
        console.warn("GPU wall targets incomplete, using the CPU trace");
        this.wallGpu = false; this.WG_W = 0;
        return this.ensureWallGrid();
      }
    } else {
      this.bdL.fill(0); this.bdW.fill(0);
    }
    this.bdRef = null;
    this.bdAvgInit = false;
  }

  // GPU wall targets for a W × H grid: the splatted irradiance (2×), and per
  // side of the ping-pong the running-average baseline (32-bit: it moves
  // by tiny steps) with the drawn wall (bdTex, 16-bit) as a second output.
  allocWallGpu(W, H) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE7);
    const spec = (tx, ifmt, fmt, w, h, filter) => {
      gl.bindTexture(gl.TEXTURE_2D, tx);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, gl.FLOAT, null);
    };
    // the splat target is 2× the grid (less coverage noise from tiny
    // caustic triangles), filtered down in WALL_UPDATE_FS
    spec(this.irrTex, gl.RGBA16F, gl.RGBA, 2 * W, 2 * H, gl.LINEAR);
    for (const tx of this.avgTex) spec(tx, gl.RG32F, gl.RG, W, H, gl.NEAREST);
    const fbo = (texs) => {
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      texs.forEach((tx, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, tx, 0));
      gl.drawBuffers(texs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
      const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      return ok ? fb : null;
    };
    for (const fb of [this.irrFbo, ...(this.updFbo || [])]) if (fb) gl.deleteFramebuffer(fb);
    this.irrFbo = fbo([this.irrTex]);
    this.updFbo = [fbo([this.avgTex[0], this.bdTex[0]]), fbo([this.avgTex[1], this.bdTex[1]])];
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return !!(this.irrFbo && this.updFbo[0] && this.updFbo[1]);
  }

  // Ray target: one RGBA32F texel per ray (azimuth × polar·sources).
  ensureRays(NAZ, NPH, NSRC) {
    const key = NAZ + "x" + NPH + "x" + NSRC;
    if (key === this.rayKey) return true;
    const gl = this.gl;
    if (!this.rayTex) this.rayTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE7);        // scratch unit: don't displace bound inputs
    gl.bindTexture(gl.TEXTURE_2D, this.rayTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, NAZ, NPH * NSRC, 0, gl.RGBA, gl.FLOAT, null);
    if (this.rayFbo) gl.deleteFramebuffer(this.rayFbo);
    this.rayFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.rayFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.rayTex, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.rayKey = ok ? key : "";
    return ok;
  }

  // Float render target helper: (re)specifies tx and returns a framebuffer
  // with it attached (null if the driver can't render to it).
  gpuTarget(tx, ifmt, fmt, w, h, filter, oldFbo) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE7);        // scratch unit: don't displace bound inputs
    gl.bindTexture(gl.TEXTURE_2D, tx);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, gl.FLOAT, null);
    if (oldFbo) gl.deleteFramebuffer(oldFbo);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tx, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return ok ? fb : null;
  }

  // One GPU light trace: the wall if wallOn, the liquid if causOn. Results:
  // bdTex[bdCurTex] and causTex[causCur] (each blended with the previous
  // one while drawing). dtWall / dtCaus: seconds between wall / liquid
  // traces; dtPool: between calls.
  gpuLight(sim, px, py, glow, lowQ, wallOn, causOn, dtWall, dtCaus, dtPool) {
    const gl = this.gl;
    const poolN = this.buildPoolTex(sim, px, py, dtPool);   // leaves it on unit 5
    let nb = Math.min(40, this.buildWallBlobs(sim, px, py));
    nb = this.addPoolBumps(sim, px, py, nb, 40);
    const B = this._wBlobs, BS = this._wBlobS, w0 = this._wb0, w1 = this._wb1;
    for (let b = 0; b < nb; b++) {
      const q = b * 6, o = b * 4;
      w0[o] = B[q]; w0[o + 1] = B[q + 1]; w0[o + 2] = B[q + 2]; w0[o + 3] = BS[b];
      w1[o] = B[q + 3]; w1[o + 1] = B[q + 4]; w1[o + 2] = B[q + 5]; w1[o + 3] = 0;
    }
    if (sim.resetCount !== this._bdResetSeen) {
      this.bdAvgInit = false; this.causInit = false; this._bdResetSeen = sim.resetCount;
    }
    gl.useProgram(this.lightProg);
    const t = this.ltU;
    gl.uniform1i(t.uPool, 5);
    gl.uniform1i(t.uPoolN, poolN);
    gl.uniform2f(t.uSim, SIM_W, SIM_H);
    gl.uniform1f(t.uViewM, VIEW_M);
    gl.uniform1f(t.uViewT, VIEW_T);
    gl.uniform1f(t.uWallD, this.WALL_DIST);
    gl.uniform1f(t.uGlow, glow);
    // The light source sits on the axis just under the pool's resting
    // surface (where the lamp draws its solid pool). At the base, all its
    // light crossed the pool's surface, which moves as wax leaves and
    // returns — shifting the whole light field in jerks. Here mostly real
    // bumps and columns above that level bend it.
    gl.uniform1f(t.uYSrc, LIGHT_SRC_T * SIM_H);
    gl.uniform1f(t.uSrcSigma, BULB_SIGMA * SIM_W);
    gl.uniform1f(t.uMuPool, 0.004);
    gl.uniform1f(t.uPoolYMin, this.poolYMin);
    gl.uniform1i(t.uNB, nb);
    gl.uniform4fv(t.uWB0, w0);
    gl.uniform4fv(t.uWB1, w1);
    gl.bindVertexArray(this.vao);
    if (wallOn) {
      this.gpuWall(t, 1, dtWall, lowQ);
      gl.activeTexture(gl.TEXTURE5);                     // the wall's filter pass used unit 5
      gl.bindTexture(gl.TEXTURE_2D, this.poolTex);
    }
    if (causOn) this.gpuCaustics(t, 1, glow, dtCaus, lowQ);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // TAA. Each target (wall, liquid) has a ring of its last N traces and a
  // cycle of N jitter offsets for the ray grid (Halton 2,3); what is drawn
  // is the ring's mean (RING_AVG_FS). A still scene then gives the same
  // result every frame. N = 8 at full quality, 4 at the low tier.
  taaRing(name, W, H, N) {
    const gl = this.gl;
    let r = this.rings[name];
    if (!r || r.W !== W || r.H !== H || r.N !== N) {
      if (r) { gl.deleteTexture(r.tex); r.acc.forEach(t => gl.deleteTexture(t)); r.tmp.forEach(t => gl.deleteTexture(t)); }
      const tex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RG16F, W, H, N, 0, gl.RG, gl.FLOAT, null);
      // the accumulators (RING_AVG_FS): running mean and count, full floats —
      // half floats can't hold a long average (small steps round away)
      const acc = [0, 1].map(() => {
        const a = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, a);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, null);
        return a;
      });
      // the denoiser's ping-pong pair (DENOISE_FS)
      const tmp = [0, 1].map(() => {
        const a = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, a);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, W, H, 0, gl.RG, gl.FLOAT, null);
        return a;
      });
      r = this.rings[name] = { tex, acc, tmp, accCur: 0, W, H, N, slot: 0, filled: 0 };
    }
    if (!this.taa) r.filled = 0;                // restart when TAA is turned back on
    return r;
  }

  taaSeed(t, r) {
    const gl = this.gl;
    // Random ray origins and directions (the wide bulb, LIGHT_TRACE_FS) are
    // their own jitter: a fresh seed every trace, so samples keep adding up
    // (RING_AVG_FS) instead of cycling through the same few sets.
    this._seedN = (this._seedN || 0) + 1;
    gl.uniform1f(t.uSeed, this._seedN % 65536 + 1);
  }

  // after the update pass wrote this trace into layer r.slot: add it to the
  // accumulator (restarting it where the light moved), denoise → outTex
  taaResolve(r, outTex, hist) {
    const gl = this.gl;
    r.filled = Math.min(r.N, r.filled + 1);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.scratchFbo);
    const an = r.accCur ^ 1;
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, r.acc[an], 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.denoise ? r.tmp[0] : outTex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, r.W, r.H);
    gl.useProgram(this.ringAvgProg);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, r.tex);
    gl.uniform1i(this.raU.uRing, 3);
    // the ring fills slot by slot: average what is there so far
    gl.uniform1i(this.raU.uCount, r.filled);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, r.acc[r.accCur]);
    gl.uniform1i(this.raU.uAcc, 2);
    gl.uniform1i(this.raU.uSlot, r.slot);
    gl.uniform1f(this.raU.uHist, hist && r.filled >= r.N ? 1 : 0);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    r.slot = (r.slot + 1) % r.N;
    r.accCur = an;
    if (!this.denoise) return;
    // à-trous passes, steps 1, 2, 4 …: tmp0 → tmp1 → tmp0 … → outTex
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, null, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.useProgram(this.denoiseProg);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, r.acc[an]);
    gl.uniform1i(this.dnU.uAcc, 2);
    gl.uniform1i(this.dnU.uSrc, 3);
    gl.uniform1f(this.dnU.uK, this.denoiseK || 6);
    const passes = this.denoisePasses || 4;
    for (let k = 0; k < passes; k++) {
      const src = r.tmp[k & 1], dst = k === passes - 1 ? outTex : r.tmp[(k + 1) & 1];
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, dst, 0);
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, src);
      gl.uniform1i(this.dnU.uStep, 1 << k);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }

  gpuWall(t, NSRC, dtTrace, lowQ) {
    const gl = this.gl;
    const rs = this.rayScale || 1;
    const NPH = Math.max(8, Math.round((lowQ ? 128 : 384) * rs)), NAZ = Math.max(16, Math.round((lowQ ? 256 : 768) * rs));
    this.wallRays = NAZ * NPH * NSRC;
    if (!this.ensureRays(NAZ, NPH, NSRC)) { this.wallGpu = false; this.WG_W = 0; this.ensureWallGrid(); return; }
    // 1. rays → wall landing points
    gl.useProgram(this.lightProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.rayFbo);
    gl.viewport(0, 0, NAZ, NPH * NSRC);
    gl.uniform1i(t.uMode, 0);
    gl.uniform1i(t.uRowOff, 0);
    gl.uniform1f(t.uLensOn, 1);
    gl.uniform1f(t.uAzSpan, Math.PI);                  // back half: toward the wall
    // Power per ray. The splat target is half-float: at ~3e-5 per texel
    // (the natural scale here) values are subnormal and round coarsely,
    // which summed into contour rings — so work around 1 instead. (The
    // normalisation, bdRef, divides the scale back out.)
    gl.uniform1f(t.uRayScale, 1e7 / (NPH * NAZ));
    gl.uniform3i(t.uRayDim, NAZ, NPH, NSRC);
    // (the same ring in both quality tiers: the adaptive quality switches
    // tier whenever frame time crosses its threshold, and rebuilding the
    // ring threw the accumulated light away — the wall flashed back to
    // noise at every switch)
    const ring = this.taaRing("wall", this.WG_W, this.WG_H, 8);
    gl.useProgram(this.lightProg);
    this.taaSeed(t, ring);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // 2. photons → wall (2× grid), additively
    const GWc = this.WG_W, GHc = this.WG_H;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.irrFbo);
    gl.viewport(0, 0, 2 * GWc, 2 * GHc);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.wallPhotonProg);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, this.rayTex);
    const pU = this.wpU;
    gl.uniform1i(pU.uRays, 6);
    gl.uniform2i(pU.uRaySize, NAZ, NPH * NSRC);
    gl.uniform2f(pU.uView, VIEW_W, VIEW_H);
    // kernel: a fixed width on the wall (sim px), in 2× grid texels — the
    // same in both tiers, and small enough for the GPU's largest point
    // sprite (some mobile GPUs stop at 64 px, cutting the splat's edges)
    this._maxPoint = this._maxPoint || gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
    gl.uniform1f(pU.uSigma, Math.min(20, (this._maxPoint - 1) / 6.5, WALL_PHOTON_SIGMA * 2 * GWc / VIEW_W));
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.POINTS, 0, NAZ * NPH * NSRC);
    gl.disable(gl.BLEND);

    // Normalisation, once per grid: the mean wall light beside the lamp
    // (one small read-back; the CPU path takes the same mean). Per grid
    // cell = sum of its 2×2 splat texels.
    if (!this.bdRef) {
      const W2 = 2 * GWc, H2 = 2 * GHc;
      const buf = new Float32Array(W2 * H2 * 4);
      gl.readPixels(0, 0, W2, H2, gl.RGBA, gl.FLOAT, buf);
      let sum = 0, cnt = 0;
      const cw = VIEW_W / W2, ch = VIEW_H / H2;
      for (let rr = 0; rr < H2; rr++) {
        const ys = (H2 - 1 - rr + 0.5) * ch - VIEW_T;       // framebuffer rows run bottom-up
        if (ys < 0.1 * SIM_H || ys > 0.9 * SIM_H) continue;
        const hwc = bottleHalfFrac(ys / SIM_H) * SIM_W;
        for (let c = 0; c < W2; c++) {
          if (Math.abs((c + 0.5) * cw - VIEW_M - SIM_W / 2) > hwc) {
            const k = (rr * W2 + c) * 4; sum += buf[k] + buf[k + 1]; cnt++;
          }
        }
      }
      // …measured on the actual wall, then scaled to the reference
      // distance, so the 1/r² falloff to the real wall shows: the
      // normalisation alone would otherwise cancel it
      this.bdRef = (cnt && sum > 0 ? 4 * sum / cnt : 1) * (this.WALL_DIST / this.WALL_DIST_REF) ** 2;
    }

    // 3. filter down, baseline, drawn difference (see the CPU path)
    if (!this.bdAvgInit) this._bdAge = 0;
    this._bdAge += dtTrace;
    const tau = this._bdAge < 20 ? 2 : this.WALL_AVG_TAU;
    const cur = this.bdCurTex, nxt = cur ^ 1;
    if (this.taa) {
      // this trace's difference → the ring's current layer (resolved below)
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.scratchFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.avgTex[nxt], 0);
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, ring.tex, 0, ring.slot);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    } else gl.bindFramebuffer(gl.FRAMEBUFFER, this.updFbo[nxt]);
    gl.viewport(0, 0, GWc, GHc);
    gl.useProgram(this.wallUpdProg);
    const uU = this.wuU;
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, this.irrTex);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, this.avgTex[cur]);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, this.bdTex[cur]);
    gl.uniform1i(uU.uIrr, 5); gl.uniform1i(uU.uAvg, 6); gl.uniform1i(uU.uPrev, 7);
    gl.uniform1f(uU.uInvRef, 1 / this.bdRef);
    gl.uniform1f(uU.uAlpha, Math.min(1, dtTrace / tau));
    // temporal smoothing: without TAA, a half-life of 2 frames; with TAA
    // the raw trace goes to the ring, whose mean is drawn
    gl.uniform1f(uU.uSM, this.taa ? 1 : 1 - Math.pow(0.707, dtTrace * 60));
    gl.uniform1f(uU.uInit, this.bdAvgInit ? 0 : 1);
    gl.uniform1f(uU.uFull, this.wallFull ? 1 : 0);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    if (this.taa) this.taaResolve(ring, this.bdTex[nxt], this.bdAvgInit);
    this.bdAvgInit = true;
    this.bdCurTex = nxt;
    this.wallFramesSince = 0;
  }

  // Caustics in the liquid: path vertices → photon bands → filtered field
  // over the lamp.
  gpuCaustics(t, NSRC, glow, dtTrace, lowQ) {
    const gl = this.gl;
    const rs = this.rayScale || 1;
    const NAZ = Math.max(16, Math.round((lowQ ? 128 : 256) * rs)), NPH = Math.max(6, Math.round((lowQ ? 48 : 96) * rs));
    this.causRays = NAZ * NPH * NSRC;
    // one grid for both quality tiers: a new grid restarts the accumulated
    // light and re-measures the normalisation (a flash at every switch)
    const cell = 2;
    const CW = Math.round(SIM_W / cell), CH = Math.round(SIM_H / cell);
    const rows = NPH * NSRC;
    // targets
    const pk = NAZ + "x" + rows;
    if (pk !== this.pathKey) {
      // path vertices: four targets written at once, two columns per ray
      this.pathTex.forEach((tx, i) => { const fb = this.gpuTarget(tx, gl.RGBA32F, gl.RGBA, NAZ * 2, 2 * rows, gl.NEAREST, null); if (fb) gl.deleteFramebuffer(fb); });
      if (this.pathFbo) gl.deleteFramebuffer(this.pathFbo);
      this.pathFbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.pathFbo);
      this.pathTex.forEach((tx, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, tx, 0));
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3]);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { gl.deleteFramebuffer(this.pathFbo); this.pathFbo = null; }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.pathKey = pk; this.refReady = false;
    }
    const ck = CW + "x" + CH;
    if (ck !== this.causKey) {
      this.causAccFbo = this.gpuTarget(this.causAcc, gl.RGBA16F, gl.RGBA, CW, CH, gl.NEAREST, this.causAccFbo);
      this.causFbo = (this.causFbo || [null, null]).map((fb, i) =>
        this.gpuTarget(this.causTex[i], gl.RG16F, gl.RG, CW, CH, gl.LINEAR, fb));
      this.refFbo = this.refFbo.map((fb, i) => this.gpuTarget(this.refTex[i], gl.RGBA16F, gl.RGBA, CW, CH, gl.NEAREST, fb));
      this.causEdgeFbo = this.gpuTarget(this.causEdgeTex, gl.RG16F, gl.RG, 2, CH, gl.LINEAR, this.causEdgeFbo);
      this.causKey = ck; this.causRef = null; this.causInit = false; this.refReady = false;
    }
    if (!this.pathFbo || !this.causAccFbo || !this.causFbo[0] || !this.causFbo[1]
        || !this.refFbo[0] || !this.refFbo[1] || !this.causEdgeFbo) {
      console.warn("GPU caustic targets incomplete, using the CPU");
      this.wallGpu = false; this.WG_W = 0; this.ensureWallGrid(); return;
    }
    // 1. path vertices: traced rays (rows 0..rows-1), then the reference
    //    without wax lenses (rows rows..2·rows-1)
    gl.useProgram(this.lightProg);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.pathFbo);
    gl.uniform1i(t.uMode, 1);
    gl.uniform1f(t.uAzSpan, 2 * Math.PI);
    const ring = this.taaRing("caus", CW, CH, 8);
    gl.useProgram(this.lightProg);
    this.taaSeed(t, ring);
    gl.uniform1f(t.uRayScale, 1e3 / (NPH * NAZ));             // ~1 per texel: see gpuWall
    gl.uniform3i(t.uRayDim, NAZ, NPH, NSRC);
    gl.bindVertexArray(this.vao);
    for (let pass = 0; pass < 2; pass++) {
      gl.viewport(0, pass * rows, NAZ * 2, rows);
      gl.uniform1i(t.uRowOff, pass * rows);
      gl.uniform1f(t.uLensOn, pass === 0 ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    // 2. photons → light, additively (path vertices on units 4-7)
    gl.useProgram(this.causPhotonProg);
    const pU = this.cpU;
    this.pathTex.forEach((tx, i) => { gl.activeTexture(gl.TEXTURE4 + i); gl.bindTexture(gl.TEXTURE_2D, tx); });
    gl.uniform1i(pU.uV0, 4); gl.uniform1i(pU.uV1, 5); gl.uniform1i(pU.uV2, 6); gl.uniform1i(pU.uV3, 7);
    gl.uniform1i(pU.uNAz, NAZ);
    gl.uniform1i(pU.uRowsLens, rows);
    gl.uniform2f(pU.uSim, SIM_W, SIM_H);
    gl.uniform2f(pU.uTgt, CW, CH);
    gl.uniform1f(pU.uSig, Math.max(0.7, CAUS_PHOTON_SIGMA / cell));
    gl.viewport(0, 0, CW, CH);
    gl.clearColor(0, 0, 0, 0);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.bindVertexArray(this.emptyVao);
    // 2a. the rays blobs (or pool bumps) touched, traced and reference
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.causAccFbo);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform1i(pU.uRefSrc, -1);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, 6 * NAZ * 2 * rows);
    // 2b. the full reference (changes slowly: every 4th update)
    if (!this.refReady || (this._refTick = ((this._refTick || 0) + 1) % 4) === 0) {
      const build = this.refReady ? this.refCur ^ 1 : 0;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.refFbo[build]);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1i(pU.uRefSrc, 0);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, 6 * NAZ * rows);
      this.refCur = build; this.refReady = true;
    }
    gl.disable(gl.BLEND);

    // Normalisation, once per grid, from the full reference, as on the
    // CPU: caustic excess is a fraction of the brightest clear-lamp light
    // (at glow 1) — over the liquid above the pool, as a high percentile
    // (robust to a hot texel); glass-edge light is relative to its mean
    // along the lit glass.
    if (!this.causRef) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.refFbo[this.refCur]);
      const buf = new Float32Array(CW * CH * 4);
      gl.readPixels(0, 0, CW, CH, gl.RGBA, gl.FLOAT, buf);
      const vals = [];
      let eSum = 0, eN = 0;
      for (let r = 0; r < CH; r++) {
        const y = (CH - 1 - r + 0.5) * SIM_H / CH;
        if (y < 0.1 * SIM_H || y > 0.7 * SIM_H) continue;
        const hw = bottleHalfFrac(y / SIM_H) * SIM_W;
        for (let c = 0; c < CW; c++) {
          const dx = Math.abs((c + 0.5) * SIM_W / CW - SIM_W / 2), g = buf[(r * CW + c) * 4 + 1];
          if (dx < hw - 4) vals.push(g);
          if (dx < hw - 8 && dx > hw - 20) { eSum += g; eN++; }
        }
      }
      vals.sort((a, b) => a - b);
      // (95th percentile: a point bulb under the round pool focuses a thin
      // bright line up the axis, which shouldn't set the scale)
      const f = vals.length ? vals[Math.floor(vals.length * 0.95)] : 1;
      const e = eN ? eSum / eN : 1;
      this.causRef = { f: (f || 1) / glow, e: (e || 1) / glow };
    }

    // 3. filter, normalise, smooth in time
    const cur = this.causCur, nxt = cur ^ 1;
    if (this.taa) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.scratchFbo);
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, ring.tex, 0, ring.slot);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, null, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    } else gl.bindFramebuffer(gl.FRAMEBUFFER, this.causFbo[nxt]);
    gl.viewport(0, 0, CW, CH);
    gl.useProgram(this.causUpdProg);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, this.causAcc);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, this.refTex[this.refCur]);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, this.causTex[cur]);
    const cU = this.cuU;
    gl.uniform1i(cU.uAcc, 5); gl.uniform1i(cU.uRef, 6); gl.uniform1i(cU.uPrev, 7);
    gl.uniform1f(cU.uInvRef, 1 / this.causRef.f);
    gl.uniform1f(cU.uSM, this.taa ? 1 : 1 - Math.pow(0.548, dtTrace * 60));   // see gpuWall
    gl.uniform1f(cU.uInit, this.causInit ? 0 : 1);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    if (this.taa) this.taaResolve(ring, this.causTex[nxt], this.causInit);
    this.causInit = true;
    this.causCur = nxt;
    this.causFramesSince = 0;
    // 4. light at the glass, per row and side
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.causEdgeFbo);
    gl.viewport(0, 0, 2, CH);
    gl.useProgram(this.causEdgeProg);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, this.causTex[nxt]);
    gl.uniform1i(this.ceU.uCaus, 7);
    gl.uniform2f(this.ceU.uSim, SIM_W, SIM_H);
    gl.uniform1f(this.ceU.uEdgeScale, this.causRef.f / this.causRef.e);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}
