// ============================================================
//  CPU light: the fallback caustic and wall tracers, used when the GPU
//  path is unavailable (no float render targets).
//  Methods of MetaballRenderer (installed by renderer.js).
// ============================================================

import { SIM_W, SIM_H, bottleHalfFrac } from "../sim/sim.js";
import { VIEW_M, VIEW_T, VIEW_W, VIEW_H } from "../view.js";
import { fresnelR } from "./util.js";

export class CpuLight {
  // ---- Caustics: 2-D ray tracing of pool light, every frame ----
  // Rays leave the lit pool surface (Lambertian, brighter at the centre)
  // and march upward row by row through the wax grid:
  //  • Wax lensing — paraffin (n 1.43) in the liquid (n 1.34) is a
  //    converging medium. Crossing a row of height Δy whose wax volume
  //    fraction φ varies in x, a ray is bent like a thin prism:
  //        Δ(dx/dy) = (n_wax/n_liquid − 1) · Δy · ∂φ/∂x
  //    toward the thicker wax. Wax also absorbs, at the volumetric rate.
  //  • Walls — at the glass the ray reflects with the Fresnel reflectance
  //    for liquid → air (the thin glass shell is ~parallel, so the exit
  //    into air decides it). Past ~48° incidence that is total internal
  //    reflection, which is why grazing light hugs and piles up along the
  //    walls. The light striking the glass at each height is recorded
  //    (wall irradiance → the glowing glass edge).
  // Each ray also has an unbent, non-reflecting twin; the fluence
  // difference (traced − straight) is the caustic: where lensing and wall
  // reflections concentrate light (+) or pull it away (−). Plain shadowing
  // is already the volumetric R channel, so this only adds the redirection.
  traceCaustics(sim, massRaw, glow, noWax, px, py) {
    const NC = this.NUM_COLS, NR = this.NUM_ROWS;
    const colW = SIM_W / NC, rowH = SIM_H / NR;
    const cx = SIM_W * 0.5;
    const hw = (y) => bottleHalfFrac(y / SIM_H) * SIM_W - 2;   // inner wall
    const phi = this.waxFrac, fT = this.fTrace, fS = this.fStraight;
    const wallE = this.wallE;
    const V0 = sim.mass / sim.restDensity;                   // area per particle
    // massRaw spreads each particle over 2 units (1 + 4 × 0.25)
    const fracPerUnit = 0.5 * V0 / (colW * rowH);
    // Wax volume fraction from the particles. Each particle stands for
    // V0 ≈ 244 px² of wax, far more than one 3.8 × 11.7 px cell, so it is
    // spread with a smooth kernel about its own size (radius KR), normalised
    // to conserve wax. The field — and every lens gradient read from it —
    // then changes smoothly as particles move, instead of flickering as they
    // cross cell boundaries.
    const phiRaw = this._phiRaw || (this._phiRaw = new Float32Array(NC * NR));
    phiRaw.fill(0);
    if (!noWax) {
      const KR = 14, KR2 = KR * KR, cellA = colW * rowH;
      const wbuf = this._kw || (this._kw = new Float32Array(64));
      const ibuf = this._ki || (this._ki = new Int32Array(64));
      for (let i = sim.nFixed; i < sim.n; i++) {
        const X = px[i], Y = py[i];
        const c0 = Math.max(0, Math.floor((X - KR) / colW)), c1 = Math.min(NC - 1, Math.floor((X + KR) / colW));
        const r0 = Math.max(0, Math.floor((Y - KR) / rowH)), r1 = Math.min(NR - 1, Math.floor((Y + KR) / rowH));
        let m = 0, wsum = 0;
        for (let rr = r0; rr <= r1; rr++) {
          const dy = (rr + 0.5) * rowH - Y;
          for (let cc = c0; cc <= c1 && m < 64; cc++) {
            const dx = (cc + 0.5) * colW - X;
            const q = 1 - (dx * dx + dy * dy) / KR2;
            if (q <= 0) continue;
            const w = q * q;
            wbuf[m] = w; ibuf[m] = rr * NC + cc; m++; wsum += w;
          }
        }
        if (wsum <= 0) continue;
        const scale = V0 / (cellA * wsum);
        for (let k = 0; k < m; k++) phiRaw[ibuf[k]] += wbuf[k] * scale;
      }
    }
    // Coverage, not a hard clamp: for randomly overlapping bits of wax the
    // chance a line of sight is blocked is 1 − e^(−φraw). A hard min(1, ·)
    // has a kink whose position jitters with the particles, and the lens
    // gradient read from it — amplified by the long lever arm up the lamp —
    // made the caustics flicker.
    for (let k = 0; k < phi.length; k++) phi[k] = 1 - Math.exp(-phiRaw[k]);
    fT.fill(0); fS.fill(0); wallE.fill(0);

    const N_REL = 1.43 / 1.34, LENS = (N_REL - 1) * rowH;
    const ABSORB = noWax ? 0 : 0.18;
    const poolRow = Math.floor(0.82 * NR);
    const ySrc = (poolRow + 1) * rowH;
    const srcHalf = hw(ySrc) * 0.97;
    const NX = 16, NA = 40;   // many angles: a broad source's caustics overlap and soften
    // Per-row wall radius / slope and per-cell transmittance, computed once
    // per trace instead of per ray step.
    if (!this._rowW) { this._rowW = new Float32Array(NR); this._rowS = new Float32Array(NR); }
    const rowWall = this._rowW, rowSlope = this._rowS;
    for (let r = 0; r < NR; r++) {
      const y = (r + 0.5) * rowH;
      rowWall[r] = hw(y);
      rowSlope[r] = (hw(y + 0.5 * rowH) - hw(y - 0.5 * rowH)) / rowH;
    }
    // Absorption from the same continuous field (massRaw units: 2 per
    // particle-area of wax, see fracPerUnit).
    const trans = this._trans || (this._trans = new Float32Array(NC * NR));
    for (let k = 0; k < trans.length; k++) trans[k] = Math.exp(-(phiRaw[k] / fracPerUnit) * ABSORB);
    // linear interpolation of a row-major grid at x within row r
    const lerpRow = (grid, base, x) => {
      const g = x / colW - 0.5;
      let c0 = Math.floor(g), f = g - c0;
      if (c0 < 0) { c0 = 0; f = 0; } else if (c0 >= NC - 1) { c0 = NC - 2; f = 1; }
      return grid[base + c0] * (1 - f) + grid[base + c0 + 1] * f;
    };
    // Per angle family, every ray's state at every row (for ray tubes)
    if (!this._rt || this._rt.NX !== NX) {
      this._rt = { NX, x: new Float32Array(NX * NR), P: new Float32Array(NX * NR),
                   b: new Int8Array(NX * NR), xs: new Float32Array(NX * NR),
                   Ps: new Float32Array(NX * NR) };
    }
    const RT = this._rt;
    const splat = (grid, r, x, e) => {
      const g = x / colW - 0.5, c0 = Math.floor(g), f = g - c0;
      if (c0 >= 0 && c0 < NC) grid[r * NC + c0] += e * (1 - f);
      if (c0 + 1 >= 0 && c0 + 1 < NC) grid[r * NC + c0 + 1] += e * f;
    };
    // Spread a tube's power uniformly over [xa, xb] (energy-conserving):
    // narrow tube = focused light, wide tube = spread-out light.
    const tube = (grid, r, xa, xb, e) => {
      if (xa > xb) { const t = xa; xa = xb; xb = t; }
      // focused tubes keep a minimum width of one cell (no switch to a point)
      if (xb - xa < colW) { const m = 0.5 * (xa + xb); xa = m - 0.5 * colW; xb = m + 0.5 * colW; }
      const len = xb - xa;
      const c0 = Math.max(0, Math.floor(xa / colW)), c1 = Math.min(NC - 1, Math.floor(xb / colW));
      for (let c = c0; c <= c1; c++) {
        const ov = Math.min(xb, (c + 1) * colW) - Math.max(xa, c * colW);
        if (ov > 0) grid[r * NC + c] += e * ov / len;
      }
    };

    for (let ia = 0; ia < NA; ia++) {
      // stratified Lambertian angles, limited to ±70° from vertical
      const ang = Math.asin((((ia + 0.5) / NA) * 2 - 1) * Math.sin(70 * Math.PI / 180));
      const slope0 = Math.tan(ang);
      for (let ix = 0; ix < NX; ix++) {
        const u = ((ix + 0.5) / NX) * 2 - 1;
        const x0 = cx + u * srcHalf;
        let P = Math.max(0, 1 - Math.abs(u * u * u)) * glow / (NX * NA);
        let x = x0, slope = slope0, bounces = 0;
        let xs = x0, Ps = P;
        for (let r = poolRow; r >= 0; r--) {
          const W = rowWall[r], o = ix * NR + r;
          if (P > 1e-7) {
            const base = r * NC;
            slope += LENS * (lerpRow(phi, base, x + 2 * colW) - lerpRow(phi, base, x - 2 * colW)) / (4 * colW);
            if (slope > 3) slope = 3; else if (slope < -3) slope = -3;
            P *= lerpRow(trans, base, x);
            const xPrev = x;
            x += slope * rowH;
            if (Math.abs(x - cx) > W && bounces < 6) {
              const side = x > cx ? 1 : 0, sgn = side ? 1 : -1;
              const dhw = rowSlope[r];
              const dl = 1 / Math.sqrt(slope * slope + 1);
              const dx = slope * dl, dy = -dl;
              const nl = 1 / Math.sqrt(1 + dhw * dhw);
              const Nx = sgn * nl, Ny = -dhw * nl;
              const ci = dx * Nx + dy * Ny;
              if (ci > 0) {
                // Where in this row the ray actually meets the wall (s = 0 at
                // the start of the step, 1 at the end). Reflecting there and
                // continuing for the rest of the step keeps paths continuous
                // as rays drift across the wall, instead of snapping a row.
                const d0 = Math.abs(xPrev - cx), d1 = Math.abs(x - cx);
                const s = d1 > d0 ? Math.min(1, Math.max(0, (W - d0) / (d1 - d0))) : 0;
                // Light striking the glass, split between the two nearest rows
                // by the exact hit height (row coordinate rr, centre = r + 0.5).
                const rr = r + 1 - s - 0.5;
                const r0 = Math.floor(rr), fr = rr - r0;
                if (r0 >= 0 && r0 < NR) wallE[r0 * 2 + side] += P * (1 - fr);
                if (r0 + 1 >= 0 && r0 + 1 < NR) wallE[(r0 + 1) * 2 + side] += P * fr;
                const R = fresnelR(ci, 1.34, 1.0) * 0.96;
                const rx = dx - 2 * ci * Nx;
                let ry = dy - 2 * ci * Ny;
                if (ry > -0.05) ry = -0.05;                  // keep marching upward
                slope = rx / -ry;
                P *= R;
                x = cx + sgn * W + slope * (1 - s) * rowH;   // rest of the step, reflected
                if (Math.abs(x - cx) > W) x = cx + sgn * (W - 0.5);
                bounces++;
              }
            }
            if (Math.abs(x - cx) > W + 1) P = 0;
          }
          RT.x[o] = x; RT.P[o] = P; RT.b[o] = bounces;
          // straight twin: no lensing, lost at the wall
          if (Ps > 1e-7) {
            Ps *= lerpRow(trans, r * NC, xs);
            xs += slope0 * rowH;
            if (Math.abs(xs - cx) > W) Ps = 0;
          }
          RT.xs[o] = xs; RT.Ps[o] = Ps;
        }
      }
      // Deposit ray tubes between neighbouring rays of this family.
      for (let r = poolRow; r >= 0; r--) {
        for (let ix = 0; ix < NX - 1; ix++) {
          const o0 = ix * NR + r, o1 = o0 + NR;
          const P0 = RT.P[o0], P1 = RT.P[o1];
          if (P0 > 0 || P1 > 0) {
            // Neighbours stay a tube even across a reflection fold as long as
            // they are close (they always are right at the fold); only a
            // genuinely torn tube falls back to point splats.
            // A very wide "tube" means the neighbours were torn apart (e.g. one
            // side of a fold); blend smoothly from tube to two point splats
            // between 4 and 8 cells wide rather than switching.
            let wSplat = 1;
            if (P0 > 0 && P1 > 0) {
              const span = Math.abs(RT.x[o0] - RT.x[o1]) / colW;
              wSplat = span <= 4 ? 0 : span >= 8 ? 1 : (span - 4) / 4;
              if (wSplat < 1) tube(fT, r, RT.x[o0], RT.x[o1], 0.5 * (P0 + P1) * (1 - wSplat));
            }
            if (wSplat > 0) {
              if (P0 > 0) splat(fT, r, RT.x[o0], 0.5 * P0 * wSplat);
              if (P1 > 0) splat(fT, r, RT.x[o1], 0.5 * P1 * wSplat);
            }
          }
          const S0 = RT.Ps[o0], S1 = RT.Ps[o1];
          if (S0 > 0 && S1 > 0) tube(fS, r, RT.xs[o0], RT.xs[o1], 0.5 * (S0 + S1));
          else { if (S0 > 0) splat(fS, r, RT.xs[o0], 0.5 * S0); if (S1 > 0) splat(fS, r, RT.xs[o1], 0.5 * S1); }
        }
      }
    }
    // wall irradiance: light 1-2-1 smoothing along the height
    const tmp = this._wallTmp || (this._wallTmp = new Float32Array(wallE.length));
    for (let r = 0; r < NR; r++) for (let sd = 0; sd < 2; sd++) {
      const a = wallE[Math.max(0, r - 1) * 2 + sd], b = wallE[r * 2 + sd], c = wallE[Math.min(NR - 1, r + 1) * 2 + sd];
      tmp[r * 2 + sd] = 0.25 * a + 0.5 * b + 0.25 * c;
    }
    wallE.set(tmp);
  }

  // ---- Pool light on the wall behind: a 3-D ray trace ---------------
  // The bulb lights the wax pool from below; paraffin scatters strongly, so
  // the pool's top surface glows as a broad, soft (Lambertian) source —
  // modelled as 7 points spread over it. Rays toward the back are followed
  // in 3-D:
  //  • the globe is the real surface of revolution ρ = R(y), so its normal
  //    (x/ρ, −R′(y), z/ρ) bends light both sideways and vertically;
  //  • each free wax blob is an ellipsoid fitted to its particles (depth
  //    from the blob depth the renderer already uses), and
  //    rays refract into and out of it (Snell, n 1.43 / 1.34), with wax
  //    absorbing along the chord;
  //  • at the glass, rays refract liquid → air with Fresnel loss; past the
  //    critical angle they reflect back inside (TIR) and try again;
  //  • the liquid attenuates along the path (the same e^-2.2 per lamp
  //    height as the in-lamp light).
  // Rays land on a flat wall WALL_DIST behind the axis. Neighbouring rays
  // (from the same source point) form small quads and each quad's power is
  // spread over the patch it covers — ray-tube density, i.e. true 2-D
  // caustics: converging light (a blob lensing) is bright, diverging light
  // is dim. The broad source softens them into drifting patches. Liquid-
  // path and wax-path light are kept separate so each can tint the wall.
  traceWall3D(sim, px, py, glow) {
    const GWc = this.WG_W, GHc = this.WG_H;
    const cellW = VIEW_W / GWc, cellH = VIEW_H / GHc;
    const L = this.wlL, Wx = this.wlW;
    L.fill(0); Wx.fill(0);
    const cx = SIM_W * 0.5, D = this.WALL_DIST, Sy = SIM_H * 0.965;
    const ETA_G = 1.34, NW_ = 1.43 / 1.34;
    const MU_WAX = 0.004, MU_LIQ = 2.2 / SIM_H;
    // globe radius / slope tables (per sim px of height)
    if (!this._Rt) {
      const n = SIM_H + 2;
      this._Rt = new Float32Array(n); this._Rs = new Float32Array(n);
      for (let y = 0; y < n; y++) this._Rt[y] = bottleHalfFrac(Math.min(1, y / SIM_H)) * SIM_W;
      for (let y = 0; y < n; y++) this._Rs[y] = 0.5 * (this._Rt[Math.min(n - 1, y + 1)] - this._Rt[Math.max(0, y - 1)]);
    }
    const Rt = this._Rt, Rs = this._Rs, RN = Rt.length - 1;
    const Rof = (y) => { const yc = y < 0 ? 0 : y > RN - 1 ? RN - 1 : y; const i = yc | 0, f = yc - i; return Rt[i] * (1 - f) + Rt[i + 1] * f; };
    const Rsl = (y) => { const yc = y < 0 ? 0 : y > RN - 1 ? RN - 1 : y; const i = yc | 0, f = yc - i; return Rs[i] * (1 - f) + Rs[i + 1] * f; };

    const nb = this.buildWallBlobs(sim, px, py);
    const B = this._wBlobs, BS = this._wBlobS;
    // ray / ellipsoid: sets h0, h1 (ray units) and returns true on a hit.
    // (No array per call — this runs thousands of times per trace.)
    let h0 = 0, h1 = 0;
    const hitE = (b, X, Y, Z, dx, dy, dz) => {
      const q = b * 6, ax = B[q + 3], ay = B[q + 4], az = B[q + 5];
      const ox = (X - B[q]) / ax, oy = (Y - B[q + 1]) / ay, oz = (Z - B[q + 2]) / az;
      const ex = dx / ax, ey = dy / ay, ez = dz / az;
      const A2 = ex * ex + ey * ey + ez * ez, Bh = ox * ex + oy * ey + oz * ez, C = ox * ox + oy * oy + oz * oz - 1;
      const disc = Bh * Bh - A2 * C;
      if (disc <= 0) return false;
      const sq = Math.sqrt(disc);
      h0 = (-Bh - sq) / A2; h1 = (-Bh + sq) / A2;
      return true;
    };
    // refract d through a surface with unit normal n facing against d
    // (GLSL refract); returns false on total internal reflection
    const out3 = this._o3 || (this._o3 = new Float64Array(3));
    const refract = (dx, dy, dz, nx, ny, nz, eta) => {
      const ci = -(dx * nx + dy * ny + dz * nz);
      const k = 1 - eta * eta * (1 - ci * ci);
      if (k < 0) return false;
      const f = eta * ci - Math.sqrt(k);
      out3[0] = eta * dx + f * nx; out3[1] = eta * dy + f * ny; out3[2] = eta * dz + f * nz;
      return true;
    };

    // ---- sources on the pool surface; per source, rays over the upward
    // cone (equal solid angle), toward the back ----
    const ySrc = 0.875 * SIM_H, rSrc = 0.55 * Rof(ySrc);
    const SRC = [[0, 0]];
    for (let k = 0; k < 4; k++) SRC.push([rSrc * Math.cos(k * Math.PI / 2 + 0.4), rSrc * Math.sin(k * Math.PI / 2 + 0.4)]);
    const NPH = 18, NAZ = 30, PHMAX = 1.35;
    const cMin = Math.cos(PHMAX);
    const RX = this._wrX || (this._wrX = new Float32Array(NPH * NAZ));
    const RY = this._wrY || (this._wrY = new Float32Array(NPH * NAZ));
    const RP = this._wrP || (this._wrP = new Float32Array(NPH * NAZ));
    const RW = this._wrW || (this._wrW = new Float32Array(NPH * NAZ));
    for (const [sx0, sz0] of SRC) {
    for (let ip = 0; ip < NPH; ip++) {
      const c = 1 - (1 - cMin) * (ip + 0.5) / NPH, sph = Math.sqrt(1 - c * c);
      for (let ia = 0; ia < NAZ; ia++) {
        const idx = ip * NAZ + ia;
        RP[idx] = 0;
        const al = Math.PI + Math.PI * (ia + 0.5) / NAZ;      // dz < 0: toward the wall
        let dx = sph * Math.cos(al), dy = -c, dz = sph * Math.sin(al);
        let X = sx0, Y = ySrc, Z = sz0;
        let P = c * glow / SRC.length;                        // Lambertian, upward
        let waxLen = 0, liqLen = 0, inB = -1, refl = 0, exited = false;
        for (let ev = 0; ev < 10 && P > 1e-4; ev++) {
          if (inB < 0) {
            // is the start point inside a blob?
            if (ev === 0) {
              for (let b = 0; b < nb; b++) {
                if (hitE(b, X, Y, Z, dx, dy, dz) && h0 < 0 && h1 > 0) { inB = b; break; }
              }
              if (inB >= 0) continue;
            }
            // nearest blob entry ahead
            let tB = 1e9, bB = -1;
            for (let b = 0; b < nb; b++) {
              if (hitE(b, X, Y, Z, dx, dy, dz) && h0 > 1e-3 && h0 < tB) { tB = h0; bB = b; }
            }
            // march to the glass (or the blob, whichever first)
            let t = 0, tExit = -1;
            const STEP = 8;
            while (t < tB) {
              const tn = Math.min(t + STEP, tB);
              const Yn = Y + dy * tn;
              if (Yn < 0.05 * SIM_H || Yn > 0.995 * SIM_H) { P = 0; break; }
              const Xn = X + dx * tn, Zn = Z + dz * tn;
              if (Xn * Xn + Zn * Zn > Rof(Yn) ** 2) {
                let lo = t, hi = tn;
                for (let k = 0; k < 10; k++) {
                  const m = 0.5 * (lo + hi), Ym = Y + dy * m, Xm = X + dx * m, Zm = Z + dz * m;
                  if (Xm * Xm + Zm * Zm > Rof(Ym) ** 2) hi = m; else lo = m;
                }
                tExit = lo; break;
              }
              t = tn;
            }
            if (P === 0) break;
            if (tExit >= 0) {
              X += dx * tExit; Y += dy * tExit; Z += dz * tExit; liqLen += tExit;
              const rho = Math.sqrt(X * X + Z * Z) || 1e-6;
              let nx = X / rho, ny = -Rsl(Y), nz = Z / rho;
              const nl = Math.sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl;
              const ci = dx * nx + dy * ny + dz * nz;
              if (ci > 0 && refract(dx, dy, dz, -nx, -ny, -nz, ETA_G)) {
                P *= 1 - fresnelR(ci, ETA_G, 1.0);
                dx = out3[0]; dy = out3[1]; dz = out3[2];
                exited = true; break;
              }
              // total internal reflection: back inside
              if (++refl > 2) { P = 0; break; }
              dx -= 2 * ci * nx; dy -= 2 * ci * ny; dz -= 2 * ci * nz;
              X -= nx * 0.5; Y -= ny * 0.5; Z -= nz * 0.5;
              P *= 0.96;
              continue;
            }
            // enter blob bB
            X += dx * tB; Y += dy * tB; Z += dz * tB; liqLen += tB;
            const q = bB * 6;
            let nx = (X - B[q]) / (B[q + 3] * B[q + 3]), ny = (Y - B[q + 1]) / (B[q + 4] * B[q + 4]), nz = (Z - B[q + 2]) / (B[q + 5] * B[q + 5]);
            const nl = Math.sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl;
            const nwB = 1 + BS[bB] * (NW_ - 1);            // faded refractive contrast
            if (refract(dx, dy, dz, nx, ny, nz, 1 / nwB)) { dx = out3[0]; dy = out3[1]; dz = out3[2]; }
            inB = bB;
          } else {
            // inside blob inB: go to its far side
            const t1 = hitE(inB, X, Y, Z, dx, dy, dz) ? Math.max(h1, 0) : 0;
            X += dx * t1; Y += dy * t1; Z += dz * t1;
            waxLen += t1 * BS[inB];
            P *= Math.exp(-MU_WAX * t1 * BS[inB]);
            const q = inB * 6;
            let nx = (X - B[q]) / (B[q + 3] * B[q + 3]), ny = (Y - B[q + 1]) / (B[q + 4] * B[q + 4]), nz = (Z - B[q + 2]) / (B[q + 5] * B[q + 5]);
            const nl = Math.sqrt(nx * nx + ny * ny + nz * nz); nx /= nl; ny /= nl; nz /= nl;
            // outward normal faces along d here → use −n for the formula
            const nwO = 1 + BS[inB] * (NW_ - 1);
            if (refract(dx, dy, dz, -nx, -ny, -nz, nwO)) { dx = out3[0]; dy = out3[1]; dz = out3[2]; }
            else { const ci = dx * nx + dy * ny + dz * nz; dx -= 2 * ci * nx; dy -= 2 * ci * ny; dz -= 2 * ci * nz; }
            X += dx * 1e-2; Y += dy * 1e-2; Z += dz * 1e-2;
            inB = -1;
          }
        }
        if (!exited || P <= 1e-5 || dz > -1e-4) continue;
        P *= Math.exp(-MU_LIQ * liqLen);
        const sW = (-D - Z) / dz;
        RX[idx] = X + dx * sW + cx + VIEW_M;              // view coords
        RY[idx] = Y + dy * sW + VIEW_T;
        RP[idx] = P;
        RW[idx] = Math.min(1, waxLen / 20);          // share of wax-path light
      }
    }

    // ---- ray-tube (quad) deposition ----
    // Each quad of neighbouring rays is a tube of light. Its power is
    // spread over the quad's actual footprint by tessellating it (m × m
    // pieces, m ∝ its size on the wall) and splatting each piece's share
    // bilinearly — smooth and energy-conserving, with no bounding-box
    // overlap (which left ripples along each ring of rays). Converging
    // tubes (small footprint) come out bright: caustics.
    const MAXSPAN = 280;              // sim px; wider = genuinely torn (e.g. across a TIR fold)
    const splat = (x, y, e, wf) => {
      const gx = x / cellW - 0.5, gy = y / cellH - 0.5;
      const c0 = Math.floor(gx), r0 = Math.floor(gy), fx = gx - c0, fy = gy - r0;
      for (let j = 0; j < 2; j++) {
        const r = r0 + j; if (r < 0 || r >= GHc) continue;
        const wy = j ? fy : 1 - fy;
        for (let i = 0; i < 2; i++) {
          const c = c0 + i; if (c < 0 || c >= GWc) continue;
          const w = wy * (i ? fx : 1 - fx) * e, k = r * GWc + c;
          L[k] += w * (1 - wf); Wx[k] += w * wf;
        }
      }
    };
    for (let ip = 0; ip < NPH - 1; ip++) {
      for (let ia = 0; ia < NAZ - 1; ia++) {
        const i00 = ip * NAZ + ia, i01 = i00 + 1, i10 = i00 + NAZ, i11 = i10 + 1;
        const P0 = RP[i00], P1 = RP[i01], P2 = RP[i10], P3 = RP[i11];
        if (P0 <= 0 || P1 <= 0 || P2 <= 0 || P3 <= 0) continue;
        const ax0 = RX[i00], ay0 = RY[i00], ax1 = RX[i01], ay1 = RY[i01];
        const bx0 = RX[i10], by0 = RY[i10], bx1 = RX[i11], by1 = RY[i11];
        const span = Math.max(Math.hypot(ax0 - bx1, ay0 - by1), Math.hypot(ax1 - bx0, ay1 - by0));
        if (span > MAXSPAN) continue;                          // torn tube
        const xmin = Math.min(ax0, ax1, bx0, bx1), xmax = Math.max(ax0, ax1, bx0, bx1);
        const ymin = Math.min(ay0, ay1, by0, by1), ymax = Math.max(ay0, ay1, by0, by1);
        if (xmax < 0 || xmin > VIEW_W || ymax < 0 || ymin > VIEW_H) continue;
        const E = 0.25 * (P0 + P1 + P2 + P3);
        const wf = 0.25 * (RW[i00] + RW[i01] + RW[i10] + RW[i11]);
        const m = Math.max(1, Math.min(10, Math.ceil(span / (2.5 * cellW))));   // pieces ≈ 2.5 cells; smoothing fills between
        const e = E / (m * m);
        for (let a = 0; a < m; a++) {
          const u = (a + 0.5) / m;
          const lx0 = ax0 + (bx0 - ax0) * u, ly0 = ay0 + (by0 - ay0) * u;   // edge ip→ip+1 at ia
          const lx1 = ax1 + (bx1 - ax1) * u, ly1 = ay1 + (by1 - ay1) * u;   // … at ia+1
          for (let b = 0; b < m; b++) {
            const v = (b + 0.5) / m;
            splat(lx0 + (lx1 - lx0) * v, ly0 + (ly1 - ly0) * v, e, wf);
          }
        }
      }
    }
    }   // sources
    // Light 2-D smoothing (≈2 cells) against residual sampling noise.
    const T = this.wgTmp;
    for (const G of [L, Wx]) {
      for (let r = 0; r < GHc; r++) this.wgRowBlur(G, r * GWc, 1, 2, T);
      for (let c = 0; c < GWc; c++) this.wgColBlur(T, c, GWc, 2, G);
    }
  }
}
