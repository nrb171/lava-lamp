// Small numeric helpers for the renderer.

// Variable-radius triangular blur along one row/column, O(1) per sample
// via double prefix sums (a triangle filter is a box filter applied twice).
// Equivalent to the direct sum
//   dst[i] = Σ_{|d|<=R_i, 0<=i+d<n} (1-|d|/(R_i+1))·src[i+d] / Σ weights
// but independent of R, so wide blurs cost the same as narrow ones.
export function makeTriBlur(n, maxR) {
  const PAD = maxR + 2, L = n + 2 * PAD + 2;
  const P1 = new Float64Array(L + 1), Q = new Float64Array(L + 2);
  const den = new Float64Array((maxR + 1) * n);   // weight sums (src ≡ 1)
  function prefix(src, off, stride) {
    let acc = 0;
    for (let e = 0; e <= L; e++) {
      P1[e] = acc;                                  // Σ_{k < e-PAD} src[k]
      const k = e - PAD;
      if (k >= 0 && k < n) acc += src[off + k * stride];
    }
    let q = 0;
    for (let e = 0; e <= L; e++) { Q[e] = q; q += P1[e]; }
    Q[L + 1] = q;
  }
  const sumP1 = (a, b) => Q[b + PAD + 1] - Q[a + PAD];      // Σ_{j=a..b} P1[j]
  const tri = (i, R) => sumP1(i + 1, i + R + 1) - sumP1(i - R, i);
  const ones = new Float32Array(n).fill(1);
  prefix(ones, 0, 1);
  for (let R = 0; R <= maxR; R++) for (let i = 0; i < n; i++) den[R * n + i] = tri(i, R);
  // radius: a number, or an Int array laid out like src
  return function blur(src, off, stride, radius, dst) {
    prefix(src, off, stride);
    const fixedR = typeof radius === "number" ? radius : -1;
    for (let i = 0; i < n; i++) {
      const idx = off + i * stride;
      const R = fixedR >= 0 ? fixedR : radius[idx];
      dst[idx] = R === 0 ? src[idx] : tri(i, R) / den[R * n + i];
    }
  };
}

// sRGB (0-1) → linear light, into out
export function srgbToLinear(c, out) {
  for (let i = 0; i < 3; i++) {
    const v = c[i];
    out[i] = v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  }
  return out;
}

// Unpolarised Fresnel reflectance from index n1 into n2 at incidence cos ci
// (1 beyond the critical angle: total internal reflection).
export function fresnelR(ci, n1, n2) {
  const si = Math.sqrt(Math.max(0, 1 - ci * ci));
  const st = (n1 / n2) * si;
  if (st >= 1) return 1;
  const ct = Math.sqrt(1 - st * st);
  const rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
  const rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
  return 0.5 * (rs * rs + rp * rp);
}
