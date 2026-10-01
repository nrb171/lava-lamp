#version 300 es
// Denoiser: one pass of an edge-avoiding à-trous wavelet filter (as in
// SVGF, Schied et al. 2017), run with steps 1, 2, 4, 8 — a 3×3 B-spline
// kernel, spread wider each pass. Each neighbour is weighted down by how
// far its light differs from this texel's, measured in units of their
// noise: the standard error of each accumulated mean (ring-avg.frag: per-
// sample variance / sample count). Where few samples have been gathered
// (light that just moved) differences are mostly noise and are blurred
// away; where the average has converged the error is tiny, any real
// difference stops the blur, and the texel is left as it is.
precision highp float;
uniform highp sampler2D uSrc;   // rg: light (liquid, wax)
uniform highp sampler2D uAcc;   // b: sample count, a: per-sample variance of the total
uniform int   uStep;
uniform float uK;
out vec4 o;
float se2(ivec2 q) { vec4 A = texelFetch(uAcc, q, 0); return max(A.a, 0.0) / max(A.b, 1.0); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 hi = textureSize(uSrc, 0) - 1;
  vec2 c = texelFetch(uSrc, p, 0).rg;
  float Lc = c.x + c.y, ec = se2(p);
  const float KW[3] = float[3](0.25, 0.5, 0.25);
  vec2 sum = vec2(0.0);
  float ws = 0.0;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      ivec2 q = clamp(p + ivec2(i, j) * uStep, ivec2(0), hi);
      vec2 v = texelFetch(uSrc, q, 0).rg;
      float sig = uK * sqrt(ec + se2(q)) + 1e-5;
      float w = KW[i + 1] * KW[j + 1] * exp(-abs(v.x + v.y - Lc) / sig);
      sum += w * v; ws += w;
    }
  }
  o = vec4(sum / ws, 0.0, 1.0);
}
