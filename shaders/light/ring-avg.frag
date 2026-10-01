#version 300 es
// TAA output. The ring holds the last uCount traces, each from fresh random
// ray origins and directions — independent samples of the light. Per
// texel, an accumulator (uAcc: running mean, sample count, per-sample
// variance; 32-bit) adds every trace — mean += (x − mean) / (n + 1),
// variance by Welford's method — so wherever the light isn't changing it
// keeps converging (up to MAX_N traces), far past the ring. Where the
// ring's mean differs from it by more than noise explains — 2.5 → 5 standard
// errors of an 8-sample mean, from the accumulated variance, which is far
// steadier than the ring's own 8-sample spread (that one tripped on noise
// every few seconds, restarting the average) — the light has moved: it
// restarts from the ring (mean and spread), which is recent.
precision highp float;
uniform highp sampler2DArray uRing;
uniform highp sampler2D uAcc;
uniform int   uCount;
uniform int   uSlot;        // this trace's layer
uniform float uHist;        // 0: no usable accumulator yet
layout(location = 0) out vec4 oAcc;
layout(location = 1) out vec4 oOut;
const float MAX_N = 1024.0;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float s = 0.0, s2 = 0.0;
  vec2 sv = vec2(0.0);
  for (int i = 0; i < 16; i++) {
    if (i >= uCount) break;
    vec2 v = texelFetch(uRing, ivec3(p, i), 0).rg;
    sv += v; s += v.x + v.y; s2 += (v.x + v.y) * (v.x + v.y);
  }
  float n = float(uCount);
  vec2 m = sv / n;
  // (tests use the total light, liquid + wax)
  float ringVar = uCount >= 2 ? max(s2 / n - (s / n) * (s / n), 0.0) * n / (n - 1.0) : 0.0;
  // restart state: the ring's mean, its sample count, its spread
  vec4 fresh = vec4(m, n, ringVar);
  vec4 acc = fresh;
  if (uHist > 0.5 && uCount >= 2) {
    vec4 A = texelFetch(uAcc, p, 0);          // (mean liquid, mean wax, n, variance of the total)
    float se = sqrt(max(A.a, ringVar * 0.25) / n);
    float z = abs(m.x + m.y - A.r - A.g) / (se + 0.01 * abs(m.x + m.y) + 1e-4);   // (a floor where there is no light)
    float t = smoothstep(2.5, 5.0, z);
    vec2 x = texelFetch(uRing, ivec3(p, uSlot), 0).rg;
    float k = 1.0 / (min(A.b, MAX_N - 1.0) + 1.0);
    vec2 mean = A.rg + (x - A.rg) * k;
    float d0 = x.x + x.y - A.r - A.g, d1 = x.x + x.y - mean.x - mean.y;
    float var = A.a + (d0 * d1 - A.a) * k;
    acc = mix(vec4(mean, min(A.b + 1.0, MAX_N), var), fresh, t);
  }
  oAcc = acc;
  oOut = vec4(acc.rg, 0.0, 1.0);
}
