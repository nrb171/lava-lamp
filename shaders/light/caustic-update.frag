#version 300 es
// 3×3 tent against residual sampling noise, normalise, smooth in time.
// Out: R = caustic excess (traced − reference, over blob-touched tubes),
//      G = the full reference field (the light without wax lenses).
precision highp float;
uniform highp sampler2D uAcc;   // R: traced, G: reference (blob tubes)
uniform highp sampler2D uRef;   // G: full reference
uniform highp sampler2D uPrev;
uniform float uInvRef, uSM, uInit;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 hi = textureSize(uAcc, 0) - 1;
  vec2 v = vec2(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      float w = float((2 - abs(i)) * (2 - abs(j)));
      v += w * texelFetch(uAcc, clamp(p + ivec2(i, j), ivec2(0), hi), 0).rg;
    }
  }
  // The reference is smooth; a taller tent (≈ a slab between planes)
  // removes the faint per-slab steps.
  float g = 0.0, gw = 0.0;
  for (int j = -4; j <= 4; j++) {
    for (int i = -1; i <= 1; i++) {
      float w = float((2 - abs(i)) * (5 - abs(j)));
      g += w * texelFetch(uRef, clamp(p + ivec2(i, 2 * j), ivec2(0), hi), 0).g;
      gw += w;
    }
  }
  vec2 cur = vec2((v.r - v.g) / 16.0, g / gw) * uInvRef;
  // (with TAA the ray grid is jittered every trace, so this history
  // averages several samplings of the light)
  vec2 s = uInit > 0.5 ? cur : mix(texelFetch(uPrev, p, 0).rg, cur, uSM);
  o = vec4(s, 0.0, 1.0);
}
