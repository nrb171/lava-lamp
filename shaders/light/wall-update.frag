#version 300 es
// Per trace: filter the 2×-resolution splat down to the wall grid (a tent
// ≈ 3 grid cells wide, from 3×3 bilinear taps that each average a 2×2
// block), normalise, then baseline = running average; drawn = light above
// it, lightly smoothed in time. Outputs: new baseline, new drawn wall.
precision highp float;
uniform highp sampler2D uIrr;   // 2× resolution
uniform highp sampler2D uAvg;
uniform highp sampler2D uPrev;
uniform float uInvRef, uAlpha, uSM, uInit;
uniform float uFull;            // 1: draw all the light on the wall, not just what's above the baseline
layout(location = 0) out vec4 oAvg;
layout(location = 1) out vec4 oWall;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 texel = 1.0 / vec2(textureSize(uIrr, 0));
  vec2 c = (2.0 * vec2(p) + 1.0) * texel;            // centre of this cell's 2×2 block
  vec2 v = vec2(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      float w = float((2 - abs(i)) * (2 - abs(j)));
      v += w * texture(uIrr, c + 2.0 * vec2(i, j) * texel).rg;
    }
  }
  // 16 weight × 4 texels per tap (the 2×2 average), per grid cell (4 texels)
  v *= uInvRef * 4.0 / 16.0;
  vec2 a = uInit > 0.5 ? v : mix(texelFetch(uAvg, p, 0).rg, v, uAlpha);
  // (with TAA the ray grid is jittered every trace, so this history
  // averages several samplings of the light)
  vec2 target = uFull > 0.5 ? v : v - a;
  vec2 s = uInit > 0.5 ? (uFull > 0.5 ? v : vec2(0.0)) : mix(texelFetch(uPrev, p, 0).rg, target, uSM);
  oAvg = vec4(a, 0.0, 1.0);
  oWall = vec4(s, 0.0, 1.0);
}
