#version 300 es
// Light reaching the glass, per row and side (texel x = 0 left, 1 right),
// for the lit glass edge: the traced light in the liquid (reference +
// excess) over a band 6-26 px in from the glass and ±30 px in height,
// relative to its clear-lamp mean along the glass.
precision highp float;
uniform highp sampler2D uCaus;
uniform vec2  uSim;
uniform float uEdgeScale;
out vec4 o;
#include "common/bottle.glsl"
void main() {
  ivec2 id = ivec2(gl_FragCoord.xy);
  ivec2 sz = textureSize(uCaus, 0);
  float side = id.x == 0 ? -1.0 : 1.0;
  float sum = 0.0, wsum = 0.0;
  for (int j = -5; j <= 5; j++) {
    float v = (float(id.y) + 0.5 + float(j) * 3.0) / float(sz.y);        // ±15 texels
    float y = (1.0 - v) * uSim.y;
    float hw = bottleHalfFrac(clamp(y / uSim.y, 0.0, 1.0)) * uSim.x;
    float wj = float(6 - abs(j));
    for (int i = 0; i < 6; i++) {
      float x = 0.5 * uSim.x + side * max(hw - 6.0 - 4.0 * float(i), 0.0);
      vec2 c = texture(uCaus, vec2(x / uSim.x, v)).rg;
      sum += wj * (c.r + c.g); wsum += wj;
    }
  }
  o = vec4(max(sum / wsum, 0.0) * uEdgeScale, 0.0, 0.0, 1.0);
}
