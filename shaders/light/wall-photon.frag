#version 300 es
precision highp float;
uniform float uSigma;
in vec2 vE;
out vec4 o;
void main() {
  float size = 2.0 * ceil(3.0 * uSigma) + 1.0;
  vec2 d = (gl_PointCoord - 0.5) * size;
  float w = exp(-dot(d, d) / (2.0 * uSigma * uSigma));
  o = vec4(vE * w, 0.0, 0.0);
}
