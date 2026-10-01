#version 300 es
// Wall photons (wide bulb, trace.frag with uSrcSigma > 0): each ray
// that reached the wall is one sample of the light; it lands as a small
// Gaussian (σ = uSigma texels) carrying its power, split liquid / wax.
// Summed here and over the TAA ring, the samples converge to the wall's
// irradiance — soft wherever the bulb's width blurs it, as it should be.
precision highp float;
uniform highp sampler2D uRays;
uniform ivec2 uRaySize;
uniform vec2  uView;        // view size (sim px)
uniform float uSigma;       // kernel σ, target texels
out vec2 vE;
void main() {
  int i = gl_VertexID;
  vec4 r = texelFetch(uRays, ivec2(i % uRaySize.x, i / uRaySize.x), 0);
  if (r.z <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; vE = vec2(0.0); return; }
  float wf = fract(r.w);
  vE = r.z * vec2(1.0 - wf, wf) / (6.2831853 * uSigma * uSigma);
  gl_Position = vec4(r.x / uView.x * 2.0 - 1.0, 1.0 - r.y / uView.y * 2.0, 0.0, 1.0);
  gl_PointSize = 2.0 * ceil(3.0 * uSigma) + 1.0;
}
