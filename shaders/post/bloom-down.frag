#version 300 es
// Bloom: optical glare from bright sources (lens / eye scattering). Mobile-
// friendly "dual filter" pyramid (Bjørge, SIGGRAPH 2015): each level is
// half the size of the last; the first downsample keeps only light above a
// soft threshold.
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uTexel;        // 1 / source size
uniform float uPrefilter;   // 1 on the first pass
uniform float uThreshold;
uniform float uKnee;
in vec2 vUv;
out vec4 o;
vec3 prefilter(vec3 c) {
  float br = max(c.r, max(c.g, c.b));
  float rq = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  rq = rq * rq / (4.0 * uKnee + 1e-4);
  return c * (max(rq, br - uThreshold) / max(br, 1e-4));
}
void main() {
  vec2 h = uTexel;
  vec3 c = texture(uSrc, vUv).rgb * 4.0;
  c += texture(uSrc, vUv - h).rgb;
  c += texture(uSrc, vUv + h).rgb;
  c += texture(uSrc, vUv + vec2(h.x, -h.y)).rgb;
  c += texture(uSrc, vUv - vec2(h.x, -h.y)).rgb;
  c *= 0.125;
  // one NaN/Inf pixel would otherwise be smeared across the whole pyramid
  c = mix(c, vec3(0.0), bvec3(isnan(c.r) || isinf(c.r) || isnan(c.g) || isinf(c.g) || isnan(c.b) || isinf(c.b)));
  if (uPrefilter > 0.5) c = prefilter(min(c, vec3(32.0)));
  o = vec4(c, 1.0);
}
