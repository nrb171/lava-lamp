#version 300 es
precision highp float;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomStrength;   // k: fraction of light scattered into the halo
uniform float uBloomNorm;       // 1 / number of pyramid levels summed
uniform vec2 uRes;
uniform float uTime;
in vec2 vUv;
out vec4 o;
#include "common/finish.glsl"
void main() {
  // Glare as optics does it: the lens/eye scatters a small fraction k of
  // ALL light into a wide halo and the image keeps the rest. Energy-
  // conserving, so it can't wash the picture out — only sources much
  // brighter than their surroundings visibly glow.
  vec3 scene = texture(uScene, vUv).rgb;
  if (any(isnan(scene)) || any(isinf(scene))) scene = vec3(0.0);
  vec3 halo = texture(uBloom, vUv).rgb * uBloomNorm;
  vec3 hdr = mix(scene, halo, uBloomStrength);
  o = vec4(finishColor(hdr, gl_FragCoord.xy, uRes, uTime), 1.0);
}
