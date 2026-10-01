#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uTexel;        // 1 / source (smaller) size
in vec2 vUv;
out vec4 o;
void main() {
  vec2 h = uTexel;
  vec3 c = texture(uSrc, vUv + vec2(-2.0 * h.x, 0.0)).rgb;
  c += texture(uSrc, vUv + vec2(-h.x,  h.y)).rgb * 2.0;
  c += texture(uSrc, vUv + vec2(0.0,  2.0 * h.y)).rgb;
  c += texture(uSrc, vUv + vec2( h.x,  h.y)).rgb * 2.0;
  c += texture(uSrc, vUv + vec2( 2.0 * h.x, 0.0)).rgb;
  c += texture(uSrc, vUv + vec2( h.x, -h.y)).rgb * 2.0;
  c += texture(uSrc, vUv + vec2(0.0, -2.0 * h.y)).rgb;
  c += texture(uSrc, vUv + vec2(-h.x, -h.y)).rgb * 2.0;
  o = vec4(c / 12.0, 1.0);
}
