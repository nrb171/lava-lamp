// ---------------- HDR pipeline ----------------
// The scene is shaded in linear light ("scene-referred": values above 1
// are fine, e.g. the glowing pool). Display happens in one place: exposure
// → filmic curve → sRGB. The curve is the ACES fit (Narkowicz 2015), with
// a toe that deepens darks and a shoulder that rolls off highlights. It is
// applied to the brightest channel and the colour scaled to match, so
// saturated wax stays saturated instead of bleaching toward white.
uniform float uExposure;
vec3 acesFit(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
float acesFit1(float x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 linearToSrgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
float finishHash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
vec3 finishColor(vec3 hdr, vec2 pix, vec2 res, float time) {
  // lens vignette (optical falloff happens in linear light)
  vec2 ndc = (pix / res) - 0.5;
  float vig = 1.0 - smoothstep(0.45, 0.95, length(ndc) * 1.2);
  hdr *= mix(0.6, 1.0, vig) * uExposure;
  // hue-preserving filmic curve, with a little per-channel mixed in so the
  // very brightest highlights still desaturate slightly, like film
  float m = max(max(hdr.r, hdr.g), hdr.b);
  vec3 huePres = m > 1e-5 ? hdr * (acesFit1(m) / m) : vec3(0.0);
  vec3 mapped = mix(huePres, acesFit(hdr), 0.1);
  vec3 outc = linearToSrgb(mapped);
  // dither / film grain after encoding (also hides 8-bit banding)
  outc += (finishHash(pix + time * 0.01) - 0.5) * (1.5 / 255.0);
  return outc;
}
