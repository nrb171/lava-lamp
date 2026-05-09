// ============================================================
//  Lava-lamp WebGL shaders — separated for modularity
//  Includes physically-based paraffin wax refractivity (n ≈ 1.43)
// ============================================================

const VERTEX_SHADER = `#version 300 es
in vec2 a_pos;
void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }`;

function fragmentShaderSource() {
  return `#version 300 es
precision highp float;

uniform sampler2D uParticles;
uniform int   uCount;
uniform vec2  uSim;     // simulation domain (px)
uniform vec2  uRes;     // canvas size (px)
uniform float uH;       // smoothing radius (sim px)
uniform vec3  uBg;
uniform vec3  uCold;
uniform vec3  uHot;
uniform float uGlow;
uniform float uTime;
uniform float uBlobZ[32];
uniform float uBlobSize[32];
uniform int   uPoolGroupId;

out vec4 fragColor;

// -------- Physical constants --------
// Refractive index of liquid paraffin wax at ~100°C
const float N_WAX = 1.43;
// Surrounding fluid (water/glycol mixture) — approximately
const float N_FLUID = 1.34;
// Relative index at the wax-fluid interface
const float N_REL = N_WAX / N_FLUID;

// Schlick's approximation for Fresnel reflectance
// R0 = ((n1 - n2) / (n1 + n2))^2
const float R0 = ((N_WAX - N_FLUID) * (N_WAX - N_FLUID)) /
                 ((N_WAX + N_FLUID) * (N_WAX + N_FLUID));

float fresnelSchlick(float cosTheta) {
  float x = 1.0 - cosTheta;
  return R0 + (1.0 - R0) * x * x * x * x * x;
}

// Snell's law refraction amount — returns cos(theta_t)
// for simulating the visual compression of internal features
float snellRefract(float cosI) {
  float sinI2 = 1.0 - cosI * cosI;
  float sinT2 = sinI2 / (N_REL * N_REL);
  if (sinT2 >= 1.0) return 0.0; // total internal reflection
  return sqrt(1.0 - sinT2);
}

// Match JS bottleHalfFrac exactly
float bottleHalfFrac(float t) {
  if (t < 0.03) return 0.0;
  if (t < 0.06) {
    float u = (t - 0.03) / 0.03;
    return 0.247 * (u * u * (3.0 - 2.0 * u));
  }
  if (t < 0.82) {
    float u = (t - 0.06) / 0.76;
    float e = u * u * (3.0 - 2.0 * u);
    return mix(0.247, 0.50, e);
  }
  if (t < 0.93) {
    float u = (t - 0.82) / 0.11;
    float e = u * u * (3.0 - 2.0 * u);
    return mix(0.50, 0.40, e);
  }
  if (t < 0.96) {
    float u = (t - 0.93) / 0.03;
    float e = u * u * (3.0 - 2.0 * u);
    return mix(0.40, 0.32, e);
  }
  if (t < 0.99) {
    float u = (t - 0.96) / 0.03;
    return 0.32 * (1.0 - u);
  }
  return 0.0;
}

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

void main() {
  vec2 pix = gl_FragCoord.xy;
  vec2 simPos = vec2(pix.x, uRes.y - pix.y) * (uSim / uRes);

  float t = simPos.y / uSim.y;
  float halfFrac = bottleHalfFrac(t);
  float halfW = halfFrac * uSim.x;
  float cx = uSim.x * 0.5;
  float distFromCenter = abs(simPos.x - cx);

  // -------- Per-blob metaball field & temperature ---------
  const int MAX_G = 32;
  float fields[MAX_G];
  float wtemps[MAX_G];
  for (int g = 0; g < MAX_G; g++) { fields[g] = 0.0; wtemps[g] = 0.0; }
  float h2 = uH * uH;

  for (int i = 0; i < 700; i++) {
    if (i >= uCount) break;
    vec4 part = texelFetch(uParticles, ivec2(i, 0), 0);
    vec2 d = simPos - part.xy;
    float r2 = dot(d, d);
    if (r2 < h2) {
      float w = 1.0 - r2 / h2;
      float k = w * w * w;
      int g = int(part.w + 0.5);
      g = clamp(g, 0, MAX_G - 1);
      fields[g] += k;
      wtemps[g] += k * part.z;
    }
  }

  // Pick the dominant blob at this pixel
  const float SHOW_THRESH = 0.55;
  int dom = 0;
  float maxF = 0.0;
  for (int g = 0; g < MAX_G; g++) {
    if (fields[g] > maxF) { maxF = fields[g]; dom = g; }
  }
  // Override by z among blobs that are "visible"
  int frontDom = -1;
  float frontZ = -1.0;
  int otherDom = -1;
  float otherF = 0.0;
  for (int g = 1; g < MAX_G; g++) {
    if (fields[g] > SHOW_THRESH) {
      if (uBlobZ[g] > frontZ) {
        otherDom = frontDom; otherF = (frontDom >= 0) ? fields[frontDom] : 0.0;
        frontDom = g; frontZ = uBlobZ[g];
      } else if (fields[g] > otherF) {
        otherDom = g; otherF = fields[g];
      }
    }
  }
  if (frontDom >= 0) dom = frontDom;
  float field = fields[dom];
  float temp = field > 0.001 ? (wtemps[dom] / field) : 0.18;

  // -------- Bottle / fluid masks ---------
  float glassEdgeSoft = 1.5;
  float insideGlass = 1.0 - smoothstep(halfW - glassEdgeSoft, halfW + glassEdgeSoft, distFromCenter);
  insideGlass *= step(0.001, halfFrac);

  // -------- Background inside glass ---------
  float bottomT = clamp((t - 0.55) / 0.40, 0.0, 1.0);
  float bulb = pow(clamp((t - 0.78) / 0.16, 0.0, 1.0), 1.7) * uGlow;
  float horiz = 1.0 - smoothstep(0.0, halfW * 0.95, distFromCenter);
  bulb *= mix(0.35, 1.0, horiz);

  vec3 warmTint = mix(uCold, uHot, 0.65) * 1.2;
  vec3 fluidBg = mix(uBg, uBg * 1.4 + warmTint * 0.35, bottomT);
  fluidBg = mix(fluidBg, warmTint, bulb * 0.55);
  fluidBg *= mix(0.85, 1.0, smoothstep(0.0, 1.0, t));

  // Permanent opaque pool block
  float poolBlockMask = smoothstep(0.89, 0.90, t);
  vec3 poolBlockColor = mix(uCold, uHot, 0.45) * mix(0.55, 1.15, pow(t, 1.4));
  fluidBg = mix(fluidBg, poolBlockColor, poolBlockMask);

  // -------- Wax shading from temperature ---------
  float tempN = clamp((temp - 0.18) / 0.85, 0.0, 1.0);
  vec3 waxColor = mix(uCold, uHot, smoothstep(0.0, 1.0, tempN));
  float lightFromBelow = mix(0.55, 1.15, pow(t, 1.4));
  waxColor *= lightFromBelow;
  waxColor = pow(waxColor, vec3(0.95));

  // -------- Physically-based spherical refractivity ---------
  float blobSz = clamp(uBlobSize[dom], 0.0, 1.0);

  float threshold = 0.55;
  float alpha = smoothstep(threshold - 0.18, threshold + 0.04, field);

  // shell: 1 at the metaball boundary, 0 deep inside
  // This approximates cos(theta) for a sphere where theta is the
  // angle between the view ray and the surface normal
  float centerness = smoothstep(threshold + 0.20, threshold + 0.55, field);
  float shell = 1.0 - centerness;

  // cosTheta: 1 at center (looking straight through), 0 at rim (glancing)
  // This is our proxy for the dot(view, normal) on a sphere
  float cosTheta = centerness;

  // ---- Fresnel reflectance at the wax-fluid interface ----
  // At the rim (glancing angles), more light reflects off the surface;
  // at the center (normal incidence), light passes through.
  // R0 for paraffin/fluid ≈ 0.001 — very little reflection at normal,
  // but rises steeply at glancing angles (the characteristic rim gleam).
  float fresnel = fresnelSchlick(cosTheta);

  // Edge darkening from Fresnel: the rim reflects surrounding (dark)
  // fluid back at the viewer, darkening the silhouette edge.
  // Scaled by blob size — small drops have less visible curvature.
  float edgeDarken = 1.0 - fresnel * 0.6 * blobSz;
  waxColor *= edgeDarken;

  // ---- Refraction-based specular highlight ----
  // A physically-motivated highlight: light from the bulb below enters
  // the sphere, refracts at the curved surface, and concentrates into
  // a caustic bright spot offset toward the top. The Fresnel term
  // modulates intensity (more light enters at normal incidence).
  // Snell's law compresses the highlight — higher n = tighter caustic.
  float cosRefracted = snellRefract(cosTheta);
  // The caustic concentration factor: ratio of solid angles maps to
  // (cosTheta / cosRefracted), which is > 1 for n > 1 — light is
  // compressed into a smaller cone inside the sphere.
  float causticConcentration = cosTheta / max(0.01, cosRefracted);
  float specBase = pow(centerness, mix(5.0, 2.5, blobSz));
  float specStrength = mix(0.08, 0.25, blobSz);
  // Modulate by (1 - fresnel) since reflected light doesn't enter
  float highlight = specBase * specStrength * (1.0 - fresnel) *
                    min(2.0, causticConcentration);
  waxColor += vec3(1.0, 0.95, 0.85) * highlight;

  // ---- Refraction-based translucency ----
  // Light passing through the sphere is attenuated by path length.
  // At the center the path is longest (most absorption → more opaque
  // wax color); at the rim the path through wax is short but Fresnel
  // reflection dominates. The net effect: hot wax glows through at
  // the center where the bulb light enters, rim stays crisp.
  //
  // Effective optical path through a sphere: proportional to cosTheta
  // (longest at center). Beer-Lambert-style absorption:
  float opticalPath = cosTheta * mix(1.5, 2.5, blobSz);
  float transmission = exp(-opticalPath * 0.4); // 0.4 = absorption coeff
  // Core translucency: blend toward transparent at center for hot wax,
  // letting the bulb glow show through. Modulated by temperature
  // (hot wax is more fluid and translucent) and by transmission.
  float coreTranslucency = mix(0.30, 0.55, blobSz);
  alpha *= 1.0 - centerness * coreTranslucency * tempN * (1.0 - transmission * 0.5);

  // ---- Fresnel rim boost ----
  // At glancing angles, total internal reflection makes the silhouette
  // edge crisper — more light bounces back from inside the sphere.
  float fresnelBoost = fresnel * 0.20 * blobSz;
  alpha = min(1.0, alpha + fresnelBoost * step(threshold, field));

  // Inner core detail
  float coreAlpha = smoothstep(threshold + 0.05, threshold + 0.55, field);
  vec3 inner = mix(waxColor, waxColor * 1.25 + uHot * 0.18 * tempN, coreAlpha);

  // Overlap layering: front blob over back blob
  if (otherDom >= 0) {
    float otherFieldRaw = fields[otherDom];
    float otherTemp = otherFieldRaw > 0.001 ? (wtemps[otherDom] / otherFieldRaw) : 0.18;
    float otherTempN = clamp((otherTemp - 0.18) / 0.85, 0.0, 1.0);
    vec3 otherWax = mix(uCold, uHot, smoothstep(0.0, 1.0, otherTempN)) * lightFromBelow;
    float otherAlpha = smoothstep(threshold - 0.18, threshold + 0.04, otherFieldRaw);
    float otherCenterness = smoothstep(threshold + 0.20, threshold + 0.55, otherFieldRaw);
    // Apply Fresnel-based translucency to back blob too
    float otherCosTheta = otherCenterness;
    float otherFresnel = fresnelSchlick(otherCosTheta);
    otherAlpha *= 1.0 - otherCenterness * 0.45 * otherTempN;
    fluidBg = mix(fluidBg, otherWax, otherAlpha);
  }

  // Mix wax over fluid bg
  vec3 inside = mix(fluidBg, inner, alpha);

  // -------- Bottle frame ---------
  float capTop  = smoothstep(0.05, 0.045, t);
  float capBot  = smoothstep(0.95, 0.955, t);

  vec3 topCapCol = vec3(0.06, 0.05, 0.10);
  float neckBand = smoothstep(0.030, 0.035, t) * (1.0 - smoothstep(0.045, 0.050, t));
  topCapCol += vec3(0.10, 0.08, 0.14) * neckBand;
  float capLight = smoothstep(0.4, 0.0, distFromCenter / (uSim.x * 0.3));
  topCapCol *= mix(1.0, 1.5, capLight * (1.0 - smoothstep(0.0, 0.05, t)));

  vec3 botCapCol = mix(vec3(0.07, 0.04, 0.09), vec3(0.16, 0.10, 0.13), smoothstep(0.95, 1.00, t));
  float baseGlow = smoothstep(0.99, 0.95, t) * uGlow;
  botCapCol += uHot * baseGlow * 0.4;
  botCapCol += uCold * baseGlow * 0.18;
  float slit = smoothstep(0.948, 0.955, t) * (1.0 - smoothstep(0.955, 0.962, t));
  botCapCol += uHot * slit * uGlow * 1.4;

  vec3 frameOut = vec3(0.0);

  vec3 col;
  if (t < 0.05) {
    col = topCapCol;
  } else if (t > 0.95) {
    col = botCapCol;
  } else {
    col = mix(frameOut, inside, insideGlass);
    float rimDist = halfW - distFromCenter;
    float rim = smoothstep(0.0, 1.5, rimDist) * (1.0 - smoothstep(1.5, 4.0, rimDist));
    col += vec3(0.22, 0.18, 0.30) * rim * 0.45;
    float streakX = clamp(1.0 - abs((simPos.x - (cx - halfW * 0.55)) / 6.0), 0.0, 1.0);
    float streakY = smoothstep(0.10, 0.40, t) * (1.0 - smoothstep(0.40, 0.62, t));
    col += vec3(0.85, 0.80, 1.0) * streakX * streakY * 0.10 * insideGlass;
  }

  // Soft outer glow
  float outerGlow = smoothstep(0.96, 0.55, t) * uGlow * 0.18;
  outerGlow *= smoothstep(uSim.x * 0.9, 0.0, distFromCenter);
  col += uHot * outerGlow * (1.0 - insideGlass) * 0.25;

  // Vignette
  vec2 ndc = (pix / uRes) - 0.5;
  float vig = 1.0 - smoothstep(0.45, 0.95, length(ndc) * 1.2);
  col *= mix(0.7, 1.0, vig);

  // Film grain
  float g = (hash(pix + uTime * 0.01) - 0.5) * 0.012;
  col += g;

  fragColor = vec4(col, 1.0);
}
`;
}

// UMD export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { VERTEX_SHADER, fragmentShaderSource };
}
