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
uniform vec3  uCompressColor;  // accent color for high-compression zones
uniform sampler2D uColMass;   // 2D mass grid (NUM_COLS × NUM_ROWS), normalized 0-1
uniform int   uNumCols;       // grid columns (50)
uniform int   uNumRows;       // grid rows (30)

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

  // -------- Per-blob metaball field, temperature & compression ---------
  const int MAX_G = 32;
  float fields[MAX_G];
  float wtemps[MAX_G];
  float wcomps[MAX_G];  // weighted compression per group
  for (int g = 0; g < MAX_G; g++) { fields[g] = 0.0; wtemps[g] = 0.0; wcomps[g] = 0.0; }
  float h2 = uH * uH;

  for (int i = 0; i < 700; i++) {
    if (i >= uCount) break;
    vec4 part = texelFetch(uParticles, ivec2(i, 0), 0);
    float comp = texelFetch(uParticles, ivec2(i, 1), 0).r;  // compression from row 1
    vec2 d = simPos - part.xy;
    float r2 = dot(d, d);
    if (r2 < h2) {
      float w = 1.0 - r2 / h2;
      float k = w * w * w;
      int g = int(part.w + 0.5);
      g = clamp(g, 0, MAX_G - 1);
      fields[g] += k;
      wtemps[g] += k * part.z;
      wcomps[g] += k * comp;
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
  // Average compression for the dominant blob at this pixel
  float compAvg = field > 0.001 ? (wcomps[dom] / field) : 0.0;
  // Smooth onset: ramp from 0 at comp=0.3 to 1 at comp=1.5
  // so only genuinely compressed regions light up
  // Log scale: compresses high values so medium blobs are visible
  // but large blobs don't blow out. Range: ~0 at rest → ~0.7 at heavy compression.
  float compRaw = max(compAvg - 0.2, 0.0);  // dead zone below 0.2
  float compIntensity = log(1.0 + compRaw * 3.0) / log(4.0);

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

  // -------- Volumetric light from below ---------
  // The texture holds per-column light intensity: bright where light from
  // the pool passes through unobstructed, dark where wax blocks it.
  // Horizontally blurred on CPU with mass-dependent kernel for refraction.
  if (uNumCols > 0 && uNumRows > 0) {
    float colU = simPos.x / uSim.x;
    float colV = 1.0 - t;   // texV: 0=top of lamp, 1=bottom
    float light = texture(uColMass, vec2(colU, colV)).r;

    // God ray: additive warm glow scaled by bulb glow slider
    vec3 rayColor = mix(uHot, vec3(1.0, 0.97, 0.90), 0.3);
    // Envelope: strong in lower-mid bulb, gentler fade toward top
    float envelope = smoothstep(0.93, 0.65, t) * smoothstep(0.02, 0.15, t);
    float glowScale = uGlow / 0.38;  // normalized so default glow=0.38 → 1.0
    fluidBg += rayColor * light * envelope * 0.80 * glowScale;

    // Darken where light is blocked (1 - light = shadow)
    float shadow = (1.0 - light) * envelope * 0.45 * glowScale;
    fluidBg *= 1.0 - shadow;
  }

  // -------- Wax shading from temperature ---------
  float tempN = clamp((temp - 0.18) / 0.85, 0.0, 1.0);
  vec3 waxColor = mix(uCold, uHot, smoothstep(0.0, 1.0, tempN));
  float lightFromBelow = mix(0.55, 1.15, pow(t, 1.4));
  waxColor *= lightFromBelow;
  waxColor = pow(waxColor, vec3(0.95));

  // -------- Compression: boost existing color ---------
  // Under pressure, brighten and saturate the current wax color
  // rather than injecting a foreign accent. This looks natural
  // at any temperature — hot wax glows hotter, cool wax gets richer.
  float compLuma = dot(waxColor, vec3(0.299, 0.587, 0.114));
  vec3 compSaturated = waxColor / max(compLuma, 0.01) * compLuma; // normalize then re-apply
  // Boost: brighten by up to 40% and increase saturation
  waxColor = mix(waxColor, waxColor * 1.4 + (waxColor - vec3(compLuma)) * 0.5, compIntensity * 0.6);

  // -------- Physically-based spherical refractivity ---------
  float blobSz = clamp(uBlobSize[dom], 0.0, 1.0);

  float threshold = 0.55;
  float alpha = smoothstep(threshold - 0.18, threshold + 0.04, field);

  // shell: 1 at the metaball boundary, 0 deep inside
  float centerness = smoothstep(threshold + 0.20, threshold + 0.55, field);
  float shell = 1.0 - centerness;

  // cosTheta: 1 at center (looking straight through), 0 at rim (glancing)
  float cosTheta = centerness;

  // ---- Fresnel reflectance at the wax-fluid interface ----
  float fresnel = fresnelSchlick(cosTheta);

  // ---- Pressure rim glow ----
  // Under compression, the rim brightens with the wax's own color
  // instead of going dark from Fresnel — like the wax is glowing hot.
  float pressureRim = shell * compIntensity * 1.8;
  waxColor += waxColor * pressureRim * 0.5;

  // Edge darkening from Fresnel — reduced under compression.
  float edgeDarken = 1.0 - fresnel * 0.6 * blobSz * (1.0 - compIntensity * 0.7);
  waxColor *= edgeDarken;

  // ---- Refraction-based specular highlight ----
  float cosRefracted = snellRefract(cosTheta);
  float causticConcentration = cosTheta / max(0.01, cosRefracted);
  // Under compression: specular intensifies
  float compSpecBoost = 1.0 + compIntensity * 1.5;
  float specBase = pow(centerness, mix(5.0, 2.5, blobSz) * (1.0 - compIntensity * 0.15));
  float specStrength = mix(0.08, 0.25, blobSz) * compSpecBoost;
  float highlight = specBase * specStrength * (1.0 - fresnel) *
                    min(2.0, causticConcentration);
  // Specular tinted by the wax color itself — warm highlight on warm wax,
  // cool highlight on cool wax.
  vec3 specTint = mix(vec3(1.0, 0.95, 0.85), waxColor * 2.0 + vec3(0.3), compIntensity * 0.6);
  waxColor += specTint * highlight;

  // ---- Uniform translucency + pressure opacity ----
  // Base 10% transparency, plus a soft center fade so the core
  // of each blob feels translucent rather than a solid disc.
  alpha *= 0.82;
  alpha *= 1.0 - centerness * 0.30;
  alpha *= max(1.0 - compIntensity * 0.65, 0.40);  // clip at 60% transparency

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
    otherAlpha *= 0.90;
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
