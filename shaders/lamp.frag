#version 300 es
precision highp float;

uniform sampler2D uParticles;   // cell-sorted: (x, y, temp, groupId + compression/4)
uniform sampler2D uCellRange;   // RGBA32F: r=start, g=count per grid cell
uniform vec2  uSim;     // simulation domain (px)
uniform float uViewM;   // wall margin shown on each side of the lamp (sim px)
uniform float uViewT;   // wall margin above and below the lamp (sim px)
uniform vec2  uRes;     // canvas size (px)
uniform float uH;       // smoothing radius (sim px) — also = cellSize
uniform ivec2 uGridDim; // active grid (cells wide, cells tall)
uniform vec3  uBg;
uniform vec3  uCold;
uniform vec3  uHot;
uniform float uGlow;
uniform float uTime;
uniform float uBlobZ[36];     // 0-31: sim blobs, 32-35: merge ghosts
uniform float uBlobSize[36];
uniform float uSizeScale;     // particles in the largest group (uBlobSize is relative to it)
uniform float uV0;            // area of one particle (sim px²)
uniform float uMuWax;         // wax absorption (per sim px)
// Pool-merge fusion. A blob that just joined the pool keeps rendering as a
// "ghost" group (id 32+k) whose fusion with the pool, uGhostMix[k], eases
// 0 → 1. At 0 the pair renders as two touching blobs; at 1 exactly as one
// body (the summed field), so the ghost can then be dropped invisibly.
uniform float uPoolId;        // pool's group id, or -1
uniform float uLens;          // 1 = view the interior through the cylindrical glass lens
uniform float uCaustics;      // 1 = draw caustics (texture G/B channels)
uniform float uDirectOut;     // 1 = no float render target: tone-map here
#include "common/finish.glsl"
// Colours below are authored in sRGB; all shading is in linear light.
#define LIN(c) pow(c, vec3(2.2))
uniform vec4  uGhostMix;
uniform sampler2D uColMass;   // 2D mass grid (NUM_COLS × NUM_ROWS), normalized 0-1
uniform sampler2D uBackdrop;  // wall behind the lamp, previous trace: R = liquid light, G = wax light
uniform sampler2D uBackdrop2; // … newest trace
uniform float uWallMix;       // 0 → previous, 1 → newest
uniform float uWall;          // 1 = draw the lit wall
// GPU caustics (see light/trace.frag): light scattered in the liquid over the
// lamp, R = excess over the lamp without wax lenses, G = that lamp's light;
// previous and newest trace
uniform sampler2D uCaus;
uniform sampler2D uCaus2;
uniform float uCausMix;
uniform float uCausGpu;       // 1 = use uCaus; 0 = the CPU grid's G/B channels
uniform sampler2D uCausEdge;  // light at the glass per row: x = 0 left, 1 right (light/caustic-edge.frag)
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

#include "common/bottle.glsl"
float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

// Opacity of a blob of relative size size01 where the metaball field is f:
// Beer–Lambert, 1 − e^(−μL), with L the path through the blob — a sphere
// of the blob's area-equivalent radius, thinner toward the rim (read from
// the field, which also stays lower in small blobs). Small blobs are thin,
// so they let more of the light behind them through.
float waxOpacity(float size01, float f) {
  float N = max(size01 * uSizeScale, 1.0);
  float rb = sqrt(N * uV0 / 3.14159265);
  // (a floor at the rim: a lens seen edge-on is still a visible surface,
  // not a fade to nothing)
  float L = 2.0 * rb * (0.3 + 0.7 * sqrt(smoothstep(0.45, 1.2, f)));
  return 1.0 - exp(-uMuWax * L);
}

void main() {
  vec2 pix = gl_FragCoord.xy;
  // canvas spans sim x ∈ [−uViewM, uSim.x + uViewM], y ∈ [−uViewT, uSim.y + uViewT]
  vec2 viewSize = vec2(uSim.x + 2.0 * uViewM, uSim.y + 2.0 * uViewT);
  vec2 simPos = vec2(pix.x * viewSize.x / uRes.x - uViewM,
                     (uRes.y - pix.y) * viewSize.y / uRes.y - uViewT);

  float t = simPos.y / uSim.y;
  // t is outside [0, 1] above and below the lamp (the full-window view);
  // clamp it wherever it feeds pow() etc., which is NaN for negative bases
  // — and one NaN pixel is smeared across the frame by the bloom.
  float tc = clamp(t, 0.0, 1.0);
  float halfFrac = bottleHalfFrac(t);
  float halfW = halfFrac * uSim.x;
  float cx = uSim.x * 0.5;
  float distFromCenter = abs(simPos.x - cx);

  // -------- The glass as a cylindrical lens ---------
  // The lamp is a solid of revolution. A view ray at apparent offset b
  // from the axis meets the curved surface at incidence sin θi = b/R and
  // refracts into the liquid (thin glass shell ≈ parallel, so air → n 1.34
  // decides it), bending toward the axis by δ = θi − θt. It crosses the
  // mid-plane — where the sim lives — at
  //     x0 = b − √(R² − b²) · tan δ
  // so the middle looks magnified ~1.34× while a wide slice near the walls
  // is squeezed into a thin band at the silhouette. The interior is
  // sampled at x0; the Fresnel reflectance at entry (→ 1 at the edge)
  // trades transmitted interior light for reflected room light.
  vec2 sp = simPos;                 // where the view ray samples the interior
  float lensT = 1.0, lensF = 0.0;   // Fresnel transmission / reflection at entry
  if (uLens > 0.5 && halfW > 1.0 && distFromCenter < halfW) {
    float b = distFromCenter;
    float sI = b / halfW;
    float sT = sI / N_FLUID;
    float delta = asin(sI) - asin(sT);
    float zin = sqrt(max(halfW * halfW - b * b, 0.0));
    float x0 = max(b - zin * tan(delta), 0.0);
    sp.x = cx + sign(simPos.x - cx) * x0;
    float cI = sqrt(max(1.0 - sI * sI, 0.0));
    float r0 = (N_FLUID - 1.0) / (N_FLUID + 1.0); r0 *= r0;
    float x5 = 1.0 - cI; x5 = x5 * x5 * x5 * x5 * x5;
    lensF = r0 + (1.0 - r0) * x5;
    lensT = 1.0 - lensF;
  }

  // -------- Per-blob metaball field, temperature & compression ---------
  // A pixel only ever overlaps a handful of blobs, so instead of 32-entry
  // per-group arrays (which mobile GPUs spill to slow memory because they
  // are indexed by texture data) we keep 4 register slots:
  //   ids = group id per slot (-1 = empty), F = Σk, WT = Σk·temp, WC = Σk·comp
  vec4 ids = vec4(-1.0);
  vec4 F = vec4(0.0), WT = vec4(0.0), WC = vec4(0.0);

  float insideRaw = distFromCenter - (halfW + 1.5);
  // Caps and everything outside the glass never show wax, so skip the
  // particle gather there (roughly a third of the canvas).
  bool needField = t >= 0.05 && t <= 0.95 && halfFrac > 0.001 && insideRaw < 0.0;
  if (needField) {
    float h2 = uH * uH;
    float invH2 = 1.0 / h2;
    // Spatial-grid lookup: only visit particles in the 3x3 cell neighborhood
    // around this pixel. cellSize == uH, so any particle within the kernel
    // radius is guaranteed to live in one of these 9 cells.
    ivec2 cellHere = ivec2(floor(sp / uH));
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        ivec2 c = cellHere + ivec2(dx, dy);
        if (c.x < 0 || c.y < 0 || c.x >= uGridDim.x || c.y >= uGridDim.y) continue;
        vec2 range = texelFetch(uCellRange, c, 0).rg;
        int start = int(range.x);
        int count = int(range.y);
        // Static bound for the compiler; the early break is the real exit.
        for (int j = 0; j < 700; j++) {
          if (j >= count) break;
          vec4 part = texelFetch(uParticles, ivec2(start + j, 0), 0);
          vec2 d = sp - part.xy;
          float r2 = dot(d, d);
          if (r2 < h2) {
            float w = 1.0 - r2 * invH2;
            float k = w * w * w;
            float gf = floor(part.w);
            float comp = fract(part.w) * 4.0;
            bvec4 m = equal(ids, vec4(gf));
            if (!any(m)) {
              // claim the first free slot (drop the particle if all 4 are taken)
              if      (ids.x < 0.0) ids.x = gf;
              else if (ids.y < 0.0) ids.y = gf;
              else if (ids.z < 0.0) ids.z = gf;
              else if (ids.w < 0.0) ids.w = gf;
              else continue;
              m = equal(ids, vec4(gf));
            }
            vec4 mv = vec4(m);
            F  += mv * k;
            WT += mv * (k * part.z);
            WC += mv * (k * comp);
          }
        }
      }
    }
  }

  // -------- Pool-merge fusion ---------
  // Where a ghost and the pool overlap, fold them into the pool slot with
  // a smooth union  F = (Fp^p + Fg^p)^(1/p),  p easing 8 → 1: p = 8 is
  // ~max(Fp, Fg) (two blobs, dark seam where they touch), p = 1 is the sum
  // (one body, neck filled in). Temperature, compression and size blend
  // with the same weights rather than being picked by depth, so no hard
  // occlusion edge sweeps through the blob while it fuses. At L = 0 the
  // result reproduces the depth-ordered look of two separate blobs (front
  // one wins wherever it is visible), which then fades into the union over
  // the first third of the ramp — so the first merged frame matches the
  // last unmerged one.
  float sizeOverride = -1.0;
  // Back layer of the depth-ordered look (the rear body seen through the
  // translucent front one), faded out with the same weight.
  float fuseBackF = 0.0, fuseBackT = 0.18, fuseBackW = 0.0;
  int poolSlot = -1;
  for (int s = 0; s < 4; s++) if (uPoolId >= 0.0 && ids[s] == uPoolId) poolSlot = s;
  if (poolSlot >= 0) {
    float poolSize = uBlobSize[int(uPoolId)];
    float sizeAcc = poolSize;
    for (int s = 0; s < 4; s++) {
      if (ids[s] >= 32.0) {
        int k = int(ids[s]) - 32;
        float L = uGhostMix[k];
        float Fp = F[poolSlot], Fg = F[s];
        float p = mix(8.0, 1.0, L);
        // normalise before pow to keep the numbers small
        float m = max(max(Fp, Fg), 1e-6);
        float a = pow(Fp / m, p), b = pow(Fg / m, p);
        float Fu = m * pow(a + b, 1.0 / p);
        float wpU = a / max(a + b, 1e-6);
        float tp = Fp > 1e-6 ? WT[poolSlot] / Fp : 0.0, tg = Fg > 1e-6 ? WT[s] / Fg : 0.0;
        float cp = Fp > 1e-6 ? WC[poolSlot] / Fp : 0.0, cg = Fg > 1e-6 ? WC[s] / Fg : 0.0;
        // depth-ordered (pre-merge) equivalent
        bool poolFront = uBlobZ[int(uPoolId)] >= uBlobZ[int(ids[s])];
        float Ffront = poolFront ? Fp : Fg;
        bool frontVis = Ffront > 0.55;                    // SHOW_THRESH
        float Focc = frontVis ? Ffront : max(Fp, Fg);
        float wpOcc = frontVis ? (poolFront ? 1.0 : 0.0) : (Fp >= Fg ? 1.0 : 0.0);
        float kU = smoothstep(0.0, 0.35, L);
        float Fback = poolFront ? Fg : Fp;
        if (frontVis && Fback > 0.55 && 1.0 - kU > fuseBackW) {
          fuseBackF = Fback;
          fuseBackT = poolFront ? tg : tp;
          fuseBackW = 1.0 - kU;
        }
        Fu = mix(Focc, Fu, kU);
        float wp = mix(wpOcc, wpU, kU), wg = 1.0 - wp;
        F[poolSlot]  = Fu;
        WT[poolSlot] = Fu * (wp * tp + wg * tg);
        WC[poolSlot] = Fu * (wp * cp + wg * cg);
        sizeAcc = wp * sizeAcc + wg * uBlobSize[int(ids[s])];
        ids[s] = -1.0; F[s] = 0.0; WT[s] = 0.0; WC[s] = 0.0;
      }
    }
    sizeOverride = sizeAcc;
  }

  // Pick the dominant blob at this pixel (largest field) ...
  const float SHOW_THRESH = 0.55;
  int dom = -1;
  float maxF = 0.0;
  for (int s = 0; s < 4; s++) {
    if (F[s] > maxF) { maxF = F[s]; dom = s; }
  }
  // ... then override by z among blobs that are "visible"
  int frontDom = -1;
  float frontZ = -1.0;
  int otherDom = -1;
  float otherF = 0.0;
  for (int s = 0; s < 4; s++) {
    if (ids[s] >= 1.0 && F[s] > SHOW_THRESH) {
      float z = uBlobZ[int(ids[s])];
      if (z > frontZ) {
        otherDom = frontDom; otherF = (frontDom >= 0) ? F[frontDom] : 0.0;
        frontDom = s; frontZ = z;
      } else if (F[s] > otherF) {
        otherDom = s; otherF = F[s];
      }
    }
  }
  if (frontDom >= 0) dom = frontDom;
  float field  = dom >= 0 ? F[dom]  : 0.0;
  float domWT  = dom >= 0 ? WT[dom] : 0.0;
  float domWC  = dom >= 0 ? WC[dom] : 0.0;
  int   domGid = dom >= 0 ? int(ids[dom]) : 0;
  float temp = field > 0.001 ? (domWT / field) : 0.18;
  // Average compression for the dominant blob at this pixel
  float compAvg = field > 0.001 ? (domWC / field) : 0.0;
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
  float horiz = 1.0 - smoothstep(0.0, halfW * 0.95, abs(sp.x - cx));
  bulb *= mix(0.35, 1.0, horiz);

  vec3 warmTint = mix(uCold, uHot, 0.65) * 1.2;
  // The liquid is a coloured medium: light scattered in it comes out as
  // lamp light filtered by the liquid's colour (not lamp light laid on top,
  // which greys the colour out). Lamp ≈ incandescent, ~2700 K.
  vec3 lampLight = LIN(vec3(1.0, 0.86, 0.66));
  vec3 liqTint = uBg / max(max(uBg.r, uBg.g), max(uBg.b, 1e-4));
  vec3 scatterCol = lampLight * liqTint;
  vec3 fluidBg = mix(uBg, uBg * 1.3 + warmTint * 0.06, bottomT);
  fluidBg += scatterCol * bulb * 0.45;
  fluidBg *= mix(0.85, 1.0, smoothstep(0.0, 1.0, t));

  // Permanent opaque pool block
  float poolBlockMask = smoothstep(0.89, 0.90, t);
  vec3 poolBlockColor = mix(uCold, uHot, 0.45) * mix(0.55, 1.15, pow(tc, 1.4));
  fluidBg = mix(fluidBg, poolBlockColor, poolBlockMask);

  // -------- Volumetric light from below ---------
  // The texture holds per-column light intensity: bright where light from
  // the pool passes through unobstructed, dark where wax blocks it.
  // Horizontally blurred on CPU with mass-dependent kernel for refraction.
  float wallIrr = 0.0;             // light striking the glass on this side
  if (uNumCols > 0 && uNumRows > 0) {
    float colU = sp.x / uSim.x;
    float colV = 1.0 - t;   // texV: 0=top of lamp, 1=bottom
    vec4 vol = texture(uColMass, vec2(colU, colV));
    // The CPU grid's per-column shadowing (light walking up from a fixed
    // row, dimmed by the wax mass above it) is only a fallback: with the
    // GPU trace, blob shadows are in its caustic difference, and this
    // model also shadowed the liquid above a thick pool with the pool's
    // own wax — a dark fan with a flat bottom.
    float light = uCausGpu > 0.5 ? 1.0 : vol.r;

    // God ray: additive warm glow scaled by bulb glow slider
    vec3 rayColor = mix(uHot, LIN(vec3(1.0, 0.97, 0.90)), 0.3);
    // Envelope: strong in lower-mid bulb, gentler fade toward top
    float envelope = smoothstep(0.93, 0.65, t) * smoothstep(0.02, 0.15, t);
    float glowScale = uGlow / 0.38;  // normalized so default glow=0.55 → 1.0
    // Beer–Lambert: bulb light is absorbed/scattered on its way up, so the
    // liquid fades with height above the pool (≈ e^-2.2 per lamp height).
    float beer = exp(-max(0.88 - t, 0.0) * 2.2);
    fluidBg += scatterCol * light * envelope * 0.55 * glowScale * beer;

    // Caustics: light redirected by wax lenses and wall reflections, as a
    // fraction of clear-lamp light (+ concentrated, − pulled away).
    if (uCaustics > 0.5) {
      // down to the light source, just under the pool's resting surface
      float causticEnv = smoothstep(0.02, 0.10, t) * (1.0 - smoothstep(0.895, 0.905, t));
      float excess = vol.g;
      wallIrr = vol.b;
      if (uCausGpu > 0.5) {
        vec2 cz = mix(texture(uCaus, vec2(colU, colV)).rg, texture(uCaus2, vec2(colU, colV)).rg, uCausMix);
        excess = cz.r;
        // Glass-edge light (light/caustic-edge.frag), both sides blended across the
        // middle: the limb term is small there but not zero, and a hard
        // switch would show as a seam.
        float eL = texture(uCausEdge, vec2(0.25, colV)).r, eR = texture(uCausEdge, vec2(0.75, colV)).r;
        wallIrr = mix(eL, eR, smoothstep(-0.5 * halfW, 0.5 * halfW, simPos.x - cx));
      }
      fluidBg += scatterCol * clamp(excess, -0.6, 2.0) * 0.3 * causticEnv * beer;
    }

    // Darken where light is blocked (1 - light = shadow)
    float shadow = (1.0 - light) * envelope * 0.45 * glowScale;
    fluidBg *= 1.0 - shadow;
  }

  // -------- Wax shading from temperature ---------
  float tempN = clamp((temp - 0.18) / 0.85, 0.0, 1.0);
  vec3 waxColor = mix(uCold, uHot, smoothstep(0.0, 1.0, tempN));
  float lightFromBelow = mix(0.55, 1.15, pow(tc, 1.4));
  waxColor *= lightFromBelow;
  waxColor = pow(waxColor, vec3(0.95));

  // -------- Compression: boost existing color ---------
  // Under pressure, brighten and saturate the current wax color
  // rather than injecting a foreign accent. This looks natural
  // at any temperature — hot wax glows hotter, cool wax gets richer.
  float compLuma = dot(waxColor, vec3(0.299, 0.587, 0.114));
  // Boost: brighten by up to 40% and increase saturation
  waxColor = mix(waxColor, waxColor * 1.4 + (waxColor - vec3(compLuma)) * 0.5, compIntensity * 0.6);

  // -------- Physically-based spherical refractivity ---------
  float blobSz = clamp(uBlobSize[domGid], 0.0, 1.0);
  if (dom == poolSlot && sizeOverride >= 0.0) blobSz = clamp(sizeOverride, 0.0, 1.0);

  float threshold = 0.55;
  // The blob's surface: a crisp edge, antialiased over one screen pixel
  // (the field's own gradient sets the width), at the field level the
  // old soft edge was centred on. Wax is a surface, not a haze.
  const float EDGE = 0.45;
  float fw = max(fwidth(field), 1e-4);
  float alpha = clamp((field - EDGE) / fw + 0.5, 0.0, 1.0);

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
  vec3 specTint = mix(LIN(vec3(1.0, 0.95, 0.85)), waxColor * 2.0 + LIN(vec3(0.3)), compIntensity * 0.6);
  waxColor += specTint * highlight;

  // ---- Opacity: absorption through the wax, plus pressure ----
  alpha *= waxOpacity(blobSz, field);
  alpha *= max(1.0 - compIntensity * 0.65, 0.40);  // clip at 60% transparency

  // Inner core detail
  float coreAlpha = smoothstep(threshold + 0.05, threshold + 0.55, field);
  vec3 inner = mix(waxColor, waxColor * 1.12 + uHot * 0.08 * tempN, coreAlpha);

  // Overlap layering: front blob over back blob
  if (fuseBackW > 0.0) {
    float bT = clamp((fuseBackT - 0.18) / 0.85, 0.0, 1.0);
    vec3 bWax = mix(uCold, uHot, smoothstep(0.0, 1.0, bT)) * lightFromBelow;
    float bA = clamp((fuseBackF - EDGE) / fw + 0.5, 0.0, 1.0) * 0.90;
    fluidBg = mix(fluidBg, bWax, bA * fuseBackW);
  }
  if (otherDom >= 0) {
    float otherFieldRaw = F[otherDom];
    float otherTemp = otherFieldRaw > 0.001 ? (WT[otherDom] / otherFieldRaw) : 0.18;
    float otherTempN = clamp((otherTemp - 0.18) / 0.85, 0.0, 1.0);
    vec3 otherWax = mix(uCold, uHot, smoothstep(0.0, 1.0, otherTempN)) * lightFromBelow;
    float otherAlpha = clamp((otherFieldRaw - EDGE) / fw + 0.5, 0.0, 1.0);
    otherAlpha *= waxOpacity(clamp(uBlobSize[int(ids[otherDom])], 0.0, 1.0), otherFieldRaw);
    fluidBg = mix(fluidBg, otherWax, otherAlpha);
  }

  // Mix wax over fluid bg
  vec3 inside = mix(fluidBg, inner, alpha);
  // Fresnel at the curved glass: toward the silhouette less of the interior
  // gets through and more of the (dim) room is reflected.
  inside = inside * lensT + lensF * (LIN(vec3(0.035, 0.028, 0.05)) + uHot * 0.04 * uGlow);

  // -------- Bottle frame ---------
  // Top cap: a tapered metal cone sitting on the glass (half-width
  // 0.15 → 0.265 of the lamp's width from its top to the glass), shaded
  // like a metal cylinder — darker toward its sides, a soft highlight
  // band — with a darker lip where it meets the glass. Crisp,
  // antialiased edges; the wall shows around it.
  const float CAP_T = 0.055;
  float capU = clamp(t / CAP_T, 0.0, 1.0);
  float capHW = uSim.x * mix(0.15, 0.265, capU);
  float capX = (simPos.x - cx) / capHW;                     // −1 … 1 across the cone
  float capCov = clamp((capHW - distFromCenter) / max(fwidth(distFromCenter), 1e-3) + 0.5, 0.0, 1.0)
               * clamp(t / max(fwidth(t), 1e-5) + 0.5, 0.0, 1.0)
               * clamp((CAP_T - t) / max(fwidth(t), 1e-5) + 0.5, 0.0, 1.0);
  float capShade = 0.35 + 0.65 * sqrt(max(1.0 - capX * capX, 0.0));
  float capSpec = pow(max(1.0 - abs(capX + 0.35) * 2.5, 0.0), 2.0);
  vec3 topCapCol = LIN(vec3(0.13, 0.11, 0.16)) * capShade + LIN(vec3(0.30, 0.27, 0.34)) * capSpec;
  topCapCol *= mix(1.0, 0.55, smoothstep(0.85, 1.0, capU));   // lip

  vec3 botCapCol = mix(LIN(vec3(0.07, 0.04, 0.09)), LIN(vec3(0.16, 0.10, 0.13)), smoothstep(0.95, 1.00, t));
  float baseGlow = smoothstep(0.99, 0.95, t) * uGlow;
  botCapCol += uHot * baseGlow * 0.4;
  botCapCol += uCold * baseGlow * 0.18;
  float slit = smoothstep(0.948, 0.955, t) * (1.0 - smoothstep(0.955, 0.962, t));
  botCapCol += uHot * slit * uGlow * 1.4;

  // Wall behind the lamp: matte beige paint (rough, so it scatters light
  // evenly — Lambertian), lit by the lamp's warm bulb light after its trip
  // through the lamp, plus a little room light so the wall reads as a wall.
  //   liquid path: bulb light × the dyed liquid's transmittance. The
  //     liquid's colour (uBg) is how it looks, saturated by depth; the light
  //     crosses only a few centimetres of it, so it keeps most of its
  //     warmth: transmittance = colour^k, k ≪ 1 (Beer–Lambert, thin path).
  //   wax path: bulb light × the wax's (much weaker) tint.
  // All in linear light, multiplied — tinting by the full liquid colour
  // made the wall a saturated magenta.
  vec3 frameOut = vec3(0.0);
  if (uWall > 0.5) {
    vec2 wuv = vec2((simPos.x + uViewM) / viewSize.x, 1.0 - (simPos.y + uViewT) / viewSize.y);
    vec2 bd = mix(texture(uBackdrop, wuv).rg, texture(uBackdrop2, wuv).rg, uWallMix);
    vec3 bulbL = LIN(vec3(1.0, 0.84, 0.62));                 // warm incandescent, ~3000 K
    vec3 lTint = uBg / max(max(uBg.r, uBg.g), max(uBg.b, 1e-4));
    vec3 liqT = pow(max(lTint, vec3(1e-3)), vec3(0.22));
    vec3 wTint = mix(uCold, uHot, 0.6);
    vec3 waxT = pow(max(wTint / max(max(wTint.r, wTint.g), max(wTint.b, 1e-4)), vec3(1e-3)), vec3(0.3));
    vec3 WALL_ALBEDO = LIN(vec3(0.84, 0.78, 0.68));          // beige paint
    vec3 room = LIN(vec3(1.0, 0.92, 0.80)) * 0.012;          // dim room light
    // bd: light beyond the lamp's default state, or all of it (uFull)
    vec3 E = 0.32 * bulbL * (max(bd.r, 0.0) * liqT + max(bd.g, 0.0) * waxT);
    frameOut = WALL_ALBEDO * (E + room);
  }

  vec3 col;
  bool inCapX = distFromCenter < uSim.x * 0.5;   // the base is the lamp's width
  if (t > 0.95 && t <= 1.0 && inCapX) {
    col = botCapCol;
  } else {
    col = mix(frameOut, inside, insideGlass);
    // Glass edge lit by the light striking it. Seen edge-on, a shell of
    // thickness tau has line-of-sight length
    //   L(b) = 2(√(R² − b²) − √((R − tau)² − b²))
    // through glass at apparent offset b, peaking at the inner surface:
    // limb brightening, which is why a lit glass shows bright rim lines.
    if (wallIrr > 0.0) {
      float tau = 3.0;
      float b = distFromCenter, R = halfW;
      float Lo = sqrt(max(R * R - b * b, 0.0));
      float Li = sqrt(max((R - tau) * (R - tau) - b * b, 0.0));
      float L = 2.0 * (Lo - Li) / (2.0 * sqrt(max(2.0 * R * tau - tau * tau, 1e-3)));
      vec3 edgeCol = mix(uHot, LIN(vec3(1.0, 0.97, 0.90)), 0.3);
      // soft response: irradiance is ~1 on average, a few × at hot spots
      col += edgeCol * (1.0 - exp(-0.7 * wallIrr)) * L * 0.32 * step(b, R);
    }
  }

  col = mix(col, topCapCol, capCov);

  // Soft outer glow
  float outerGlow = smoothstep(0.96, 0.55, t) * uGlow * 0.18;
  outerGlow *= smoothstep(uSim.x * 0.9, 0.0, distFromCenter);
  // (most of the glow around the lamp now comes from bloom)
  col += uHot * outerGlow * (1.0 - insideGlass) * 0.06;

  // Output is linear, scene-referred light; the composite pass tone-maps.
  if (uDirectOut > 0.5) col = finishColor(col, pix, uRes, uTime);
  fragColor = vec4(col, 1.0);
}
