// ============================================================
//  Lava-lamp WebGL shaders — separated for modularity
//  Includes physically-based paraffin wax refractivity (n ≈ 1.43)
// ============================================================

const VERTEX_SHADER = `#version 300 es
in vec2 a_pos;
void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }`;

// ---------------- HDR pipeline ----------------
// The scene is shaded in linear light ("scene-referred": values above 1
// are fine, e.g. the glowing pool). Display happens in one place: exposure
// → filmic curve → sRGB. The curve is the ACES fit (Narkowicz 2015), with
// a toe that deepens darks and a shoulder that rolls off highlights. It is
// applied to the brightest channel and the colour scaled to match, so
// saturated wax stays saturated instead of bleaching toward white.
const FINISH_GLSL = `
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
`;

// Fullscreen pass vertex shader with UVs, for the post chain.
const POST_VS = `#version 300 es
in vec2 a_pos;
out vec2 vUv;
void main(){ vUv = a_pos * 0.5 + 0.5; gl_Position = vec4(a_pos, 0.0, 1.0); }`;

// Bloom: optical glare from bright sources (lens / eye scattering). Mobile-
// friendly "dual filter" pyramid (Bjørge, SIGGRAPH 2015): each level is
// half the size of the last; the first downsample keeps only light above a
// soft threshold.
const BLOOM_DOWN_FS = `#version 300 es
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
}`;

const BLOOM_UP_FS = `#version 300 es
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
}`;

const COMPOSITE_FS = `#version 300 es
precision highp float;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomStrength;   // k: fraction of light scattered into the halo
uniform float uBloomNorm;       // 1 / number of pyramid levels summed
uniform vec2 uRes;
uniform float uTime;
in vec2 vUv;
out vec4 o;
${FINISH_GLSL}
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
}`;

const BOTTLE_GLSL = `
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
`;

function fragmentShaderSource() {
  return `#version 300 es
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
${FINISH_GLSL}
// Colours below are authored in sRGB; all shading is in linear light.
#define LIN(c) pow(c, vec3(2.2))
uniform vec4  uGhostMix;
uniform sampler2D uColMass;   // 2D mass grid (NUM_COLS × NUM_ROWS), normalized 0-1
uniform sampler2D uBackdrop;  // wall behind the lamp, previous trace: R = liquid light, G = wax light
uniform sampler2D uBackdrop2; // … newest trace
uniform float uWallMix;       // 0 → previous, 1 → newest
uniform float uWall;          // 1 = draw the lit wall
// GPU caustics (see LIGHT_TRACE_FS): light scattered in the liquid over the
// lamp, R = excess over the lamp without wax lenses, G = that lamp's light;
// previous and newest trace
uniform sampler2D uCaus;
uniform sampler2D uCaus2;
uniform float uCausMix;
uniform float uCausGpu;       // 1 = use uCaus; 0 = the CPU grid's G/B channels
uniform sampler2D uCausEdge;  // light at the glass per row: x = 0 left, 1 right (CAUS_EDGE_FS)
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

${BOTTLE_GLSL}
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
      float causticEnv = smoothstep(0.02, 0.10, t) * (1.0 - smoothstep(0.86, 0.89, t));
      float excess = vol.g;
      wallIrr = vol.b;
      if (uCausGpu > 0.5) {
        vec2 cz = mix(texture(uCaus, vec2(colU, colV)).rg, texture(uCaus2, vec2(colU, colV)).rg, uCausMix);
        excess = cz.r;
        // Glass-edge light (CAUS_EDGE_FS), both sides blended across the
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
  vec3 topCapCol = LIN(vec3(0.06, 0.05, 0.10));
  float neckBand = smoothstep(0.030, 0.035, t) * (1.0 - smoothstep(0.045, 0.050, t));
  topCapCol += LIN(vec3(0.10, 0.08, 0.14)) * neckBand;
  float capLight = smoothstep(0.4, 0.0, distFromCenter / (uSim.x * 0.3));
  topCapCol *= mix(1.0, 1.5, capLight * (1.0 - smoothstep(0.0, 0.05, t)));

  vec3 botCapCol = mix(LIN(vec3(0.07, 0.04, 0.09)), LIN(vec3(0.16, 0.10, 0.13)), smoothstep(0.95, 1.00, t));
  float baseGlow = smoothstep(0.99, 0.95, t) * uGlow;
  botCapCol += uHot * baseGlow * 0.4;
  botCapCol += uCold * baseGlow * 0.18;
  float slit = smoothstep(0.948, 0.955, t) * (1.0 - smoothstep(0.955, 0.962, t));
  botCapCol += uHot * slit * uGlow * 1.4;

  // Wall behind the lamp: a neutral diffuse surface that only scatters the
  // light the lamp throws onto it (traced on the CPU), so its colour is the
  // lamp's light — liquid-filtered lamp light and the wax's own glow.
  vec3 frameOut = vec3(0.0);
  if (uWall > 0.5) {
    vec2 wuv = vec2((simPos.x + uViewM) / viewSize.x, 1.0 - (simPos.y + uViewT) / viewSize.y);
    vec2 bd = mix(texture(uBackdrop, wuv).rg, texture(uBackdrop2, wuv).rg, uWallMix);
    vec3 lampL = LIN(vec3(1.0, 0.86, 0.66));
    vec3 lTint = uBg / max(max(uBg.r, uBg.g), max(uBg.b, 1e-4));
    vec3 waxL = mix(uCold, uHot, 0.6);
    const float WALL_ALBEDO = 0.6;           // matte, neutral
    // bd holds light beyond the lamp's default state (see traceWall3D),
    // so the wall is dark except for moving caustic patches.
    // Clamp each light component before combining: extra liquid-filtered
    // light shows purple, extra light through / from wax shows amber, and
    // light taken away just leaves the wall dark (clamping the mixed colour
    // per channel would leave odd hues, e.g. green from amber − purple).
    frameOut = WALL_ALBEDO * 0.32 * (max(bd.r, 0.0) * lampL * lTint + max(bd.g, 0.0) * waxL);
  }

  vec3 col;
  bool inCapX = distFromCenter < uSim.x * 0.5;   // cap / base are the lamp's width
  if (t >= 0.0 && t < 0.05 && inCapX) {
    col = topCapCol;
  } else if (t > 0.95 && t <= 1.0 && inCapX) {
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

  // Soft outer glow
  float outerGlow = smoothstep(0.96, 0.55, t) * uGlow * 0.18;
  outerGlow *= smoothstep(uSim.x * 0.9, 0.0, distFromCenter);
  // (most of the glow around the lamp now comes from bloom)
  col += uHot * outerGlow * (1.0 - insideGlass) * 0.06;

  // Output is linear, scene-referred light; the composite pass tone-maps.
  if (uDirectOut > 0.5) col = finishColor(col, pix, uRes, uTime);
  fragColor = vec4(col, 1.0);
}
`;
}

// UMD export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { VERTEX_SHADER, fragmentShaderSource, POST_VS,
                     BLOOM_DOWN_FS, BLOOM_UP_FS, COMPOSITE_FS };
}

// ============================================================
//  Lamp light on the GPU: caustics in the liquid and on the wall behind
//
//  One ray tracer (LIGHT_TRACE_FS) follows the lamp's light:
//   • it starts at the bulb — one diffuse (Lambertian) source at the
//     centre of the base — and passes up through the pool wax, dimmed
//     along the way, to the pool's top surface: a height field built each
//     trace from the pool particles, bumps and necks included, which
//     refracts it (wax → liquid, Fresnel, total internal reflection);
//   • refracts into and out of the wax blobs (ellipsoids),
//   • and leaves through the curved glass (Fresnel, TIR) to the wall.
//  Mode 0 writes where each ray lands on the wall; mode 1 writes the
//  vertices of its path (the pool, each blob surface and glass hit), and
//  CAUS_CROSS_FS finds from those where it crosses each of a stack of
//  horizontal planes in the liquid.
//
//  Wall:   WALL_SPLAT_* draws each tube of four neighbouring rays as two
//          triangles carrying the tube's power over its area (converging
//          light is bright: caustics), at 2× the wall grid's resolution;
//          WALL_UPDATE_FS filters it down, then keeps the running-average
//          baseline and the difference that is drawn.
//  Liquid: CAUS_TUBE_* does the same between successive planes: a tube's
//          light scattered out in that slab (power × path length) is spread
//          over the band it spans in the view — as traced and without the
//          wax lenses and pool bumps (the reference), for the tubes a blob
//          or bump touched (the rest cancel). The full reference changes
//          slowly and is rebuilt every few updates. CAUS_UPDATE_FS filters
//          and normalises.
// ============================================================

const LIGHT_TRACE_FS = `#version 300 es
precision highp float;
uniform vec2  uSim;
uniform float uViewM, uViewT;
uniform float uWallD;       // wall plane behind the lamp axis (sim px)
uniform float uGlow;
uniform float uRayScale;    // power per ray (keeps totals independent of ray count)
uniform ivec3 uRayDim;      // rays: azimuth × polar × sources
uniform float uAzSpan;      // azimuths covered: π (toward the wall) or 2π
uniform float uYSrc;        // the bulb: centre of the base (sim px)
uniform float uSrcR;        // … and its radius
uniform float uMuPool;      // attenuation in the pool wax (per sim px)
uniform int   uNB;
uniform vec4  uWB0[40];     // blob centre (x from the axis, y, z), lens strength
uniform vec4  uWB1[40];     // blob semi-axes
uniform highp sampler2D uPool;  // row 0: pool base top y vs radius
uniform int   uPoolN;
uniform float uLensOn;      // 0 = reference: no blobs, no bumps
uniform int   uMode;        // 0 = wall landing, 1 = path vertices
uniform float uPoolYMin;
// Refinement (mode 0, uSub > 0): for each tube of the main wall trace that
// touches a blob or is broken (see tubeRefined), a (uSub+1)² grid of rays
// across it; other tubes output nothing.
uniform int   uSub;
uniform highp sampler2D uRays;  // the main wall trace
uniform vec2  uJitter;      // sub-ray offset of the whole ray grid (TAA), in ray steps
uniform float uMaxSpan;    // highest point of the pool surface (smallest y)
uniform int   uRowOff;
// mode 0: wall x, wall y (view px), power, signature + wax share (o only)
// mode 1: path vertices 4·chunk … 4·chunk+3 (o, o1, o2, o3); vertex 7's
//         slot holds (1 if the path met a blob, its signature, …) instead
layout(location = 0) out vec4 o;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
${BOTTLE_GLSL}
const float PI = 3.14159265;
const float ETA_G = 1.34;             // liquid → air
const float NW = 1.43 / 1.34;         // wax / liquid
float Rof(float y) { return bottleHalfFrac(clamp(y / uSim.y, 0.0, 1.0)) * uSim.x; }
float Rsl(float y) { return 0.5 * (Rof(y + 1.0) - Rof(y - 1.0)); }
// Smooth (cubic B-spline) lookup, u in texels. Caustics follow the
// surface's curvature: linear interpolation makes the slope jump at every
// texel, and Catmull-Rom the curvature, each drawing a line on the wall
// per texel. The B-spline's curvature is continuous.
float poolAt(int row, int i) { return texelFetch(uPool, ivec2(clamp(i, 0, uPoolN - 1), row), 0).r; }
float poolRow(int row, float u) {
  float x = clamp(u, 0.0, float(uPoolN - 1));
  int i = int(floor(x)); float f = x - float(i), g = 1.0 - f;
  float w0 = g * g * g / 6.0, w1 = (4.0 - 6.0 * f * f + 3.0 * f * f * f) / 6.0;
  float w2 = (4.0 - 6.0 * g * g + 3.0 * g * g * g) / 6.0, w3 = f * f * f / 6.0;
  return w0 * poolAt(row, i - 1) + w1 * poolAt(row, i) + w2 * poolAt(row, i + 1) + w3 * poolAt(row, i + 2);
}
// y of the pool's base surface above (x, z) (x from the axis); its bumps
// and columns are ellipsoids in the blob list
float poolTop(vec2 xz) {
  return poolRow(0, length(xz) / (0.5 * uSim.x) * float(uPoolN - 1));
}
bool hitE(int b, vec3 p, vec3 d, out float h0, out float h1) {
  vec3 a = uWB1[b].xyz;
  vec3 q = (p - uWB0[b].xyz) / a, e = d / a;
  float A = dot(e, e), B = dot(q, e), C = dot(q, q) - 1.0;
  float disc = B * B - A * C;
  h0 = 0.0; h1 = 0.0;
  if (disc <= 0.0) return false;
  float s = sqrt(disc);
  h0 = (-B - s) / A; h1 = (-B + s) / A;
  return true;
}
vec3 eNormal(int b, vec3 p) {
  vec3 a = uWB1[b].xyz;
  return normalize((p - uWB0[b].xyz) / (a * a));
}
float fresnelR(float ci, float n1, float n2) {
  float si = sqrt(max(0.0, 1.0 - ci * ci));
  float st = (n1 / n2) * si;
  if (st >= 1.0) return 1.0;
  float ct = sqrt(1.0 - st * st);
  float rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
  float rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
  return 0.5 * (rs * rs + rp * rp);
}
// path recording (mode 1): vertices kBase … kBase+3 of the path, each with
// the power that leaves it
int vc = 0, kBase = -100;
vec4 pv[4];
vec4 lastV = vec4(0.0);
bool hitBlob = false;
// The path's signature: which blobs it entered (first three), plus its
// internal reflections. Neighbouring rays with different signatures are
// not one bundle of light — e.g. one grazes a blob's rim and is thrown
// wide while its neighbour misses the blob — and the tube between them
// must not be drawn: stretched across, it becomes a long bright spike.
float sig = 0.0;
int nEnt = 0;
void rec(vec3 p, float P) {
  vec4 v = vec4(p.x + 0.5 * uSim.x, p.y, p.z, P);
  int i = vc - kBase;
  if (i >= 0 && i < 4) pv[i] = v;
  lastV = v; vc++;
}
// this chunk's vertices are all in, and the blob flag can't change: stop
bool enough() { return uMode == 1 && vc >= kBase + 4 && (kBase == 0 || hitBlob); }
// A main-trace tube is drawn as is (WALL_SPLAT_VS) only if its four rays
// all reached the wall, landed close together, took the same path and met
// no blob. The rest — blob images, their edges, folds — are refined.
bool tubeRefined(vec4 r00, vec4 r01, vec4 r10, vec4 r11) {
  if (min(min(r00.z, r01.z), min(r10.z, r11.z)) <= 0.0) return true;
  if (max(distance(r00.xy, r11.xy), distance(r01.xy, r10.xy)) > uMaxSpan) return true;
  float s0 = floor(r00.w);
  if (floor(r01.w) != s0 || floor(r10.w) != s0 || floor(r11.w) != s0) return true;
  return mod(s0, 68921.0) > 0.0;                        // entered a blob
}

void main() {
  ivec2 id = ivec2(gl_FragCoord.xy);
  id.y -= uRowOff;
  int ia = id.x;
  if (uMode == 1) { ia = id.x / 2; kBase = 4 * (id.x - ia * 2); }
  for (int i = 0; i < 4; i++) pv[i] = vec4(0.0);
  int ip = id.y % uRayDim.y, src = id.y / uRayDim.y;
  float fia = float(ia), fip = float(ip);                 // ray (fractional when refining)
  float subW = 1.0;                                       // share of a main ray's power
  if (uSub > 0) {
    int S1 = uSub + 1;
    int a = id.x / S1, i = id.x - a * S1, t = id.y / S1, j = id.y - t * S1;
    int p = t % (uRayDim.y - 1);
    src = t / (uRayDim.y - 1);
    int row = src * uRayDim.y + p;
    if (!tubeRefined(texelFetch(uRays, ivec2(a, row), 0), texelFetch(uRays, ivec2(a + 1, row), 0),
                     texelFetch(uRays, ivec2(a, row + 1), 0), texelFetch(uRays, ivec2(a + 1, row + 1), 0))) {
      o = vec4(0.0); return;
    }
    fia = float(a) + float(i) / float(uSub);
    fip = float(p) + float(j) / float(uSub);
    subW = 1.0 / float(uSub * uSub);
  }
  fia += uJitter.x; fip += uJitter.y;
  // The bulb: one diffuse (Lambertian) source at the centre of the base.
  // (uRayDim.z > 1 would add points on a ring of radius uSrcR — but each
  // point casts its own image of every blob, so one blob would show as
  // several refractions on the wall.)
  float cMin = cos(1.35);
  float c = 1.0 - (1.0 - cMin) * (fip + 0.5) / float(uRayDim.y);
  float sph = sqrt(1.0 - c * c);
  float al = PI + uAzSpan * (fia + 0.5) / float(uRayDim.x);        // from −x round through the back
  vec3 d = vec3(sph * cos(al), -c, sph * sin(al));
  vec2 s0 = vec2(0.0);
  if (src > 0) { float an = float(src - 1) * 2.0 * PI / float(uRayDim.z - 1) + 0.3; s0 = uSrcR * vec2(cos(an), sin(an)); }
  vec3 p = vec3(s0.x, uYSrc, s0.y);
  float P = c * uGlow * uRayScale * subW / float(uRayDim.z);         // Lambertian
  float P0 = P;                         // cut-offs are relative to this
  float h0, h1;

  // ---- up through the pool wax to its surface ----
  // The wax dims the light along its path; its top surface refracts it
  // (wax → liquid), with Fresnel loss and total internal reflection.
  bool out_ = p.y < poolTop(p.xz);      // no wax above the bulb
  float poolLen = 0.0;
  for (int bounce = 0; bounce < 3 && !out_ && P > 0.0; bounce++) {
    float t = 0.0, tHit = -1.0;
    for (int k = 0; k < 96; k++) {
      float tn = t + 3.0;
      vec3 pn = p + d * tn;
      if (pn.y > 0.995 * uSim.y || dot(pn.xz, pn.xz) > Rof(pn.y) * Rof(pn.y)) { P = 0.0; break; }
      if (pn.y < poolTop(pn.xz)) {
        float lo = t, hi = tn;
        for (int j = 0; j < 8; j++) {
          float m = 0.5 * (lo + hi); vec3 pm = p + d * m;
          if (pm.y < poolTop(pm.xz)) hi = m; else lo = m;
        }
        tHit = hi; break;
      }
      t = tn;
    }
    if (tHit < 0.0) { P = 0.0; break; }
    p += d * tHit; poolLen += tHit;
    float e = 3.0;
    float gx = (poolTop(p.xz + vec2(e, 0.0)) - poolTop(p.xz - vec2(e, 0.0))) / (2.0 * e);
    float gz = (poolTop(p.xz + vec2(0.0, e)) - poolTop(p.xz - vec2(0.0, e))) / (2.0 * e);
    vec3 nUp = normalize(vec3(gx, -1.0, gz));          // up, into the liquid
    float ci = dot(d, nUp);
    vec3 r = refract(d, -nUp, 1.43 / 1.34);
    if (ci > 0.0 && dot(r, r) > 0.0) {
      P *= 1.0 - fresnelR(ci, 1.43, 1.34);
      d = r; p += d * 0.05; out_ = true;
    } else {
      d -= 2.0 * ci * nUp; p -= nUp * 0.1; P *= 0.9;   // reflected back into the pool
      sig += 275684.0;
    }
  }
  P *= exp(-uMuPool * poolLen);
  if (!out_) P = 0.0;
  rec(p, P);
  if (P <= 0.0) {
    if (uMode == 1) {
      vec4 w[4];
      for (int i = 0; i < 4; i++) w[i] = kBase + i < vc ? pv[i] : vec4(lastV.xyz, 0.0);
      if (kBase == 4) w[3] = vec4(hitBlob ? 1.0 : 0.0, sig, 0.0, 0.0);
      o = w[0]; o1 = w[1]; o2 = w[2]; o3 = w[3];
    } else o = vec4(0.0);
    return;
  }

  // ---- through the liquid, blobs and glass ----
  int inB = -1, refl = 0;
  bool exited = false;
  float waxLen = 0.0, liqLen = 0.0;
  int nb = uLensOn > 0.5 ? uNB : 0;
  for (int b = 0; b < 40; b++) {
    if (b >= nb) break;
    if (hitE(b, p, d, h0, h1) && h0 < 0.0 && h1 > 0.0) { inB = b; break; }
  }
  for (int ev = 0; ev < 10; ev++) {
    if (P <= 1e-3 * P0 || enough()) break;
    if (inB < 0) {
      float tB = 1e9; int bB = -1;
      for (int b = 0; b < 40; b++) {
        if (b >= nb) break;
        if (hitE(b, p, d, h0, h1) && h0 > 1e-3 && h0 < tB) { tB = h0; bB = b; }
      }
      // march to the glass (or the blob, whichever first)
      float t = 0.0, tExit = -1.0;
      for (int k = 0; k < 256; k++) {
        if (t >= tB) break;
        float tn = min(t + 12.0, tB);
        vec3 pn = p + d * tn;
        if (pn.y < 0.05 * uSim.y || (pn.y > uPoolYMin && pn.y > poolTop(pn.xz))) { P = 0.0; break; }   // top, or back into the pool
        float R = Rof(pn.y);
        if (dot(pn.xz, pn.xz) > R * R) {
          float lo = t, hi = tn;
          for (int j = 0; j < 10; j++) {
            float m = 0.5 * (lo + hi); vec3 pm = p + d * m; float Rm = Rof(pm.y);
            if (dot(pm.xz, pm.xz) > Rm * Rm) hi = m; else lo = m;
          }
          tExit = lo; break;
        }
        t = tn;
      }
      if (P == 0.0) { rec(p + d * t, 0.0); break; }
      if (tExit >= 0.0) {
        p += d * tExit; liqLen += tExit;
        float rho = max(length(p.xz), 1e-6);
        vec3 n = normalize(vec3(p.x / rho, -Rsl(p.y), p.z / rho));
        float ci = dot(d, n);
        if (ci > 0.0) {
          vec3 r = refract(d, -n, ETA_G);
          if (dot(r, r) > 0.0) {
            rec(p, 0.0);                                  // leaves the lamp
            P *= 1.0 - fresnelR(ci, ETA_G, 1.0);
            d = r; exited = true; break;
          }
        }
        // total internal reflection: back inside
        P *= 0.96;
        rec(p, P);
        refl++;
        sig += 68921.0;
        if (refl > 2) { P = 0.0; break; }
        d -= 2.0 * ci * n;
        p -= n * 0.5;
        continue;
      }
      // enter blob bB
      p += d * tB; liqLen += tB;
      vec3 n = eNormal(bB, p);
      float nwB = 1.0 + uWB0[bB].w * (NW - 1.0);   // faded refractive contrast
      // Fresnel: near the blob's rim (grazing) most light is reflected, not
      // transmitted — otherwise rim rays, bent hardest and spread thinnest,
      // fan out into faceted wedges on the wall
      P *= 1.0 - fresnelR(max(-dot(d, n), 0.0), 1.0, nwB);
      vec3 r = refract(d, n, 1.0 / nwB);
      if (dot(r, r) > 0.0) d = r;
      hitBlob = true;
      if (nEnt < 3) sig += float(bB + 1) * (nEnt == 0 ? 1.0 : nEnt == 1 ? 41.0 : 1681.0);
      nEnt++;
      rec(p, P);
      inB = bB;
    } else {
      // inside blob inB: to its far side
      float t1 = hitE(inB, p, d, h0, h1) ? max(h1, 0.0) : 0.0;
      p += d * t1;
      float st = uWB0[inB].w;
      waxLen += t1 * st;
      P *= exp(-0.004 * t1 * st);
      vec3 n = eNormal(inB, p);
      float nwO = 1.0 + st * (NW - 1.0);
      vec3 r = refract(d, -n, nwO);
      if (dot(r, r) > 0.0) { P *= 1.0 - fresnelR(max(dot(d, n), 0.0), nwO, 1.0); d = r; }
      else d -= 2.0 * dot(d, n) * n;                          // total internal reflection
      rec(p, P);
      p += d * 1e-2;
      inB = -1;
    }
  }
  if (uMode == 1) {
    // past the path's end: its last point, carrying nothing
    vec4 w[4];
    for (int i = 0; i < 4; i++) w[i] = kBase + i < vc ? pv[i] : vec4(lastV.xyz, 0.0);
    if (kBase == 4) w[3] = vec4(hitBlob ? 1.0 : 0.0, sig, 0.0, 0.0);
    o = w[0]; o1 = w[1]; o2 = w[2]; o3 = w[3];
    return;
  }
  if (!exited || P <= 1e-4 * P0 || d.z > -1e-4) { o = vec4(0.0); return; }
  P *= exp(-2.2 / uSim.y * liqLen);
  float sW = (-uWallD - p.z) / d.z;
  // w: signature, plus the wax share (< 1) in the fraction
  o = vec4(p.x + d.x * sW + 0.5 * uSim.x + uViewM, p.y + d.y * sW + uViewT, P, sig + min(0.999, waxLen / 20.0));
}`;

// Ray tubes → wall. No vertex attributes: gl_VertexID picks the quad (and
// which of its 6 triangle corners). A tube that is torn (a ray lost, or
// neighbours landing far apart across a TIR fold) collapses to nothing.
const WALL_SPLAT_VS = `#version 300 es
precision highp float;
uniform highp sampler2D uRays;
uniform ivec3 uRayDim;
uniform vec2  uView;        // view size (sim px)
uniform vec2  uCell;        // target texel size (sim px)
uniform float uMaxSpan;
out vec2 vE;                // irradiance per texel: liquid-path, wax-path
int src;
vec4 ray(int a, int p) { return texelFetch(uRays, ivec2(a, src * uRayDim.y + p), 0); }
float cross2(vec2 u, vec2 v) { return u.x * v.y - u.y * v.x; }
// the tube with corner rays (a, p) … (a+1, p+1): its irradiance (power
// over its area on the wall, in texels) split liquid / wax, or x < 0 if
// there is none (off the grid, a ray lost, or torn across a fold)
vec2 tube(int a, int p) {
  if (a < 0 || p < 0 || a >= uRayDim.x - 1 || p >= uRayDim.y - 1) return vec2(-1.0);
  vec4 r00 = ray(a, p), r01 = ray(a + 1, p), r10 = ray(a, p + 1), r11 = ray(a + 1, p + 1);
  float span = max(distance(r00.xy, r11.xy), distance(r01.xy, r10.xy));
  if (min(min(r00.z, r01.z), min(r10.z, r11.z)) <= 0.0 || span > uMaxSpan) return vec2(-1.0);
  // all four rays must have taken the same path (LIGHT_TRACE_FS: sig), and
  // not through a blob: those tubes are drawn refined (WALL_SUB_VS)
  float s0 = floor(r00.w);
  if (floor(r01.w) != s0 || floor(r10.w) != s0 || floor(r11.w) != s0) return vec2(-1.0);
  if (mod(s0, 68921.0) > 0.0) return vec2(-1.0);
  vec2 a0 = r00.xy / uCell, a1 = r01.xy / uCell, b0 = r10.xy / uCell, b1 = r11.xy / uCell;
  float area = 0.5 * abs(cross2(a1 - a0, b0 - a0)) + 0.5 * abs(cross2(b1 - a1, b0 - a1));
  float E = 0.25 * (r00.z + r01.z + r10.z + r11.z);
  float wf = 0.25 * (fract(r00.w) + fract(r01.w) + fract(r10.w) + fract(r11.w));
  float irr = E / max(area, 0.7);
  return irr * vec2(1.0 - wf, wf);
}
void main() {
  int q = gl_VertexID / 6, k = gl_VertexID - q * 6;
  int qa = uRayDim.x - 1, qp = uRayDim.y - 1;
  int ia = q % qa, t = q / qa, ip = t % qp;
  src = t / qp;
  if (tube(ia, ip).x < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vE = vec2(0.0); return; }
  // triangles (00, 01, 10) and (01, 11, 10); this corner's ray
  int kk = k < 3 ? k : k - 3;
  ivec2 c = k < 3 ? (kk == 0 ? ivec2(0, 0) : kk == 1 ? ivec2(1, 0) : ivec2(0, 1))
                  : (kk == 0 ? ivec2(1, 0) : kk == 1 ? ivec2(1, 1) : ivec2(0, 1));
  int va = ia + c.x, vp = ip + c.y;
  // Smooth shading: the ray carries the mean irradiance of the tubes
  // around it, interpolated across the triangle. Each tube drawn flat made
  // visible terraces where tubes are large on the wall.
  vec2 sum = vec2(0.0); float n = 0.0;
  for (int dy = -1; dy <= 0; dy++) for (int dx = -1; dx <= 0; dx++) {
    vec2 e = tube(va + dx, vp + dy);
    if (e.x >= 0.0) { sum += e; n += 1.0; }
  }
  vE = sum / max(n, 1.0);
  vec4 me = ray(va, vp);
  vec2 ndc = vec2(me.x / uView.x * 2.0 - 1.0, 1.0 - me.y / uView.y * 2.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}`;

const WALL_SPLAT_FS = `#version 300 es
precision highp float;
in vec2 vE;
out vec4 o;
void main() { o = vec4(vE, 0.0, 0.0); }`;

// Per trace: filter the 2×-resolution splat down to the wall grid (a tent
// ≈ 3 grid cells wide, from 3×3 bilinear taps that each average a 2×2
// block), normalise, then baseline = running average; drawn = light above
// it, lightly smoothed in time. Outputs: new baseline, new drawn wall.
const WALL_UPDATE_FS = `#version 300 es
precision highp float;
uniform highp sampler2D uIrr;   // 2× resolution
uniform highp sampler2D uAvg;
uniform highp sampler2D uPrev;
uniform float uInvRef, uAlpha, uSM, uInit;
layout(location = 0) out vec4 oAvg;
layout(location = 1) out vec4 oWall;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 texel = 1.0 / vec2(textureSize(uIrr, 0));
  vec2 c = (2.0 * vec2(p) + 1.0) * texel;            // centre of this cell's 2×2 block
  vec2 v = vec2(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      float w = float((2 - abs(i)) * (2 - abs(j)));
      v += w * texture(uIrr, c + 2.0 * vec2(i, j) * texel).rg;
    }
  }
  // 16 weight × 4 texels per tap (the 2×2 average), per grid cell (4 texels)
  v *= uInvRef * 4.0 / 16.0;
  vec2 a = uInit > 0.5 ? v : mix(texelFetch(uAvg, p, 0).rg, v, uAlpha);
  // (with TAA the ray grid is jittered every trace, so this history
  // averages several samplings of the light)
  vec2 s = uInit > 0.5 ? vec2(0.0) : mix(texelFetch(uPrev, p, 0).rg, v - a, uSM);
  oAvg = vec4(a, 0.0, 1.0);
  oWall = vec4(s, 0.0, 1.0);
}`;

// Path vertices (up to 7; vertex 0 is where the light leaves the pool) →
// where each path crosses each plane (planes go up from uPlaneY0 in steps
// of uPlaneDY), with the power it carries there. A plane below the path's
// start (inside the pool) gets the start, carrying ~0.
const CAUS_CROSS_FS = `#version 300 es
precision highp float;
uniform highp sampler2D uV0, uV1, uV2, uV3;   // vertex k: texture k % 4, column 2·ray + k / 4
uniform int   uPlanes;
uniform float uPlaneY0, uPlaneDY;
out vec4 o;
vec4 vert(int ray, int k, int row) {
  ivec2 c = ivec2(2 * ray + k / 4, row);
  int t = k - (k / 4) * 4;
  return t == 0 ? texelFetch(uV0, c, 0) : t == 1 ? texelFetch(uV1, c, 0)
       : t == 2 ? texelFetch(uV2, c, 0) : texelFetch(uV3, c, 0);
}
void main() {
  ivec2 id = ivec2(gl_FragCoord.xy);
  int ray = id.x / uPlanes, j = id.x - ray * uPlanes;
  float py = uPlaneY0 - float(j) * uPlaneDY;
  vec4 v0 = vert(ray, 0, id.y);
  if (v0.w <= 0.0) { o = vec4(v0.xyz, 0.0); return; }
  if (v0.y <= py) { o = vec4(v0.xyz, v0.w * 1e-4); return; }
  vec4 a = v0;
  for (int k = 1; k < 7; k++) {
    vec4 b = vert(ray, k, id.y);
    if ((a.y - py) * (b.y - py) <= 0.0 && a.y != b.y) {
      vec3 q = mix(a.xyz, b.xyz, (py - a.y) / (b.y - a.y));
      o = vec4(q, a.w);
      return;
    }
    if (a.w <= 0.0) break;
    a = b;
  }
  o = vec4(a.xyz, 0.0);           // left the lamp before this plane
}`;

// Plane crossings → light in the liquid. Each tube of four neighbouring
// rays, between plane j and j+1, scatters out power × path length (the
// liquid scatters a little everywhere); that is spread evenly over the band
// the tube spans in the view — its x range at each plane — so converging
// tubes are bright. Azimuths wrap (the rays go all the way round). Rows
// below uRowsLens are the traced rays (→ R), the rest the reference (→ G).
const CAUS_TUBE_VS = `#version 300 es
precision highp float;
uniform highp sampler2D uPath;
uniform int   uPathK;       // planes
uniform ivec3 uRayDim;      // azimuth × polar × sources
uniform int   uRowsLens;
uniform vec2  uSim;
uniform vec2  uTgt;         // target size (texels)
uniform float uMaxW;        // widest believable tube (sim px); wider = torn
uniform highp sampler2D uFlag;  // path texture 3: column 2·ray+1 holds (blob-hit flag, signature)
uniform int   uRefSrc;      // ≥ 0: draw only this source's reference tubes, all of them
out vec2 vW;
int ia, ib, row;
// the tube's four rays at planes j-1 … j+2 (index 0 … 3), fetched once
vec4 R[16];
void load(int j) {
  for (int m = 0; m < 4; m++) {
    int jj = clamp(j - 1 + m, 0, uPathK - 1);
    R[m * 4 + 0] = texelFetch(uPath, ivec2(ia * uPathK + jj, row), 0);
    R[m * 4 + 1] = texelFetch(uPath, ivec2(ib * uPathK + jj, row), 0);
    R[m * 4 + 2] = texelFetch(uPath, ivec2(ia * uPathK + jj, row + 1), 0);
    R[m * 4 + 3] = texelFetch(uPath, ivec2(ib * uPathK + jj, row + 1), 0);
  }
}
// x range, mean y, weakest and mean power at loaded plane m
void plane(int m, out float xm, out float xM, out float y, out float pmin, out float pm) {
  vec4 a = R[m * 4], b = R[m * 4 + 1], c = R[m * 4 + 2], d = R[m * 4 + 3];
  xm = min(min(a.x, b.x), min(c.x, d.x)); xM = max(max(a.x, b.x), max(c.x, d.x));
  y = 0.25 * (a.y + b.y + c.y + d.y);
  pmin = min(min(a.w, b.w), min(c.w, d.w));
  pm = 0.25 * (a.w + b.w + c.w + d.w);
}
// light per texel the tube puts in the slab between loaded planes m and
// m+1 (plane j-1+m), or −1
float density(int m, int j) {
  if (j < 0 || j >= uPathK - 1) return -1.0;
  float x0, X0, y0, q0, p0, x1, X1, y1, q1, p1;
  plane(m, x0, X0, y0, q0, p0);
  plane(m + 1, x1, X1, y1, q1, p1);
  if (min(q0, q1) <= 0.0 || X0 - x0 > uMaxW || X1 - x1 > uMaxW || abs(y1 - y0) < 0.05) return -1.0;
  vec2 s = uTgt / uSim;
  float L = 0.0;
  for (int r = 0; r < 4; r++) L += distance(R[m * 4 + r].xyz, R[m * 4 + 4 + r].xyz);
  float E = p0 * 0.25 * L * s.x;                                    // power × length (texels)
  float w0 = max((X0 - x0) * s.x, 1.0), w1 = max((X1 - x1) * s.x, 1.0);
  return E / max(0.5 * (w0 + w1) * abs(y1 - y0) * s.y, 0.5);
}
void main() {
  // One instance per tube slab, drawn as a 6-vertex strip: a tent across
  // the tube's x range — 0 at twice the range's half-width from its
  // centre, peaking at the centre (same light as a flat band over the
  // range). Flat bands left parallel stripes where neighbouring tubes
  // carry slightly different light; overlapping tents add up smoothly.
  // Vertices: 0/1 left (lower/upper), 2/3 centre, 4/5 right.
  int q = gl_InstanceID, k = gl_VertexID;
  int nsl = uPathK - 1;
  int j = q % nsl; q /= nsl;
  ia = q % uRayDim.x; q /= uRayDim.x;
  int ip = q % (uRayDim.y - 1); q /= (uRayDim.y - 1);
  // q = source + pass · sources, or (reference mode) that reference source
  if (uRefSrc >= 0) q = uRefSrc + uRayDim.z;
  row = q * uRayDim.y + ip;
  ib = (ia + 1) % uRayDim.x;
  // Tubes whose rays met no blob or pool bump are the same with and
  // without the lenses and cancel in the difference: skip them (in both
  // passes).
  if (uRefSrc < 0) {
    int rt = row < uRowsLens ? row : row - uRowsLens;
    if (texelFetch(uFlag, ivec2(2 * ia + 1, rt), 0).x + texelFetch(uFlag, ivec2(2 * ib + 1, rt), 0).x
      + texelFetch(uFlag, ivec2(2 * ia + 1, rt + 1), 0).x + texelFetch(uFlag, ivec2(2 * ib + 1, rt + 1), 0).x <= 0.0) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vW = vec2(0.0); return;
    }
  }
  // all four rays must have taken the same path (LIGHT_TRACE_FS: sig)
  float s0 = texelFetch(uFlag, ivec2(2 * ia + 1, row), 0).y;
  if (texelFetch(uFlag, ivec2(2 * ib + 1, row), 0).y != s0 || texelFetch(uFlag, ivec2(2 * ia + 1, row + 1), 0).y != s0
      || texelFetch(uFlag, ivec2(2 * ib + 1, row + 1), 0).y != s0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vW = vec2(0.0); return;
  }
  load(j);
  float dj = density(1, j);
  if (dj < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vW = vec2(0.0); return; }
  // The tube's light varies linearly through the slab, matching its
  // neighbouring slabs at the shared planes — a constant per slab would
  // make the field a staircase in height.
  bool top = (k & 1) == 1;
  float dn = top ? density(2, j + 1) : density(0, j - 1);
  float w = dn >= 0.0 ? 0.5 * (dn + dj) : dj;
  float x0, X0, y0, q0, p0, x1, X1, y1, q1, p1;
  plane(1, x0, X0, y0, q0, p0);
  plane(2, x1, X1, y1, q1, p1);
  vec2 s = uTgt / uSim;
  float hw0 = 0.5 * max((X0 - x0) * s.x, 1.0) / s.x, hw1 = 0.5 * max((X1 - x1) * s.x, 1.0) / s.x;
  float cx0 = 0.5 * (x0 + X0), cx1 = 0.5 * (x1 + X1);
  int col = k / 2;                                           // 0 left, 1 centre, 2 right
  float side = float(col - 1) * 2.0;                         // −2, 0, +2 half-widths
  vec2 P = top ? vec2(cx1 + side * hw1, y1) : vec2(cx0 + side * hw0, y0);
  if (col != 1) w = 0.0;
  vW = row < uRowsLens ? vec2(w, 0.0) : vec2(0.0, w);
  vec2 t = P * s;
  gl_Position = dj >= 0.0 ? vec4(t.x / uTgt.x * 2.0 - 1.0, 1.0 - t.y / uTgt.y * 2.0, 0.0, 1.0)
                          : vec4(2.0, 2.0, 2.0, 1.0);
}`;

const CAUS_TUBE_FS = `#version 300 es
precision highp float;
in vec2 vW;
out vec4 o;
void main() { o = vec4(vW, 0.0, 0.0); }`;

// 3×3 tent against residual sampling noise, normalise, smooth in time.
// Out: R = caustic excess (traced − reference, over blob-touched tubes),
//      G = the full reference field (the light without wax lenses).
const CAUS_UPDATE_FS = `#version 300 es
precision highp float;
uniform highp sampler2D uAcc;   // R: traced, G: reference (blob tubes)
uniform highp sampler2D uRef;   // G: full reference
uniform highp sampler2D uPrev;
uniform float uInvRef, uSM, uInit;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 hi = textureSize(uAcc, 0) - 1;
  vec2 v = vec2(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      float w = float((2 - abs(i)) * (2 - abs(j)));
      v += w * texelFetch(uAcc, clamp(p + ivec2(i, j), ivec2(0), hi), 0).rg;
    }
  }
  // The reference is smooth; a taller tent (≈ a slab between planes)
  // removes the faint per-slab steps.
  float g = 0.0, gw = 0.0;
  for (int j = -4; j <= 4; j++) {
    for (int i = -1; i <= 1; i++) {
      float w = float((2 - abs(i)) * (5 - abs(j)));
      g += w * texelFetch(uRef, clamp(p + ivec2(i, 2 * j), ivec2(0), hi), 0).g;
      gw += w;
    }
  }
  vec2 cur = vec2((v.r - v.g) / 16.0, g / gw) * uInvRef;
  // (with TAA the ray grid is jittered every trace, so this history
  // averages several samplings of the light)
  vec2 s = uInit > 0.5 ? cur : mix(texelFetch(uPrev, p, 0).rg, cur, uSM);
  o = vec4(s, 0.0, 1.0);
}`;

// Light reaching the glass, per row and side (texel x = 0 left, 1 right),
// for the lit glass edge: the traced light in the liquid (reference +
// excess) over a band 6-26 px in from the glass and ±30 px in height,
// relative to its clear-lamp mean along the glass.
const CAUS_EDGE_FS = `#version 300 es
precision highp float;
uniform highp sampler2D uCaus;
uniform vec2  uSim;
uniform float uEdgeScale;
out vec4 o;
${BOTTLE_GLSL}
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
}`;

// Refined ray tubes → wall: each refined main tube's (uSub+1)² rays as
// uSub² small tubes, drawn with the same rules (all four rays alive, close
// together, same path). Tubes that weren't refined have no rays here and
// draw nothing.
const WALL_SUB_VS = `#version 300 es
precision highp float;
uniform highp sampler2D uSubRays;
uniform int   uSub;
uniform ivec3 uRayDim;      // of the main trace
uniform vec2  uView;
uniform vec2  uCell;
uniform float uMaxSpan;
out vec2 vE;
int S, S1, a, t;
float cross2(vec2 u, vec2 v) { return u.x * v.y - u.y * v.x; }
vec4 sr(int i, int j) { return texelFetch(uSubRays, ivec2(a * S1 + i, t * S1 + j), 0); }
// small tube (i, j) of this block: irradiance (liquid, wax), or x < 0
vec2 sub(int i, int j) {
  if (i < 0 || j < 0 || i >= S || j >= S) return vec2(-1.0);
  vec4 r00 = sr(i, j), r01 = sr(i + 1, j), r10 = sr(i, j + 1), r11 = sr(i + 1, j + 1);
  float s0 = floor(r00.w);
  bool ok = min(min(r00.z, r01.z), min(r10.z, r11.z)) > 0.0
         && max(distance(r00.xy, r11.xy), distance(r01.xy, r10.xy)) <= uMaxSpan
         && floor(r01.w) == s0 && floor(r10.w) == s0 && floor(r11.w) == s0;
  if (!ok) return vec2(-1.0);
  vec2 a0 = r00.xy / uCell, a1 = r01.xy / uCell, b0 = r10.xy / uCell, b1 = r11.xy / uCell;
  float area = 0.5 * abs(cross2(a1 - a0, b0 - a0)) + 0.5 * abs(cross2(b1 - a1, b0 - a1));
  float E = 0.25 * (r00.z + r01.z + r10.z + r11.z);
  float wf = 0.25 * (fract(r00.w) + fract(r01.w) + fract(r10.w) + fract(r11.w));
  return E / max(area, 0.7) * vec2(1.0 - wf, wf);
}
void main() {
  int q = gl_VertexID / 6, k = gl_VertexID - q * 6;
  S = uSub; S1 = S + 1;
  int i = q % S; q /= S;
  int j = q % S; q /= S;
  a = q % (uRayDim.x - 1); t = q / (uRayDim.x - 1);     // t = source · (polar − 1) + tube row
  if (sub(i, j).x < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vE = vec2(0.0); return; }
  // triangles (00, 01, 10) and (01, 11, 10); this corner's ray
  int kk = k < 3 ? k : k - 3;
  ivec2 c = k < 3 ? (kk == 0 ? ivec2(0, 0) : kk == 1 ? ivec2(1, 0) : ivec2(0, 1))
                  : (kk == 0 ? ivec2(1, 0) : kk == 1 ? ivec2(1, 1) : ivec2(0, 1));
  int vi = i + c.x, vj = j + c.y;
  // smooth shading, as WALL_SPLAT_VS (within this block)
  vec2 sum = vec2(0.0); float n = 0.0;
  for (int dy = -1; dy <= 0; dy++) for (int dx = -1; dx <= 0; dx++) {
    vec2 e = sub(vi + dx, vj + dy);
    if (e.x >= 0.0) { sum += e; n += 1.0; }
  }
  vE = sum / max(n, 1.0);
  vec4 me = sr(vi, vj);
  gl_Position = vec4(me.x / uView.x * 2.0 - 1.0, 1.0 - me.y / uView.y * 2.0, 0.0, 1.0);
}`;


// TAA output: the mean of the last uCount traces (a ring of layers). The
// traces cycle through uCount jitter offsets of the ray grid, so in a
// still scene the window always holds the same set of samplings and the
// result doesn't change at all — no flicker; moving light is averaged
// over the window.
const RING_AVG_FS = `#version 300 es
precision highp float;
uniform highp sampler2DArray uRing;
uniform int uCount;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 s = vec2(0.0);
  for (int i = 0; i < 16; i++) {
    if (i >= uCount) break;
    s += texelFetch(uRing, ivec3(p, i), 0).rg;
  }
  o = vec4(s / float(uCount), 0.0, 1.0);
}`;
