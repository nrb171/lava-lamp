#version 300 es
precision highp float;
uniform vec2  uSim;
uniform float uViewM, uViewT;
uniform float uWallD;       // wall plane behind the lamp axis (sim px)
uniform float uGlow;
uniform float uRayScale;    // power per ray (keeps totals independent of ray count)
uniform ivec3 uRayDim;      // rays: azimuth × polar × sources
uniform float uAzSpan;      // azimuths covered: π (toward the wall) or 2π
uniform float uYSrc;        // the light source: on the axis just under the pool's resting surface (sim px)
// The bulb is wide and diffuse (uSrcSigma > 0): every ray leaves from its
// own point, Gaussian-distributed about the lamp's axis (σ = uSrcSigma, sim
// px), in its own direction within its cell of the ray grid — both random,
// from uSeed (one per TAA slot). Rays are then independent samples
// ("photons"), splatted and summed, not tubes.
uniform float uSrcSigma;
uniform float uSeed;
uniform float uMuPool;      // attenuation in the pool wax (per sim px)
uniform int   uNB;
uniform vec4  uWB0[40];     // blob centre (x from the axis, y, z), lens strength
uniform vec4  uWB1[40];     // blob semi-axes
uniform highp sampler2D uPool;  // row 0: pool base top y vs radius
uniform int   uPoolN;
uniform float uLensOn;      // 0 = reference: no blobs, no bumps
uniform int   uMode;        // 0 = wall landing, 1 = path vertices
uniform float uPoolYMin;    // highest point of the pool surface (smallest y)
uniform int   uRowOff;
// mode 0: wall x, wall y (view px), power, signature + wax share (o only)
// mode 1: path vertices 4·chunk … 4·chunk+3 (o, o1, o2, o3); vertex 7's
//         slot holds (1 if the path met a blob, its signature, …) instead
layout(location = 0) out vec4 o;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
#include "common/bottle.glsl"
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
// the lens ellipsoid (other than skip) that p is inside of and d goes on
// through, or -1
int insideWax(vec3 p, vec3 d, int skip) {
  if (uLensOn < 0.5) return -1;
  float h0, h1;
  for (int b = 0; b < 40; b++) {
    if (b >= uNB) break;
    if (b != skip && hitE(b, p, d, h0, h1) && h0 < 0.0 && h1 > 1e-2) return b;
  }
  return -1;
}
vec3 eNormal(int b, vec3 p) {
  vec3 a = uWB1[b].xyz;
  return normalize((p - uWB0[b].xyz) / (a * a));
}
// PCG-style 4D hash (Jarzynski & Olano 2020)
uvec4 pcg4(uvec4 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
  v ^= v >> 16u;
  v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
  return v;
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

void main() {
  ivec2 id = ivec2(gl_FragCoord.xy);
  id.y -= uRowOff;
  int ia = id.x;
  if (uMode == 1) { ia = id.x / 2; kBase = 4 * (id.x - ia * 2); }
  for (int i = 0; i < 4; i++) pv[i] = vec4(0.0);
  int ip = id.y % uRayDim.y;
  float fia = float(ia), fip = float(ip);
  vec4 rnd = vec4(0.0);
  if (uSrcSigma > 0.0) {
    // per ray, not per fragment: mode 1 runs each ray as two fragments
    uvec4 h = pcg4(uvec4(uint(ia), uint(id.y), floatBitsToUint(uSeed), 0x9e3779b9u));
    rnd = vec4(h) * (1.0 / 4294967296.0);
    // the origins stratified over each 4×4 tile of rays (one per cell of a
    // 4×4 grid over the Gaussian's (u, v), shuffled per tile and seed):
    // independent draws clump, leaving blotches on the wall
    uint tileH = pcg4(uvec4(uint(ia / 4), uint(id.y / 4), floatBitsToUint(uSeed), 7u)).x;
    uint k = (uint(ia % 4) + 4u * uint(id.y % 4) + tileH) % 16u;
    k = (k * 7u + (tileH >> 8u)) % 16u;                   // a shuffle (7 is coprime to 16)
    rnd.zw = (vec2(float(k % 4u), float(k / 4u)) + rnd.zw) * 0.25;
    fia = float(ia) + rnd.x - 0.5; fip = float(ip) + rnd.y - 0.5;
    fip = clamp(fip, 0.0, float(uRayDim.y - 1));
  }
  // The bulb: diffuse (Lambertian), from a Gaussian spot about the axis.
  // Out to (almost) horizontal: a Lambertian source's light fades to
  // nothing there by itself. Stopping at 77° cut it off at a quarter of
  // its peak — a hard edge across the wall.
  float cMin = cos(1.55);
  float c = 1.0 - (1.0 - cMin) * fip / float(uRayDim.y - 1);
  float sph = sqrt(1.0 - c * c);
  float al = PI + uAzSpan * (fia + 0.5) / float(uRayDim.x);        // from −x round through the back
  vec3 d = vec3(sph * cos(al), -c, sph * sin(al));
  vec2 s0 = vec2(0.0);
  if (uSrcSigma > 0.0) {                                  // Box–Muller
    float r = uSrcSigma * min(sqrt(-2.0 * log(max(rnd.z, 1e-7))), 2.5);
    s0 = r * vec2(cos(2.0 * PI * rnd.w), sin(2.0 * PI * rnd.w));
  }
  vec3 p = vec3(s0.x, uYSrc, s0.y);
  float P = c * uGlow * uRayScale / float(uRayDim.z);         // Lambertian
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
    // under a bump or column (wax ellipsoids standing on the base): wax on
    // both sides, no surface here
    if (ci > 0.0 && insideWax(p, d, -1) >= 0) { p += d * 0.05; out_ = true; break; }
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
    if (hitE(b, p, d, h0, h1) && h0 < 0.0 && h1 > 0.0) {
      inB = b; hitBlob = true; sig += float(b + 1); nEnt = 1; break;
    }
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
      // Still inside another ellipsoid here (blobs merging, a column on its
      // bump): wax on both sides, no surface. Refracting there bent rays
      // near grazing by ~20° — a dark V above the bulb, thrown out of the
      // middle by the columns of a rising pool.
      int nx = insideWax(p, d, inB);
      if (nx >= 0) { inB = nx; continue; }
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
}
