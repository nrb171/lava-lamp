#version 300 es
// Liquid photons (wide bulb): each ray's path through the liquid is
// straight between its vertices (pool surface, blob surfaces, glass), and
// scatters a little light all along: power × length. Each segment is drawn
// once as a thin band, a Gaussian across it (σ = uSig texels), so the sum
// over rays is the light in the liquid. Rows below uRowsLens are the traced
// rays (→ R), the rest the reference without wax lenses (→ G); for the
// difference only rays a blob or pool bump touched are drawn (the rest are
// identical in both and cancel), for the full reference (uRefSrc ≥ 0)
// every reference ray.
precision highp float;
uniform highp sampler2D uV0, uV1, uV2, uV3;   // path vertex k: texture k % 4, column 2·ray + k / 4
uniform int   uNAz;         // rays per row
uniform int   uRowsLens;
uniform int   uRefSrc;
uniform vec2  uSim;
uniform vec2  uTgt;
uniform float uSig;
out vec2 vW;
out float vU;               // across the band, in σ
vec4 vert(int ray, int k, int row) {
  ivec2 c = ivec2(2 * ray + k / 4, row);
  int t = k - (k / 4) * 4;
  return t == 0 ? texelFetch(uV0, c, 0) : t == 1 ? texelFetch(uV1, c, 0)
       : t == 2 ? texelFetch(uV2, c, 0) : texelFetch(uV3, c, 0);
}
void main() {
  int q = gl_InstanceID;
  int k = q % 6; q /= 6;                   // segment: vertex k → k+1
  int ia = q % uNAz; int row = q / uNAz;
  if (uRefSrc >= 0) row += uRowsLens;
  int rt = row < uRowsLens ? row : row - uRowsLens;
  vec4 a = vert(ia, k, row), b = vert(ia, k + 1, row);
  vec2 s = uTgt / uSim;
  vec2 A = a.xy * s, B = b.xy * s;
  float L2 = distance(A, B);
  bool skip = a.w <= 0.0 || distance(a.xyz, b.xyz) < 1e-3
           || (uRefSrc < 0 && vert(ia, 7, rt).x <= 0.0);
  if (skip) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vW = vec2(0.0); vU = 0.0; return; }
  // A segment pointing nearly along the view is short on screen but carries
  // the light of its whole 3D length: drawn at its own length that all
  // landed in a texel or two — a bright point. Bands are at least 2σ long
  // (stretched about their middle), so it spreads like a photon's splat.
  float Lb = max(L2, 2.0 * uSig);
  vec2 dir = L2 > 1e-4 ? (B - A) / L2 : vec2(0.0, 1.0);
  vec2 mid = 0.5 * (A + B);
  A = mid - dir * 0.5 * Lb; B = mid + dir * 0.5 * Lb;
  // power × 3D length (texels), over the band's area: per unit of its
  // length, the Gaussian across it integrates to 1
  float w = a.w * distance(a.xyz, b.xyz) * s.x / Lb / (2.5066283 * uSig);
  vW = row < uRowsLens ? vec2(w, 0.0) : vec2(0.0, w);
  vec2 nrm = vec2(-dir.y, dir.x);
  int side = gl_VertexID & 1, end = gl_VertexID >> 1;
  vU = side == 0 ? -3.0 : 3.0;
  vec2 P = (end == 0 ? A : B) + nrm * vU * uSig;
  gl_Position = vec4(P.x / uTgt.x * 2.0 - 1.0, 1.0 - P.y / uTgt.y * 2.0, 0.0, 1.0);
}
