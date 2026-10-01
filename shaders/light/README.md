# Lamp light on the GPU

These shaders compute the caustics in the liquid and the light on the wall behind the lamp.

One ray tracer, `trace.frag`, follows the lamp's light:

- **The bulb:** rays start from a diffuse (Lambertian) source, spread as a Gaussian spot about the axis just under the pool's resting surface.
  - Each ray starts from its own random point, in its own random direction within its cell of the ray grid.
- **The pool:** where there is wax above the bulb, a ray passes up through it, dimmed, to the pool's base surface. That surface is a smooth height field rebuilt each trace from the pool particles.
  - The surface refracts the ray (wax → liquid), with Fresnel loss and total internal reflection.
  - The pool's bumps and columns are ellipsoids, like the blobs. Overlapping wax is one body, with no surface between the parts.
- **The blobs:** rays refract into and out of the wax blobs (ellipsoids).
- **The glass:** rays leave through the curved glass (Fresnel, total internal reflection) toward the wall.

The tracer has two modes:

- **Mode 0** writes where each ray lands on the wall.
- **Mode 1** writes the vertices of each ray's path: the pool, each blob surface and the glass.

Each ray is one sample ("photon") of the light. The samples are summed, then averaged over the TAA ring with a fresh random seed each trace (`ring-avg.frag`), and denoised (`denoise.frag`).

## Wall

1. `wall-photon.vert` / `.frag` splats each landing as a small Gaussian, at 2× the wall grid's resolution.
2. `wall-update.frag` filters that down and keeps a running-average baseline.
3. It then outputs either the difference from the baseline or all of the light, for drawing.

## Liquid

1. `caustic-photon.vert` / `.frag` draws each straight piece of a ray's path as a thin Gaussian band. The band carries the light the ray scatters there, which is power × length.
2. Two versions are drawn: the traced light, and a reference without the wax lenses and pool bumps.
   - Only rays that a blob or bump touched are drawn, since the rest are identical in both and cancel.
   - The full reference changes slowly, so it is rebuilt only every few updates.
3. `caustic-update.frag` filters and normalises the result.
4. `caustic-edge.frag` measures the light arriving at the glass, for the lit glass edge.

Shared code is pulled in with `#include "common/….glsl"` (see `src/render/shaders.js`).
