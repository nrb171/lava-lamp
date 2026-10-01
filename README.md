# Lava lamp

A WebGL2 lava lamp: an SPH fluid simulation of the wax, ray-traced light (caustics in the liquid and on the wall behind), and bloom with filmic tone mapping.

Live at https://nrb171.github.io/lava-lamp/

## Running it

There is no build step. Serve the folder over HTTP and open it:

```sh
python3 -m http.server 8765    # or: npm run serve
```

Then go to http://localhost:8765. Opening `index.html` straight from disk (`file://`) won't work: browsers block ES modules and shader `fetch` from `file://`.

## Layout

```
index.html              the page: markup only
src/
  main.js               start-up and main loop: loads shaders, builds the sim and renderer,
                        adapts render quality, steps physics, draws
  style.css             the page's styles
  view.js               how the lamp is framed in the window (wall margins)
  sim/sim.js            the SPH simulation (no DOM; also run headless by tools/)
  render/
    renderer.js         MetaballRenderer: lamp, glass, pool, wall, bloom, tone mapping
    light-gpu.js        GPU light: ray tracing, photon splats, TAA accumulation, denoising
    light-cpu.js        CPU fallback for the light (no float render targets)
    light-geometry.js   blobs as ellipsoids and the pool as a height field, for the tracers
    shaders.js          loads shaders/ at start-up (with #include)
    constants.js, util.js
  app/
    controls.js         the controls panel, presets, keyboard shortcuts
    grab.js             click-and-drag the wax
    bench.js            ?bench=<seconds> frame-time benchmark
    dom.js
shaders/
  lamp.frag             the lamp: metaballs, glass, pool, wall
  fullscreen.vert
  post/                 bloom and the final composite
  light/                the light tracer and its passes (see shaders/light/README.md)
  common/               shared snippets, pulled in with #include "common/….glsl"
tools/                  headless Node scripts run against src/sim/sim.js
```

### Editing

- **Shaders:** edit any file in `shaders/` and reload. Each file is a complete GLSL ES 3.00 source apart from its `#include` lines.
- **Rendering:** code is in `src/render/`. The light-tracing methods live in their own modules and are installed onto `MetaballRenderer` at the end of `renderer.js`.
- **Physics:** code is in `src/sim/sim.js`. Check a change headlessly before looking at it in the browser (see below).

In the browser console, `lavaLamp.sim`, `lavaLamp.renderer` and `lavaLamp.params` give access to the running lamp.

## Tools

These need Node 18 or later.

```sh
node tools/blobstats.js 300 6                    # blob behaviour over 6 seeds × 300 s
node tools/blobstats.js 300 6 gravityScale=2     # … with a parameter changed
node tools/simsheet.js out.png 150 4 8 1 1       # PNG snapshots with motion trails
node tools/droptest.js                           # a single drop's surface tension
```

`blobstats.js` reports how fast blobs move and how often they tear, merge or break into fragments. It also reports their roundness and the state of the pool. See the comment at the top of the file for what each number means.
