// ============================================================
//  Shader sources, loaded from shaders/ at start-up.
//
//  Each shader is its own .glsl / .vert / .frag file, so it can be edited
//  (and syntax-highlighted) on its own. A line
//      #include "common/bottle.glsl"
//  is replaced by that file (paths relative to shaders/). The exports
//  below are filled in by loadShaders(), which src/main.js awaits before
//  building the renderer; ES module exports are live, so importers see the
//  loaded text.
// ============================================================

const FILES = {
  VERTEX_SHADER:  "fullscreen.vert",
  LAMP_FS:        "lamp.frag",
  POST_VS:        "post/post.vert",
  BLOOM_DOWN_FS:  "post/bloom-down.frag",
  BLOOM_UP_FS:    "post/bloom-up.frag",
  COMPOSITE_FS:   "post/composite.frag",
  LIGHT_TRACE_FS: "light/trace.frag",
  WALL_PHOTON_VS: "light/wall-photon.vert",
  WALL_PHOTON_FS: "light/wall-photon.frag",
  WALL_UPDATE_FS: "light/wall-update.frag",
  CAUS_PHOTON_VS: "light/caustic-photon.vert",
  CAUS_PHOTON_FS: "light/caustic-photon.frag",
  CAUS_UPDATE_FS: "light/caustic-update.frag",
  CAUS_EDGE_FS:   "light/caustic-edge.frag",
  RING_AVG_FS:    "light/ring-avg.frag",
  DENOISE_FS:     "light/denoise.frag",
};

export let VERTEX_SHADER, LAMP_FS, POST_VS, BLOOM_DOWN_FS, BLOOM_UP_FS, COMPOSITE_FS,
  LIGHT_TRACE_FS, WALL_PHOTON_VS, WALL_PHOTON_FS, WALL_UPDATE_FS,
  CAUS_PHOTON_VS, CAUS_PHOTON_FS, CAUS_UPDATE_FS, CAUS_EDGE_FS, RING_AVG_FS, DENOISE_FS;

const ROOT = new URL("../../shaders/", import.meta.url);

async function fetchText(path) {
  const res = await fetch(new URL(path, ROOT));
  if (!res.ok) throw new Error(`shader ${path}: ${res.status} ${res.statusText}`);
  return res.text();
}

// Replace #include lines with the included file (once per file per shader).
async function resolveIncludes(text, cache) {
  const lines = text.split("\n");
  const out = [];
  for (const line of lines) {
    const m = line.match(/^\s*#include\s+"([^"]+)"\s*$/);
    if (!m) { out.push(line); continue; }
    if (!cache.has(m[1])) cache.set(m[1], fetchText(m[1]).then((t) => resolveIncludes(t, cache)));
    out.push((await cache.get(m[1])).replace(/\n$/, ""));
  }
  return out.join("\n");
}

export async function loadShaders() {
  const cache = new Map();
  const entries = await Promise.all(Object.entries(FILES).map(async ([name, path]) =>
    [name, await resolveIncludes(await fetchText(path), cache)]));
  const src = Object.fromEntries(entries);
  ({ VERTEX_SHADER, LAMP_FS, POST_VS, BLOOM_DOWN_FS, BLOOM_UP_FS, COMPOSITE_FS,
     LIGHT_TRACE_FS, WALL_PHOTON_VS, WALL_PHOTON_FS, WALL_UPDATE_FS,
     CAUS_PHOTON_VS, CAUS_PHOTON_FS, CAUS_UPDATE_FS, CAUS_EDGE_FS, RING_AVG_FS, DENOISE_FS } = src);
}
