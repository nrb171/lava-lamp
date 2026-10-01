// ============================================================
//  The WebGL renderer: the lamp (metaballs, glass, pool), the wall behind
//  it, bloom and tone mapping. Its light-tracing methods live in
//  light-gpu.js, light-cpu.js and light-geometry.js and are installed onto
//  the class below; shaders are in shaders/ (see shaders.js).
// ============================================================

import { SIM_W, SIM_H, bottleHalfFrac } from "../sim/sim.js";
import { VIEW_M, VIEW_T, VIEW_W, VIEW_H } from "../view.js";
import { makeTriBlur, srgbToLinear } from "./util.js";
import { VERTEX_SHADER, LAMP_FS, POST_VS, BLOOM_DOWN_FS, BLOOM_UP_FS, COMPOSITE_FS, LIGHT_TRACE_FS, WALL_PHOTON_VS, WALL_PHOTON_FS, WALL_UPDATE_FS, CAUS_PHOTON_VS, CAUS_PHOTON_FS, CAUS_UPDATE_FS, CAUS_EDGE_FS, RING_AVG_FS, DENOISE_FS } from "./shaders.js";
import { GpuLight } from "./light-gpu.js";
import { CpuLight } from "./light-cpu.js";
import { LightGeometry } from "./light-geometry.js";

export class MetaballRenderer {
  constructor(canvas, maxParticles) {
    this.canvas = canvas;
    this.maxParticles = maxParticles;
    // No MSAA: we draw a single fullscreen quad, so there are no geometry
    // edges to antialias — it would only cost bandwidth on mobile GPUs.
    const gl = canvas.getContext("webgl2", {
      alpha: false, antialias: false, powerPreference: "high-performance",
    });
    if (!gl) throw new Error("WebGL2 is required for this lava lamp.");
    // Need float textures (core in WebGL2; need EXT to actually use as samplable, but RGBA32F sampling NEAREST is core)
    if (!gl.getExtension("EXT_color_buffer_float") && !gl.getExtension("OES_texture_float_linear")) {
      // Sampling RGBA32F with NEAREST works in core WebGL2; this is a soft check.
    }
    this.gl = gl;

    const VS = VERTEX_SHADER;
    const FS = LAMP_FS;

    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error("shader: " + gl.getShaderInfoLog(sh));
      }
      return sh;
    };
    // All passes draw the same fullscreen quad, bound at attribute 0.
    const link = (vsSrc, fsSrc) => {
      const p = gl.createProgram();
      gl.attachShader(p, compile(gl.VERTEX_SHADER, vsSrc));
      gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));
      gl.bindAttribLocation(p, 0, "a_pos");
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        throw new Error("link: " + gl.getProgramInfoLog(p));
      }
      return p;
    };
    const prog = link(VS, FS);
    this.program = prog;

    // -------- HDR targets --------
    // The scene renders linear light into a half-float target, then bloom
    // and the filmic tone map run as small post passes. Without float
    // render-target support the scene shader tone-maps directly.
    this.hdr = !!(gl.getExtension("EXT_color_buffer_half_float") ||
                  gl.getExtension("EXT_color_buffer_float"));
    this.exposure = 0.8;
    this.bloomStrength = 0.06;     // k: fraction of light in the glare halo
    this.bloomThreshold = 0.0;     // physical glare: no threshold
    this.bloomKnee = 0.0;
    if (this.hdr) {
      this.downProg = link(POST_VS, BLOOM_DOWN_FS);
      this.upProg = link(POST_VS, BLOOM_UP_FS);
      this.compProg = link(POST_VS, COMPOSITE_FS);
      const L = (p, names) => Object.fromEntries(names.map(n => [n, gl.getUniformLocation(p, n)]));
      this.downU = L(this.downProg, ["uSrc", "uTexel", "uPrefilter", "uThreshold", "uKnee"]);
      this.upU = L(this.upProg, ["uSrc", "uTexel"]);
      this.compU = L(this.compProg, ["uScene", "uBloom", "uBloomStrength", "uBloomNorm", "uRes", "uTime", "uExposure"]);
      this.targets = null;       // allocated to the canvas size on first render
      // Verify the driver really can render to RGBA16F; else fall back.
      try { this.ensureTargets(4, 4); } catch (e) { this.hdr = false; }
    }

    // Fullscreen quad
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1,-1,  1,-1, -1, 1,
      -1, 1,  1,-1,  1, 1,
    ]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // Particle texture (1 row): (x, y, temperature, groupId + compression/4)
    // per particle. Compression (0..3) rides in the fractional part of the
    // group id so the shader needs one texel fetch per particle, not two.
    // Particles are uploaded each frame in cell-sorted order so the
    // shader can do an O(1) range lookup per cell.
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, maxParticles, 1, 0, gl.RGBA, gl.FLOAT, null);

    this.pixData = new Float32Array(maxParticles * 4);

    // -------- Spatial grid (CPU → shader) --------
    // cellSize equals the kernel radius h_render = sim.h * renderScale.
    // The smallest cellSize the UI permits is sim.h (=26) * renderScale_min
    // (=0.7) ≈ 18.2, which gives ceil(380/18.2)=21 wide and ceil(700/18.2)=39
    // tall. We allocate 32 × 48 with margin so the texture is fixed-size
    // and re-uploads are just texSubImage updates.
    this.MAX_GRID_W = 32;
    this.MAX_GRID_H = 48;
    const gridCells = this.MAX_GRID_W * this.MAX_GRID_H;

    // Use RGBA32F (not RG32F): identical format to the particle texture,
    // which we know samples correctly on this driver. We only fill R and
    // G; B and A are unused. The 2x memory cost is trivial (~24 KB).
    this.cellRangeTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.cellRangeTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F,
                  this.MAX_GRID_W, this.MAX_GRID_H, 0,
                  gl.RGBA, gl.FLOAT, null);

    // Scratch buffers for the per-frame counting sort.
    this.cellCount     = new Int32Array(gridCells);
    this.cellStart     = new Int32Array(gridCells);
    this.cellTmpOffset = new Int32Array(gridCells);
    this.particleCell  = new Int32Array(maxParticles);

    // -------- Pool-merge ghosts --------
    // When a blob joins the pool, the sim regroups its particles instantly;
    // rendering that directly makes the neck, rim and shading pop in one
    // frame. Instead its particles keep rendering as ghost group 32+k whose
    // fusion with the pool (uGhostMix) eases 0 → 1 over mergeFadeSec, while
    // its depth and size uniforms slide to the pool's. See the shader.
    this.MAX_GHOSTS = 4;
    this.mergeFadeSec = 1.0;
    this.MIN_GHOST_PARTICLES = 3;
    this.ghosts = Array.from({ length: this.MAX_GHOSTS },
                             () => ({ active: false, m: 0, z0: 0, size0: 0, members: 0 }));
    this.ghostOf  = new Int8Array(maxParticles).fill(-1);
    this.prevGid  = new Int32Array(maxParticles);
    this.prevBlobZ    = new Float32Array(32);
    this.prevBlobSize = new Float32Array(32);
    this.prevPool = -1;
    this.prevN = -1;
    this.prevReset = -1;
    this.lastTime = -1;
    this.joinTally = new Int32Array(32);
    this.joinGhost = new Int8Array(32);
    this.blobZ36    = new Float32Array(36);
    this.blobSize36 = new Float32Array(36);
    this.ghostMix   = new Float32Array(4);
    // 4 floats per cell to match RGBA32F; only .r and .g are populated.
    this.cellRangeData = new Float32Array(gridCells * 4);

    // -------- 2D mass grid (50 cols × 30 rows) --------
    // Each cell accumulates wax mass at that (x, y) region.
    // A mass-dependent horizontal blur simulates light scattering:
    // dense regions stay sharp, sparse regions spread.
    this.NUM_COLS = 100;
    this.NUM_ROWS = 60;
    // Enable float texture linear filtering if available; fall back to NEAREST
    const floatLinear = gl.getExtension('OES_texture_float_linear');
    const colMassFilter = floatLinear ? gl.LINEAR : gl.NEAREST;
    this.colMassTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.colMassTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, colMassFilter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, colMassFilter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F,
                  this.NUM_COLS, this.NUM_ROWS, 0, gl.RGBA, gl.FLOAT, null);
    this.colMassData = new Float32Array(this.NUM_COLS * this.NUM_ROWS * 4);
    // Smoothed 2D mass grid for temporal stability
    this.colMassSmooth = new Float32Array(this.NUM_COLS * this.NUM_ROWS);
    // Per-frame scratch grids, allocated once (no per-frame GC churn).
    const G = this.NUM_COLS * this.NUM_ROWS;
    this.massRaw       = new Float32Array(G);
    this.lightField    = new Float32Array(G);
    this.massBelow     = new Float32Array(G);
    this.blurRadius    = new Int32Array(G);
    this.lightBlurred  = new Float32Array(G);
    this.lightSmoothed = new Float32Array(G);
    this.lightFinal    = new Float32Array(G);
    this.VOL_MAX_RADIUS = 24;
    this.VOL_SMOOTH_R = 4;
    this.rowBlur = makeTriBlur(this.NUM_COLS, this.VOL_MAX_RADIUS);
    this.colBlur = makeTriBlur(this.NUM_ROWS, this.VOL_SMOOTH_R);
    // Caustics (see traceCaustics): per-frame buffers + smoothed outputs
    this.waxFrac      = new Float32Array(G);          // wax volume fraction φ
    this.fTrace       = new Float32Array(G);          // fluence, refracted+reflected rays
    this.fStraight    = new Float32Array(G);          // fluence, same rays unbent
    this.causticRaw   = new Float32Array(G);
    this.causticSmooth = new Float32Array(G);
    this.wallE        = new Float32Array(this.NUM_ROWS * 2);   // [row*2 + side]
    this.wallSmooth   = new Float32Array(this.NUM_ROWS * 2);
    this.causticRef   = null;                          // no-wax normalisation
    // last two traces (caustic field, wall irradiance) for interpolation
    this.capPrev  = new Float32Array(G); this.capCur  = new Float32Array(G);
    this.wallPrev = new Float32Array(this.NUM_ROWS * 2);
    this.wallCur  = new Float32Array(this.NUM_ROWS * 2);
    // two-stage (critically damped) temporal filter state
    this.capF1 = new Float32Array(G);  this.wallF1 = new Float32Array(this.NUM_ROWS * 2);
    // Wall behind the lamp (see traceWall3D): light from the bulb that
    // reaches the wall through the globe, split into liquid-path and
    // wax-path light, on a square-celled grid spanning the view.
    // wall plane behind the axis (sim px): about 0.6 lamp heights — close
    // enough to catch the light, far enough for it to spread and fall off
    this.WALL_DIST = 400;
    this.WALL_DIST_REF = 200;              // brightness is normalised as if the wall were here
    // Two wall textures (previous and newest trace); the shader blends
    // between them every frame, so the CPU only touches the wall grid on
    // the frames it traces.
    this.bdTex = [gl.createTexture(), gl.createTexture()];
    for (const tx of this.bdTex) {
      gl.bindTexture(gl.TEXTURE_2D, tx);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);   // half-float: filterable
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    // Caustics and the wall light are traced on the GPU (LIGHT_TRACE_FS)
    // when the driver can render to 32-bit float targets (ray landing
    // points need the precision); otherwise traceCaustics / traceWall3D do
    // it on the CPU.
    this.wallGpu = false;
    if (gl.getExtension("EXT_color_buffer_float")) {
      try {
        this.lightProg = link(VS, LIGHT_TRACE_FS);
        this.wallPhotonProg = link(WALL_PHOTON_VS, WALL_PHOTON_FS);
        this.wallUpdProg = link(VS, WALL_UPDATE_FS);
        this.causPhotonProg = link(CAUS_PHOTON_VS, CAUS_PHOTON_FS);
        this.causEdgeProg = link(VS, CAUS_EDGE_FS);
        this.ringAvgProg = link(VS, RING_AVG_FS);
        this.denoiseProg = link(VS, DENOISE_FS);
        this.scratchFbo = gl.createFramebuffer();
        this.rings = {};
        this.causUpdProg = link(VS, CAUS_UPDATE_FS);
        const L = (p, names) => Object.fromEntries(names.map(n => [n, gl.getUniformLocation(p, n)]));
        this.ltU = L(this.lightProg, ["uSim", "uViewM", "uViewT", "uWallD", "uGlow", "uRayScale",
                                      "uRayDim", "uAzSpan", "uYSrc", "uMuPool", "uNB", "uWB0", "uWB1",
                                      "uPool", "uPoolN", "uLensOn", "uMode", "uRowOff",
                                      "uPoolYMin", "uSrcSigma", "uSeed"]);
        this.wpU = L(this.wallPhotonProg, ["uRays", "uRaySize", "uView", "uSigma"]);
        this.wuU = L(this.wallUpdProg, ["uIrr", "uAvg", "uPrev", "uInvRef", "uAlpha", "uSM", "uInit", "uFull"]);
        this.cpU = L(this.causPhotonProg, ["uV0", "uV1", "uV2", "uV3", "uNAz", "uRowsLens", "uRefSrc", "uSim", "uTgt", "uSig"]);
        this.cuU = L(this.causUpdProg, ["uAcc", "uRef", "uPrev", "uInvRef", "uSM", "uInit"]);
        this.ceU = L(this.causEdgeProg, ["uCaus", "uSim", "uEdgeScale"]);
        this.raU = L(this.ringAvgProg, ["uRing", "uAcc", "uCount", "uSlot", "uHist"]);
        this.dnU = L(this.denoiseProg, ["uSrc", "uAcc", "uStep", "uK"]);
        this.causEdgeTex = gl.createTexture();
        this.emptyVao = gl.createVertexArray();     // the splat / line passes have no attributes
        this.avgTex = [gl.createTexture(), gl.createTexture()];
        this.irrTex = gl.createTexture();
        this.poolTex = gl.createTexture();
        this.rayTex = null; this.rayKey = "";
        this.pathTex = [0, 1, 2, 3].map(() => gl.createTexture()); this.pathKey = "";
        this.refTex = [gl.createTexture(), gl.createTexture()];
        this.refFbo = [null, null];
        this.causTex = [gl.createTexture(), gl.createTexture()];
        this.causAcc = gl.createTexture();
        this.causCur = 0; this.causFramesSince = 0; this.causKey = "";
        this._wb0 = new Float32Array(160); this._wb1 = new Float32Array(160);
        this.wallGpu = true;
      } catch (e) { console.warn("GPU light trace unavailable, using the CPU:", e); }
    }
    this.bdCurTex = 0;
    this.wallFramesSince = 0;
    this.WALL_AVG_TAU = 25;                // seconds (running-average baseline)
    this.WG_W = 0; this.WG_H = 0;
    this.ensureWallGrid();
    // per-particle time since the particle left the pool (for fading a
    // blob's lens in as it breaks free)
    this.freeAge = new Float32Array(maxParticles).fill(10);
    this.poolTime = new Float32Array(maxParticles);

    // Uniforms
    this.uloc = (n) => gl.getUniformLocation(prog, n);
    this.u = {
      particles:   this.uloc("uParticles"),
      cellRange:   this.uloc("uCellRange"),
      sim:         this.uloc("uSim"),
      res:         this.uloc("uRes"),
      h:           this.uloc("uH"),
      gridDim:     this.uloc("uGridDim"),
      bg:          this.uloc("uBg"),
      cold:        this.uloc("uCold"),
      hot:         this.uloc("uHot"),
      glow:        this.uloc("uGlow"),
      time:        this.uloc("uTime"),
      blobZ:       this.uloc("uBlobZ"),
      blobSize:    this.uloc("uBlobSize"),
      sizeScale:   this.uloc("uSizeScale"),
      v0:          this.uloc("uV0"),
      muWax:       this.uloc("uMuWax"),
      colMass:       this.uloc("uColMass"),
      backdrop:      this.uloc("uBackdrop"),
      backdrop2:     this.uloc("uBackdrop2"),
      caus:          this.uloc("uCaus"),
      caus2:         this.uloc("uCaus2"),
      causMix:       this.uloc("uCausMix"),
      causGpu:       this.uloc("uCausGpu"),
      causEdge:      this.uloc("uCausEdge"),
      wallMix:       this.uloc("uWallMix"),
      viewM:         this.uloc("uViewM"),
      viewT:         this.uloc("uViewT"),
      wallOn:        this.uloc("uWall"),
      poolId:        this.uloc("uPoolId"),
      lens:          this.uloc("uLens"),
      directOut:     this.uloc("uDirectOut"),
      exposure:      this.uloc("uExposure"),
      caustics:      this.uloc("uCaustics"),
      ghostMix:      this.uloc("uGhostMix"),
      numCols:       this.uloc("uNumCols"),
      numRows:       this.uloc("uNumRows"),
    };
  }

  // Half-float scene target plus a bloom pyramid (½, ¼, … of the canvas).
  ensureTargets(w, h) {
    const gl = this.gl;
    if (this.targets && this.targets.w === w && this.targets.h === h) return;
    if (this.targets) {
      for (const t of [this.targets.scene, ...this.targets.mips]) {
        gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo);
      }
    }
    const make = (tw, th) => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, tw, th, 0, gl.RGBA, gl.HALF_FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      if (!ok) throw new Error("RGBA16F render target unsupported");
      return { tex, fbo, w: tw, h: th };
    };
    const mips = [];
    let mw = w, mh = h;
    for (let i = 0; i < 6; i++) {
      mw = Math.max(1, mw >> 1); mh = Math.max(1, mh >> 1);
      mips.push(make(mw, mh));
      if (mw <= 8 || mh <= 8) break;
    }
    this.targets = { w, h, scene: make(w, h), mips };
  }

  // Bloom (dual-filter pyramid) and composite to the canvas.
  postProcess(params) {
    const gl = this.gl, T = this.targets;
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    // downsample chain; the first pass keeps only bright light
    gl.useProgram(this.downProg);
    gl.uniform1i(this.downU.uSrc, 0);
    gl.uniform1f(this.downU.uThreshold, this.bloomThreshold);
    gl.uniform1f(this.downU.uKnee, this.bloomKnee);
    let src = T.scene;
    for (let i = 0; i < T.mips.length; i++) {
      const dst = T.mips[i];
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
      gl.viewport(0, 0, dst.w, dst.h);
      gl.bindTexture(gl.TEXTURE_2D, src.tex);
      gl.uniform2f(this.downU.uTexel, 1 / src.w, 1 / src.h);
      gl.uniform1f(this.downU.uPrefilter, i === 0 && this.bloomThreshold > 0 ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      src = dst;
    }
    // upsample back, adding each blurred level onto the next larger one
    gl.useProgram(this.upProg);
    gl.uniform1i(this.upU.uSrc, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = T.mips.length - 1; i > 0; i--) {
      const s2 = T.mips[i], dst = T.mips[i - 1];
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
      gl.viewport(0, 0, dst.w, dst.h);
      gl.bindTexture(gl.TEXTURE_2D, s2.tex);
      gl.uniform2f(this.upU.uTexel, 1 / s2.w, 1 / s2.h);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.disable(gl.BLEND);
    // composite: scene + bloom → exposure → filmic → sRGB
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.compProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, T.scene.tex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, T.mips[0].tex);
    gl.uniform1i(this.compU.uScene, 0);
    gl.uniform1i(this.compU.uBloom, 1);
    gl.uniform1f(this.compU.uBloomStrength, params.bloom !== false ? this.bloomStrength : 0);
    gl.uniform1f(this.compU.uBloomNorm, 1 / T.mips.length);
    gl.uniform2f(this.compU.uRes, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.compU.uTime, params.time);
    gl.uniform1f(this.compU.uExposure, this.exposure);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  // Track particles joining the pool and advance/release merge ghosts.
  updateGhosts(sim, dt) {
    const n = sim.n, nFixed = sim.nFixed, gid = sim.groupId;
    const pool = nFixed > 0 ? gid[0] : -1;
    const ghostOf = this.ghostOf, ghosts = this.ghosts;

    // After a reset / particle-count change, indices and groups are new.
    if (n !== this.prevN || sim.resetCount !== this.prevReset) {
      ghostOf.fill(-1);
      for (const g of ghosts) g.active = false;
    } else {
      for (const g of ghosts) if (g.active) g.m += dt / this.mergeFadeSec;
      // Particles that leave the pool again are their own blob, not a ghost.
      for (let i = nFixed; i < n; i++) {
        if (ghostOf[i] >= 0 && gid[i] !== pool) ghostOf[i] = -1;
      }
      // New joiners, grouped by the blob they came from.
      const tally = this.joinTally, map = this.joinGhost;
      tally.fill(0); map.fill(-1);
      const prevGid = this.prevGid, prevPool = this.prevPool;
      for (let i = nFixed; i < n; i++) {
        if (gid[i] === pool && prevGid[i] !== prevPool && ghostOf[i] < 0) tally[prevGid[i]]++;
      }
      for (let p = 0; p < 32; p++) {
        if (tally[p] < this.MIN_GHOST_PARTICLES) continue;
        const k = ghosts.findIndex(g => !g.active);
        if (k < 0) break;                       // all slots busy: plain merge
        const g = ghosts[k];
        g.active = true; g.m = 0;
        g.z0 = this.prevBlobZ[p]; g.size0 = this.prevBlobSize[p];
        map[p] = k;
      }
      for (let i = nFixed; i < n; i++) {
        if (gid[i] === pool && prevGid[i] !== prevPool && ghostOf[i] < 0) {
          const k = map[prevGid[i]];
          if (k >= 0) ghostOf[i] = k;
        }
      }
      // Release finished or emptied ghosts.
      for (const g of ghosts) g.members = 0;
      for (let i = nFixed; i < n; i++) if (ghostOf[i] >= 0) ghosts[ghostOf[i]].members++;
      for (let k = 0; k < ghosts.length; k++) {
        const g = ghosts[k];
        if (g.active && (g.m >= 1 || g.members === 0)) {
          g.active = false;
          for (let i = nFixed; i < n; i++) if (ghostOf[i] === k) ghostOf[i] = -1;
        }
      }
    }

    // Uniforms: sim blobs 0-31, ghosts 32-35 easing toward the pool's look.
    this.blobZ36.set(sim.blobZ.subarray(0, 32));
    this.blobSize36.set(sim.blobSizeSmooth.subarray(0, 32));
    const poolZ = pool >= 0 ? sim.blobZ[pool] : 0.5;
    const poolSize = pool >= 0 ? sim.blobSizeSmooth[pool] : 1;
    for (let k = 0; k < 4; k++) {
      const g = ghosts[k];
      const m = g.active ? Math.min(1, g.m) : 1;
      const e = m * m * (3 - 2 * m);                       // smoothstep ease
      this.ghostMix[k] = e;
      this.blobZ36[32 + k]    = g.z0 + (poolZ - g.z0) * e;
      this.blobSize36[32 + k] = g.size0 + (poolSize - g.size0) * e;
    }

    this.prevGid.set(gid.subarray(0, n));
    this.prevPool = pool;
    this.prevN = n;
    this.prevReset = sim.resetCount;
    this.prevBlobZ.set(sim.blobZ.subarray(0, 32));
    this.prevBlobSize.set(sim.blobSizeSmooth.subarray(0, 32));
    return pool;
  }

  render(sim, params) {
    const gl = this.gl;
    const n = sim.n;
    const dt = this.lastTime < 0 ? 0
      : Math.min(0.1, Math.max(0, params.time - this.lastTime)) * (sim.timeScale || 1);
    this.lastTime = params.time;
    const poolGid = this.updateGhosts(sim, dt);
    // Time since each particle left the pool. It only restarts once the
    // particle has really been back in the pool for 0.3 s: particles on the
    // pool's edge flicker in and out of its group frame to frame, and
    // restarting on every flicker made the lens strengths shimmer.
    {
      const fa = this.freeAge, pt = this.poolTime, gid = sim.groupId;
      if (sim.resetCount !== this._faReset) { fa.fill(10); pt.fill(0); this._faReset = sim.resetCount; }
      for (let i = sim.nFixed; i < n; i++) {
        if (gid[i] === poolGid) { pt[i] += dt; if (pt[i] > 0.3) fa[i] = 0; }
        else { pt[i] = 0; fa[i] += dt; }
      }
    }
    this.ensureWallGrid();
    const ghostOf = this.ghostOf;

    // -------- 1. Counting-sort particles into spatial grid cells --------
    // cellSize == kernel radius the shader uses (uH). With this choice,
    // every particle that can contribute to a pixel lives in the pixel's
    // cell or one of the 8 neighbors → 3×3 cell sweep is exact.
    const hRender = sim.h * (sim.renderScale || 1.0);
    const cellSize = Math.max(hRender, 1.0);
    const MAX_GW = this.MAX_GRID_W;
    const MAX_GH = this.MAX_GRID_H;
    const gridW = Math.min(MAX_GW, Math.max(1, Math.ceil(SIM_W / cellSize)));
    const gridH = Math.min(MAX_GH, Math.max(1, Math.ceil(SIM_H / cellSize)));
    const numCells = gridW * gridH;
    const invCell = 1.0 / cellSize;

    const cellCount = this.cellCount;
    const cellStart = this.cellStart;
    const tmpOffset = this.cellTmpOffset;
    const partCell  = this.particleCell;

    // (a) Count particles per active cell + record cell index per particle
    for (let c = 0; c < numCells; c++) cellCount[c] = 0;
    // Interpolated positions from the main loop when available (fixed
    // physics step), else the sim's current positions.
    const sx = params.px || sim.x, sy = params.py || sim.y;
    for (let i = 0; i < n; i++) {
      let cx = (sx[i] * invCell) | 0;
      let cy = (sy[i] * invCell) | 0;
      if (cx < 0) cx = 0; else if (cx >= gridW) cx = gridW - 1;
      if (cy < 0) cy = 0; else if (cy >= gridH) cy = gridH - 1;
      const c = cy * gridW + cx;
      partCell[i] = c;
      cellCount[c]++;
    }
    // (b) Prefix sum → cellStart
    let total = 0;
    for (let c = 0; c < numCells; c++) {
      cellStart[c] = total;
      total += cellCount[c];
      tmpOffset[c] = 0;
    }
    // (c) Scatter into cell-sorted particle data
    const data = this.pixData;
    const stemp = sim.temp, sgroup = sim.groupId;
    const scomp = sim.compression;
    for (let i = 0; i < n; i++) {
      const c = partCell[i];
      const slot = cellStart[c] + tmpOffset[c];
      const dst = slot << 2;  // ×4
      data[dst    ] = sx[i];
      data[dst + 1] = sy[i];
      data[dst + 2] = stemp[i];
      const cmp = scomp[i];
      const rg = ghostOf[i] >= 0 ? 32 + ghostOf[i] : sgroup[i];
      data[dst + 3] = rg + (cmp < 3 ? cmp : 2.999) * 0.25;
      tmpOffset[c]++;
    }
    // (d) Build cellRange texture (RGBA32F: r=start, g=count, b/a unused)
    // — laid out with MAX_GRID_W stride, only active gridW × gridH region
    // populated.
    const rangeData = this.cellRangeData;
    // Zero the rows we'll touch (cheaper than zeroing the whole array).
    const rowStride = MAX_GW * 4;
    const activeRowEnd = gridH * rowStride;
    for (let i = 0; i < activeRowEnd; i++) rangeData[i] = 0;
    for (let cy = 0; cy < gridH; cy++) {
      const rowBase = cy * rowStride;
      const cellRowBase = cy * gridW;
      for (let cx = 0; cx < gridW; cx++) {
        const t = rowBase + (cx << 2);
        const c = cellRowBase + cx;
        rangeData[t    ] = cellStart[c];
        rangeData[t + 1] = cellCount[c];
      }
    }

    // -------- 1b. 2D mass grid for volumetric background modulation --------
    const NC = this.NUM_COLS;
    const NR = this.NUM_ROWS;
    const colData = this.colMassData;
    const colSmooth = this.colMassSmooth;
    const volAbsorbOn = params.volAbsorb !== false;
    if (!volAbsorbOn) {
      // Skip all mass computation — zero the texture and jump ahead
      colData.fill(0);
    } else {
    // Step 1: Bin particles into a 2D grid (NUM_COLS × NUM_ROWS).
    const colW = SIM_W / NC;
    const rowH = SIM_H / NR;
    const gridSize = NC * NR;
    const massRaw = this.massRaw;
    massRaw.fill(0);
    const nFixed = sim.nFixed;
    for (let i = nFixed; i < n; i++) {
      const ci = (sx[i] / colW) | 0;
      const ri = (sy[i] / rowH) | 0;
      const c0 = ci < 0 ? 0 : ci >= NC ? NC - 1 : ci;
      const r0 = ri < 0 ? 0 : ri >= NR ? NR - 1 : ri;
      const idx = r0 * NC + c0;
      massRaw[idx] += 1.0;
      if (r0 > 0)      massRaw[idx - NC] += 0.25;
      if (r0 < NR - 1) massRaw[idx + NC] += 0.25;
      if (c0 > 0)      massRaw[idx - 1] += 0.25;
      if (c0 < NC - 1) massRaw[idx + 1] += 0.25;
    }

    // Step 2: Per-column light transport (bottom → top).
    // Light starts just above the pool surface and walks upward.
    // Pool particles would absorb everything, so we begin the transport
    // at the pool boundary row and ignore mass below it.
    // Pool surface is at t ≈ 0.85 → sim y ≈ 0.85 * SIM_H.
    const lightField = this.lightField;
    const ABSORB_RATE = 0.18;  // absorption per unit mass (tuned for ~160 particles)
    const poolRow = Math.floor(0.82 * NR);  // row index of pool surface
    for (let c = 0; c < NC; c++) {
      // Pool base brightness: based on bottle width at pool surface.
      // Use a wider falloff so light reaches close to the bottle edge.
      const cx = (c + 0.5) * colW;
      const poolHalf = bottleHalfFrac(0.85) * SIM_W;
      const distFromCenter = Math.abs(cx - SIM_W * 0.5);
      const edgeT = Math.min(distFromCenter / poolHalf, 1.0);
      // Remap edgeT so brightness stays high until close to the edge
      const edgeFalloff = edgeT * edgeT * edgeT;  // cubic — stays bright longer, drops fast at edge
      const poolBrightness = Math.max(0, 1.0 - edgeFalloff) * (params.glow / 0.38);
      // Rows below pool surface: full brightness (pool glow handled by shader)
      for (let r = NR - 1; r > poolRow; r--) {
        lightField[r * NC + c] = poolBrightness;
      }
      // Walk upward from pool surface, attenuating by mass
      let light = poolBrightness;
      for (let r = poolRow; r >= 0; r--) {
        const m = massRaw[r * NC + c];
        light *= Math.exp(-m * ABSORB_RATE);
        lightField[r * NC + c] = light;
      }
    }

    // Step 2b: caustics from wax lensing + wall reflection (see traceCaustics).
    const causticsOn = params.caustics !== false;
    const lightGpu = causticsOn && this.wallGpu && params.lightCpu !== true;
    this.rayScale = params.rayScale || 1;
    this.taa = params.taa !== false;
    this.denoise = params.denoise !== false;
    this.wallFull = params.wallFull !== false;
    this._lightGpuOn = lightGpu;
    if (lightGpu) {
      // Everything on the GPU (gpuLight); the CPU only builds the blob list
      // and the pool surface. Full quality: the liquid every frame, the
      // wall every 2nd. Low quality: a trace every 2 frames, wall and
      // liquid taking turns (each every 4th). The shader eases from each
      // one's previous trace to its newest in between.
      // (the first step down of the adaptive quality already switches the
      // light to its low tier: cheaper to give up than pixel sharpness)
      const lowQ = params.quality !== undefined && params.quality < 0.9;
      const wallOn = params.wall !== false;
      const every = lowQ ? 2 : 1;
      this._causticTick = (this._causticTick || 0) + 1;
      if (this._causticTick % every === 0) {
        const turn = (this._lightTurn = (this._lightTurn || 0) + 1) & 1;
        const doWall = wallOn && turn === 0;
        const doCaus = !lowQ || !wallOn || turn === 1 || !this.causInit;
        this.causInterval = (lowQ && wallOn ? 2 : 1) * every;
        this.wallInterval = (wallOn ? 2 : 1) * every;
        this.gpuLight(sim, sx, sy, params.glow / 0.38, lowQ, doWall, doCaus,
                      this.wallInterval / 60, this.causInterval / 60, every / 60);
      }
    } else if (causticsOn) {
      if (!this.causticRef) {
        // Normalise against an empty lamp at the reference glow, so the
        // numbers are "fraction of clear-lamp light".
        this.traceCaustics(sim, massRaw, 1.0, true, sx, sy);
        // f: brightest clear-lamp fluence; e: mean irradiance on lit glass
        let fMax = 0, eSum = 0, eN = 0;
        for (const v of this.fStraight) fMax = Math.max(fMax, v);
        for (const v of this.wallE) if (v > 0) { eSum += v; eN++; }
        this.causticRef = { f: fMax || 1, e: eN ? eSum / eN : 1 };
      }
      const glowN = params.glow / 0.38;
      // Trace every `every` frames and ease linearly from the previous trace
      // to the newest over the frames in between (a couple of frames of
      // latency, but the caustics move every frame instead of stepping).
      const every = (params.quality !== undefined && params.quality < 0.8) ? 4 : 2;
      this._causticTick = (this._causticTick || 0) + 1;
      const phase = this._causticTick % every;
      if (phase === 0) {
        this.traceCaustics(sim, massRaw, glowN, false, sx, sy);
        const cr = this.causticRaw, ref = this.causticRef;
        for (let k = 0; k < gridSize; k++) cr[k] = (this.fTrace[k] - this.fStraight[k]) / ref.f;
        // smooth out the discreteness of the ray families
        const ct = this.lightSmoothed;              // scratch; recomputed below
        for (let r = 0; r < NR; r++) this.rowBlur(cr, r * NC, 1, 3, ct);
        for (let c = 0; c < NC; c++) this.colBlur(ct, c, NC, 3, cr);
        // shift: prev ← what's currently displayed, cur ← new trace
        this.capPrev.set(this.capCur); this.capCur.set(cr);
        this.wallPrev.set(this.wallCur);
        for (let k = 0; k < this.wallCur.length; k++) this.wallCur[k] = this.wallE[k] / ref.e;
        // The wall trace is the heaviest part: run it every other caustic
        // trace (the interpolation + filter below keeps it smooth).
        this._wallTick = (this._wallTick || 0) + 1;
        if (params.wall !== false && (this._wallTick & 1) === 0) {
          this.traceWall3D(sim, sx, sy, glowN);
          const GWn = this.bdL.length;
          if (!this.bdRef) {
            // normalise by the mean wall light beside the lamp at first trace
            let sum = 0, cnt = 0;
            const cw = VIEW_W / this.WG_W, ch = VIEW_H / this.WG_H;
            for (let r = 0; r < this.WG_H; r++) {
              const ys = (r + 0.5) * ch - VIEW_T;
              if (ys < 0.1 * SIM_H || ys > 0.9 * SIM_H) continue;
              const hwc = bottleHalfFrac(ys / SIM_H) * SIM_W;
              for (let c = 0; c < this.WG_W; c++) {
                if (Math.abs((c + 0.5) * cw - VIEW_M - SIM_W / 2) > hwc) {
                  const k = r * this.WG_W + c; sum += this.wlL[k] + this.wlW[k]; cnt++;
                }
              }
            }
            this.bdRef = cnt && sum > 0 ? sum / cnt : 1;
          }
          // Only light above the lamp's running-average (default) wall light
          // is drawn; negative where wax takes light away (the shader clamps
          // each component at zero).
          if (sim.resetCount !== this._bdResetSeen) { this.bdAvgInit = false; this._bdResetSeen = sim.resetCount; }
          const aL = this.bdAvgL, aW = this.bdAvgW;
          // While the lamp settles (first ~20 s after start/reset) the
          // average follows quickly, so the wall starts dark.
          if (!this.bdAvgInit) this._bdAge = 0;
          this._bdAge += 2 * every / 60;
          const tau = this._bdAge < 20 ? 2 : this.WALL_AVG_TAU;
          const alpha = Math.min(1, 2 * every / 60 / tau);
          // Light temporal smoothing per trace (the shader then blends the
          // last two traces linearly between trace frames).
          const sL = this.bdL, sW = this.bdW, SM = 0.5;
          for (let k = 0; k < GWn; k++) {
            const vL = this.wlL[k] / this.bdRef, vW = this.wlW[k] / this.bdRef;
            if (!this.bdAvgInit) { aL[k] = vL; aW[k] = vW; sL[k] = 0; sW[k] = 0; }
            else { aL[k] += (vL - aL[k]) * alpha; aW[k] += (vW - aW[k]) * alpha; }
            sL[k] += ((vL - aL[k]) - sL[k]) * SM;
            sW[k] += ((vW - aW[k]) - sW[k]) * SM;
          }
          this.bdAvgInit = true;
          // pack (rows flipped: texture v = 1 at the top) and upload into
          // the older of the two textures, which becomes the newest
          const NW = this.WG_W, NH = this.WG_H, bd = this.bdData;
          for (let r = 0; r < NH; r++) {
            const src = r * NW, dst = (NH - 1 - r) * NW;
            for (let c = 0; c < NW; c++) { const o = (dst + c) << 1; bd[o] = sL[src + c]; bd[o + 1] = sW[src + c]; }
          }
          this.bdCurTex ^= 1;
          gl.bindTexture(gl.TEXTURE_2D, this.bdTex[this.bdCurTex]);
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, NW, NH, gl.RG, gl.FLOAT, bd, 0);
          this.wallFramesSince = 0;
          this.wallInterval = 2 * every;
        }
      }
      // Linear interpolation between traces, then a critically damped
      // two-stage low-pass (≈45 ms per stage) — like a camera or eye
      // integrating over a frame, which is why real caustics look flowing.
      // Sharp caustic cusps sweeping across cells otherwise read as flicker.
      const u = (phase + 1) / every;
      const A = 0.3;
      const cs = this.causticSmooth, cp = this.capPrev, cc = this.capCur, f1 = this.capF1;
      for (let k = 0; k < gridSize; k++) {
        const v = cp[k] + (cc[k] - cp[k]) * u;
        f1[k] += (v - f1[k]) * A;
        cs[k] += (f1[k] - cs[k]) * A;
      }
      const ws = this.wallSmooth, wp = this.wallPrev, wc = this.wallCur, g1 = this.wallF1;
      for (let k = 0; k < ws.length; k++) {
        const v = wp[k] + (wc[k] - wp[k]) * u;
        g1[k] += (v - g1[k]) * A;
        ws[k] += (g1[k] - ws[k]) * A;
      }
    }

    // Step 3: Vertical convolution for blur-kernel field.
    // For each cell, sum mass in the same column over LOOK_BELOW rows
    // below it. This determines how much horizontal smoothing to apply.
    const LOOK_BELOW = 10;
    const massBelow = this.massBelow;
    for (let c = 0; c < NC; c++) {
      // Running sum with a sliding window (bottom to top)
      let runSum = 0;
      // Seed: sum the bottom LOOK_BELOW rows
      for (let r = NR - 1; r >= Math.max(0, NR - LOOK_BELOW); r--) {
        runSum += massRaw[r * NC + c];
      }
      massBelow[(NR - 1) * NC + c] = runSum;
      // Slide upward
      for (let r = NR - 2; r >= 0; r--) {
        // Add the row that just entered the window (r itself)
        // The window covers rows [r .. r+LOOK_BELOW-1]
        runSum += massRaw[r * NC + c];
        // Remove the row that fell out (r + LOOK_BELOW)
        if (r + LOOK_BELOW < NR) {
          runSum -= massRaw[(r + LOOK_BELOW) * NC + c];
        }
        massBelow[r * NC + c] = runSum;
      }
    }

    // Step 4: Mass-dependent horizontal blur of the light field.
    // High massBelow → narrow blur (shadow stays sharp).
    // Low massBelow → wide blur (light scatters/refracts).
    const lightBlurred = this.lightBlurred;
    const MAX_RADIUS = this.VOL_MAX_RADIUS;
    const blurRadius = this.blurRadius;
    for (let k = 0; k < gridSize; k++) {
      const massFactor = Math.min(massBelow[k] / 10.0, 1.0);
      blurRadius[k] = Math.round(MAX_RADIUS * (1.0 - massFactor));
    }
    for (let r = 0; r < NR; r++) {
      this.rowBlur(lightField, r * NC, 1, blurRadius, lightBlurred);
    }

    // Step 5: Universal spatial blur — smooth the entire light field
    // with a fixed-radius triangle kernel (horizontal, then vertical) so
    // small mass changes don't cause the light to jump around.
    const SMOOTH_R = this.VOL_SMOOTH_R;
    const lightSmoothed = this.lightSmoothed;
    for (let r = 0; r < NR; r++) {
      this.rowBlur(lightBlurred, r * NC, 1, SMOOTH_R, lightSmoothed);
    }
    const lightFinal = this.lightFinal;
    for (let c = 0; c < NC; c++) {
      this.colBlur(lightSmoothed, c, NC, SMOOTH_R, lightFinal);
    }

    // Step 6: Temporal smoothing (EMA) — no normalization needed,
    // light values are already 0-1 from the transport.
    const smoothK = 0.15;
    for (let k = 0; k < gridSize; k++) {
      colSmooth[k] += (lightFinal[k] - colSmooth[k]) * smoothK;
    }
    // Pack into RGBA32F texture.
    // R = light intensity (0 = fully blocked, 1 = full pool brightness).
    // Texture row 0 = sim bottom (pool), row NR-1 = sim top.
    for (let r = 0; r < NR; r++) {
      const srcOff = r * NC;
      // Flip: sim row r (y down) → texture row (NR-1-r) so texV=0 is top
      const dstRow = (NR - 1 - r) * NC;
      for (let c = 0; c < NC; c++) {
        const dstIdx = (dstRow + c) << 2;
        const light = Math.min(colSmooth[srcOff + c], 1.0);
        colData[dstIdx]     = light;
        // G = caustic excess, B = irradiance on the glass on this side
        colData[dstIdx + 1] = causticsOn && !lightGpu ? this.causticSmooth[srcOff + c] : 0;
        colData[dstIdx + 2] = causticsOn && !lightGpu ? this.wallSmooth[r * 2 + (c < NC / 2 ? 0 : 1)] : 0;
        colData[dstIdx + 3] = 0;
      }
    }
    } // end volAbsorb gate

    // -------- 2. Upload textures (always reset active unit first) --------
    // (Render targets first: allocating them binds textures on the active
    // unit, which would displace the ones bound below.)
    const useHdr = this.hdr && params.hdr !== false;
    if (useHdr) this.ensureTargets(this.canvas.width, this.canvas.height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, n, 1, gl.RGBA, gl.FLOAT, data, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.cellRangeTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, MAX_GW, gridH, gl.RGBA, gl.FLOAT, rangeData, 0);

    // 2D mass grid texture on unit 2
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.colMassTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, NC, NR, gl.RGBA, gl.FLOAT, colData, 0);

    // light on the wall behind the lamp: previous trace on unit 3, newest
    // on unit 4; the shader blends them by how far we are to the next trace
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.bdTex[this.bdCurTex ^ 1]);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.bdTex[this.bdCurTex]);
    this.wallFramesSince++;
    // GPU caustics: previous trace on unit 5, newest on unit 6
    if (this.wallGpu) {
      gl.activeTexture(gl.TEXTURE5);
      gl.bindTexture(gl.TEXTURE_2D, this.causTex[this.causCur ^ 1]);
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, this.causTex[this.causCur]);
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, this.causEdgeTex);
      this.causFramesSince++;
    }

    // -------- 3. Draw --------
    if (useHdr) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.targets.scene.fbo);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.program);
    if (this.u.directOut) gl.uniform1f(this.u.directOut, useHdr ? 0 : 1);
    if (this.u.exposure) gl.uniform1f(this.u.exposure, this.exposure);
    gl.bindVertexArray(this.vao);
    // Textures are already bound on units 0/1 from the upload above.
    gl.uniform1i(this.u.particles, 0);
    if (this.u.cellRange) gl.uniform1i(this.u.cellRange, 1);
    gl.uniform2f(this.u.sim, SIM_W, SIM_H);
    if (this.u.viewM) gl.uniform1f(this.u.viewM, VIEW_M);
    if (this.u.viewT) gl.uniform1f(this.u.viewT, VIEW_T);
    gl.uniform2f(this.u.res, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.u.h, hRender);
    if (this.u.gridDim) gl.uniform2i(this.u.gridDim, gridW, gridH);
    // colour pickers are sRGB; shading is in linear light
    gl.uniform3fv(this.u.bg, srgbToLinear(params.bg, this._linBg || (this._linBg = new Float32Array(3))));
    gl.uniform3fv(this.u.cold, srgbToLinear(params.cold, this._linCold || (this._linCold = new Float32Array(3))));
    gl.uniform3fv(this.u.hot, srgbToLinear(params.hot, this._linHot || (this._linHot = new Float32Array(3))));
    gl.uniform1f(this.u.glow, params.glow);
    gl.uniform1f(this.u.time, params.time);
    // Per-blob z values for the front/back picker and the overlap
    // highlight. Pass the whole array (size = MAX_G).
    if (this.u.blobZ) gl.uniform1fv(this.u.blobZ, this.blobZ36);
    // Per-blob smoothed visual size (normalized 0-1). Uses the
    // smoothed values so split-off blobs transition gradually.
    // blob opacity (waxOpacity): absolute sizes from the relative ones
    if (this.u.sizeScale) {
      let maxN = 1;
      for (let k = 0; k < sim.MAX_BLOBS; k++) if (sim.cmn[k] > maxN) maxN = sim.cmn[k];
      gl.uniform1f(this.u.sizeScale, maxN);
      gl.uniform1f(this.u.v0, sim.mass / sim.restDensity);
      gl.uniform1f(this.u.muWax, 0.0125);      // ≈ 0.5 opaque at 10 particles, 0.7 at 30, 0.9 at 100
    }
    if (this.u.blobSize) {
      gl.uniform1fv(this.u.blobSize, this.blobSize36);
    }
    if (this.u.colMass) { gl.uniform1i(this.u.colMass, 2); }
    if (this.u.backdrop) gl.uniform1i(this.u.backdrop, 3);
    if (this.u.backdrop2) gl.uniform1i(this.u.backdrop2, 4);
    if (this.u.wallMix) gl.uniform1f(this.u.wallMix,
      Math.min(1, this.wallFramesSince / (this.wallInterval || 4)));
    const causGpu = this._lightGpuOn && params.volAbsorb !== false && this.causInit;
    if (this.u.causGpu) gl.uniform1f(this.u.causGpu, causGpu ? 1 : 0);
    if (this.u.caus) gl.uniform1i(this.u.caus, 5);
    if (this.u.caus2) gl.uniform1i(this.u.caus2, 6);
    if (this.u.causEdge) gl.uniform1i(this.u.causEdge, 7);
    if (this.u.causMix) gl.uniform1f(this.u.causMix,
      Math.min(1, this.causFramesSince / (this.causInterval || 2)));
    if (this.u.wallOn) gl.uniform1f(this.u.wallOn,
      params.wall !== false && params.caustics !== false && params.volAbsorb !== false ? 1 : 0);
    if (this.u.poolId) gl.uniform1f(this.u.poolId, poolGid);
    if (this.u.ghostMix) gl.uniform4fv(this.u.ghostMix, this.ghostMix);
    if (this.u.lens) gl.uniform1f(this.u.lens, params.lens !== false ? 1 : 0);
    if (this.u.caustics) gl.uniform1f(this.u.caustics,
      params.caustics !== false && params.volAbsorb !== false ? 1 : 0);
    if (this.u.numCols) gl.uniform1i(this.u.numCols, params.volAbsorb !== false ? this.NUM_COLS : 0);
    if (this.u.numRows) gl.uniform1i(this.u.numRows, params.volAbsorb !== false ? this.NUM_ROWS : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    if (useHdr) this.postProcess(params);
  }
}

// The light-tracing methods, defined in their own modules, become methods
// of MetaballRenderer (same `this`, as if written in the class).
for (const Part of [GpuLight, CpuLight, LightGeometry]) {
  for (const key of Object.getOwnPropertyNames(Part.prototype)) {
    if (key === "constructor") continue;
    if (key in MetaballRenderer.prototype) throw new Error(`MetaballRenderer.${key} defined twice`);
    Object.defineProperty(MetaballRenderer.prototype, key, Object.getOwnPropertyDescriptor(Part.prototype, key));
  }
}
