#!/usr/bin/env node
// ============================================================
//  Headless contact sheet: runs the sim as the page does (src/main.js: 60 fps,
//  5 substeps, seeded) and writes a PNG of snapshots side by side —
//  particles coloured by blob (pool grey, walls dark), each with a
//  trail of where it was over the last `trail` seconds, so speed and
//  collisions can be judged without the browser.
//
//  Usage: node tools/simsheet.js out.png [start=90] [every=2] [count=8]
//                          [seed=1] [trail=1] [key=value ...]
//  key=value pairs are assigned onto the SPH instance (as blobstats.js).
// ============================================================

import fs from 'node:fs';
import zlib from 'node:zlib';
import { SPH, SIM_W, SIM_H, bottleHalfWidth } from '../src/sim/sim.js';

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function writePNG(path, W, H, rgb) {
  const raw = Buffer.alloc((W * 3 + 1) * H);
  for (let y = 0; y < H; y++) {
    raw[y * (W * 3 + 1)] = 0;
    rgb.copy(raw, y * (W * 3 + 1) + 1, y * W * 3, (y + 1) * W * 3);
  }
  const crcTable = new Int32Array(256).map((_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c;
  });
  const crc = (buf) => { let c = -1; for (const b of buf) c = crcTable[(c ^ b) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; };
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const c = Buffer.alloc(4); c.writeUInt32BE(crc(td));
    return Buffer.concat([len, td, c]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4);
  ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  fs.writeFileSync(path, Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0)),
  ]));
}

const PALETTE = [
  [230, 90, 70], [80, 170, 230], [240, 190, 60], [120, 210, 110], [200, 110, 220],
  [250, 140, 40], [70, 210, 200], [230, 100, 160], [160, 160, 250], [190, 220, 90],
];

const args = process.argv.slice(2);
const out = args[0] || 'sheet.png';
const start = parseFloat(args[1] ?? 90), every = parseFloat(args[2] ?? 2);
const count = parseInt(args[3] ?? 8, 10), seed = parseInt(args[4] ?? 1, 10);
const trail = parseFloat(args[5] ?? 1);
const overrides = {};
for (const kv of args.slice(6)) { const [k, v] = kv.split('='); overrides[k] = parseFloat(v); }

Math.random = mulberry32(seed * 7919);
const sim = new SPH({ numParticles: 160 });
Object.assign(sim, overrides);
const dt = 1 / 60, substeps = 5, sub = dt / substeps;

const S = 1;                                   // px per sim px
const PW = Math.round(SIM_W * S), PH = Math.round(SIM_H * S), GAP = 6;
const W = count * PW + (count - 1) * GAP, H = PH;
const img = Buffer.alloc(W * H * 3, 12);
const put = (x, y, c, a) => {
  x |= 0; y |= 0;
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  const o = (y * W + x) * 3;
  for (let k = 0; k < 3; k++) img[o + k] = img[o + k] * (1 - a) + c[k] * a;
};
const disc = (cx, cy, r, c, a) => {
  for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if (x * x + y * y <= r * r) put(cx + x, cy + y, c, a);
};

const trailFrames = Math.round(trail / dt);
const hist = [];
let frame = 0, shot = 0;
const firstShot = Math.round(start / dt), shotEvery = Math.round(every / dt);
while (shot < count) {
  sim.stepFrame(dt);
  for (let s = 0; s < substeps; s++) { sim.step(sub); sim.applyGrab(sub); }
  frame++;
  hist.push({ x: Float32Array.from(sim.x.subarray(0, sim.n)), y: Float32Array.from(sim.y.subarray(0, sim.n)) });
  if (hist.length > trailFrames + 1) hist.shift();
  if (frame >= firstShot && (frame - firstShot) % shotEvery === 0) {
    const ox = shot * (PW + GAP);
    // glass outline
    for (let y = 0; y < PH; y++) {
      const hw = bottleHalfWidth(y / S) * S;
      if (hw > 0) { put(ox + PW / 2 - hw, y, [90, 90, 110], 1); put(ox + PW / 2 + hw, y, [90, 90, 110], 1); }
    }
    const pool = sim.groupId[sim.nFixed] !== undefined ? sim._poolBlobId : -1;
    const colOf = (i) => {
      if (i < sim.nFixed) return [70, 70, 80];
      const g = sim.groupId[i];
      if (g === pool) return [150, 150, 150];
      return PALETTE[g % PALETTE.length];
    };
    // trails (every 3rd frame, fading)
    for (let k = 0; k < hist.length - 1; k += 3) {
      const a = 0.08 + 0.3 * k / hist.length, hk = hist[k];
      for (let i = sim.nFixed; i < Math.min(sim.n, hk.x.length); i++) put(ox + hk.x[i] * S, hk.y[i] * S, colOf(i), a);
    }
    for (let i = 0; i < sim.n; i++) disc(ox + sim.x[i] * S, sim.y[i] * S, 3, colOf(i), 1);
    // time label as tick marks: one per 10 s
    const secs = Math.round(frame * dt);
    for (let k = 0; k < Math.floor(secs / 10); k++) disc(ox + 6 + k * 6, 6, 2, [200, 200, 200], 1);
    shot++;
  }
}
writePNG(out, W, H, img);
console.log(`wrote ${out}: ${count} shots from ${start}s every ${every}s`);
