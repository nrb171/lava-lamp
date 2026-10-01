import { SIM_W, SIM_H } from "./sim/sim.js";

// The canvas fills the window. The lamp (SIM_W × SIM_H) is fitted in the
// middle with a small margin above and below; the wall behind it fills the
// rest. The canvas shows sim x ∈ [−VIEW_M, SIM_W + VIEW_M] and
// y ∈ [−VIEW_T, SIM_H + VIEW_T]. On a wide window VIEW_M grows; on a
// narrow (portrait) one the lamp fits the width and VIEW_T grows.
export let VIEW_M = 130, VIEW_T = 0.04 * SIM_H;
export let VIEW_W = SIM_W + 2 * VIEW_M, VIEW_H = SIM_H + 2 * VIEW_T;
export function updateView(cssW, cssH) {
  if (!(cssW > 0 && cssH > 0)) return;
  const aspect = cssW / cssH;
  const hFit = SIM_H * 1.08, wFit = SIM_W * 1.06;
  if (aspect * hFit >= wFit) { VIEW_H = hFit; VIEW_W = aspect * VIEW_H; }
  else { VIEW_W = wFit; VIEW_H = VIEW_W / aspect; }
  VIEW_M = 0.5 * (VIEW_W - SIM_W);
  VIEW_T = 0.5 * (VIEW_H - SIM_H);
}
