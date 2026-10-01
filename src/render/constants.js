// Light-source and photon-splat constants shared by the renderer's modules.

// The pool's resting surface, as a fraction of the lamp's height: the
// lamp draws a solid pool below it (shaders: poolBlockMask). The light
// source sits just under it, in the top of the pool.
export const POOL_REST_T = 0.89;
export const LIGHT_SRC_T = POOL_REST_T + 0.015;
// the bulb's light leaves the base from a Gaussian spot about the axis:
// its σ, as a fraction of the lamp's width
export const BULB_SIGMA = 0.07;
// width (σ, sim px) of each wall photon's splat
export const WALL_PHOTON_SIGMA = 8;
// … and of each liquid photon's, across (sim px)
export const CAUS_PHOTON_SIGMA = 5;
