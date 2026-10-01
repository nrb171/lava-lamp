import { SIM_W, SIM_H, bottleHalfFrac } from "../sim/sim.js";
import { VIEW_M, VIEW_T, VIEW_W, VIEW_H } from "../view.js";
import { $ } from "./dom.js";

export function initGrab({ canvas, sim, params }) {
  // --- Mouse / touch "grab" interaction ---
  // Clicking and holding inside the lava lamp captures the fluid particles
  // nearest to the cursor. While the pointer is held down, those particles
  // follow the cursor via a spring + damping force (see sim.applyGrab),
  // which composes with the regular SPH physics — gravity, pressure,
  // surface tension, etc. all still apply.
  (() => {
    const grabRing = $("grabRing");
    let pointerId = null;
    let lastEventTime = 0;
    let lastSimX = 0, lastSimY = 0;
    // We use the canvas getBoundingClientRect each event so the mapping
    // works under fullscreen scaling, devicePixelRatio, and panel toggles.
    // The view of the interior goes through the glass lens (see shader), so
    // map the pointer the same way: grab what you see, not what's behind it.
    function lensToSim(sx, sy) {
      if (params.lens === false) return sx;
      const cx = SIM_W * 0.5, R = bottleHalfFrac(sy / SIM_H) * SIM_W;
      const b = Math.abs(sx - cx);
      if (R <= 1 || b >= R) return sx;
      const sI = b / R, delta = Math.asin(sI) - Math.asin(sI / 1.34);
      const x0 = Math.max(0, b - Math.sqrt(R * R - b * b) * Math.tan(delta));
      return cx + Math.sign(sx - cx) * x0;
    }
    function clientToSim(ev) {
      const rect = canvas.getBoundingClientRect();
      const ax = (ev.clientX - rect.left) * (VIEW_W / rect.width) - VIEW_M;
      const sy = (ev.clientY - rect.top)  * (VIEW_H / rect.height) - VIEW_T;
      // ringX: apparent (on-screen) position for the halo
      return { sx: lensToSim(ax, sy), sy, rect, ringX: ax };
    }
    function placeRing(rect, sx, sy, visible) {
      // grab-ring is absolutely positioned inside .stage (same offset
      // parent as the canvas). Convert sim coords back to CSS pixels
      // relative to the stage by using the canvas offset within the stage.
      const stage = canvas.parentElement;
      const stageRect = stage.getBoundingClientRect();
      const cssX = (rect.left - stageRect.left) + (sx + VIEW_M) * (rect.width / VIEW_W);
      const cssY = (rect.top  - stageRect.top ) + (sy + VIEW_T) * (rect.height / VIEW_H);
      const cssDiameter = 2 * sim.grab.radius * (rect.width / VIEW_W);
      grabRing.style.width  = cssDiameter + "px";
      grabRing.style.height = cssDiameter + "px";
      grabRing.style.left = cssX + "px";
      grabRing.style.top  = cssY + "px";
      grabRing.classList.toggle("active", !!visible);
    }

    canvas.addEventListener("pointerdown", (ev) => {
      // Ignore secondary buttons (right-click etc.)
      if (ev.button !== 0 && ev.pointerType === "mouse") return;
      const { sx, sy, rect, ringX } = clientToSim(ev);
      const grabbed = sim.beginGrab(sx, sy);
      if (!grabbed) return; // empty space — let the click do nothing
      pointerId = ev.pointerId;
      canvas.setPointerCapture(pointerId);
      canvas.classList.add("grabbing");
      lastEventTime = performance.now();
      lastSimX = sx; lastSimY = sy;
      placeRing(rect, ringX, sy, true);
      ev.preventDefault();
    });

    canvas.addEventListener("pointermove", (ev) => {
      if (pointerId === null || ev.pointerId !== pointerId) return;
      const now = performance.now();
      const dt = Math.max(0.001, (now - lastEventTime) / 1000);
      const { sx, sy, rect, ringX } = clientToSim(ev);
      const vx = (sx - lastSimX) / dt;
      const vy = (sy - lastSimY) / dt;
      sim.updateGrab(sx, sy, vx, vy);
      lastEventTime = now;
      lastSimX = sx; lastSimY = sy;
      placeRing(rect, ringX, sy, true);
    });

    function releaseGrab() {
      if (pointerId !== null) {
        try { canvas.releasePointerCapture(pointerId); } catch (_) {}
      }
      pointerId = null;
      canvas.classList.remove("grabbing");
      grabRing.classList.remove("active");
      sim.endGrab();
    }
    canvas.addEventListener("pointerup",     releaseGrab);
    canvas.addEventListener("pointercancel", releaseGrab);
    // If the mouse leaves while not actually capturing (e.g. user
    // released outside the window), make sure we clean up.
    window.addEventListener("blur", releaseGrab);
  })();
}
