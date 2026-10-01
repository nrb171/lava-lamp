export function initBench(sim) {
  // -------- Benchmark mode (?bench=<seconds>) --------
  // Records every rAF timestamp for the configured duration, then writes a
  // summary to document.title and an on-page panel. First 60 frames are
  // dropped as JIT warmup. Read progress externally via the tab title.
  (() => {
    const benchSec = parseFloat(new URLSearchParams(location.search).get("bench"));
    if (!benchSec || benchSec <= 0) return;
    const WARMUP_FRAMES = 60;
    const ts = [];   // rAF timestamps after warmup
    let warm = 0;
    let lastTitleSec = -1;
    let benchStart = -1;
    function pct(sortedArr, p) {
      return sortedArr[Math.min(sortedArr.length - 1, Math.floor(sortedArr.length * p))];
    }
    function snapshotStats(timestamps) {
      if (timestamps.length < 2) return null;
      const ft = new Float64Array(timestamps.length - 1);
      for (let i = 1; i < timestamps.length; i++) ft[i - 1] = timestamps[i] - timestamps[i - 1];
      const sorted = Array.from(ft).sort((a, b) => a - b);
      const sum = sorted.reduce((a, b) => a + b, 0);
      const mean = sum / sorted.length;
      return {
        frames: ft.length,
        sec: sum / 1000,
        fps: ft.length / (sum / 1000),
        mean, min: sorted[0], max: sorted[sorted.length - 1],
        p50: pct(sorted, 0.50), p95: pct(sorted, 0.95), p99: pct(sorted, 0.99),
      };
    }
    function fmt(s) {
      return `frames=${s.frames} sec=${s.sec.toFixed(1)} fps=${s.fps.toFixed(1)} `
        + `mean=${s.mean.toFixed(2)}ms p50=${s.p50.toFixed(2)} p95=${s.p95.toFixed(2)} `
        + `p99=${s.p99.toFixed(2)} min=${s.min.toFixed(2)} max=${s.max.toFixed(2)} n=${sim.n}`;
    }
    function tick(now) {
      if (warm < WARMUP_FRAMES) { warm++; requestAnimationFrame(tick); return; }
      if (benchStart < 0) benchStart = now;
      ts.push(now);
      const elapsed = (now - benchStart) / 1000;
      if (elapsed >= benchSec) {
        const s = snapshotStats(ts);
        const summary = "BENCH " + fmt(s);
        document.title = summary;
        const div = document.createElement("div");
        div.id = "__benchPanel";
        div.style.cssText = "position:fixed;top:10px;left:10px;right:10px;z-index:9999;"
          + "background:#0f1f10;color:#a8ffa8;font:12px ui-monospace,monospace;"
          + "padding:12px 14px;border:1px solid #5fbf60;border-radius:8px;"
          + "white-space:pre-wrap;";
        div.textContent = summary;
        document.body.appendChild(div);
        console.log(summary);
        return;  // stop bench loop; main frame() loop continues
      }
      const sec = Math.floor(elapsed);
      if (sec !== lastTitleSec) {
        lastTitleSec = sec;
        const s = snapshotStats(ts);
        if (s) document.title = `bench t=${sec}/${benchSec}s ${fmt(s)}`;
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  })();
}
