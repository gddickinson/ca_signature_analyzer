// Subtle animated background: a slowly scrolling calcium-spike trace on the #bg canvas.
// Off under prefers-reduced-motion and while the tab is hidden.
export function startBackground() {
  const cv = document.getElementById('bg');
  if (!cv || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const ctx = cv.getContext('2d');
  const size = () => { cv.width = innerWidth; cv.height = innerHeight; };
  size(); addEventListener('resize', size);
  const spike = x => { const p = ((x % 260) + 260) % 260; return p < 30 ? Math.exp(-((p - 12) ** 2) / 30) : 0.25 * Math.exp(-(p - 30) / 60) * (p > 30); };
  let last = 0;
  const draw = t => {
    requestAnimationFrame(draw);
    if (document.hidden || t - last < 50) return;
    last = t;
    const w = cv.width, h = cv.height, off = t / 40;
    ctx.clearRect(0, 0, w, h);
    [[0.78, '#7dff6a', 1], [0.88, '#ffcf3f', 0.6]].forEach(([yb, col, a], k) => {
      ctx.strokeStyle = col; ctx.globalAlpha = 0.35 * a; ctx.lineWidth = 1.4; ctx.beginPath();
      for (let x = 0; x <= w; x += 4) {
        const y = h * yb - 90 * spike(x + off * (k ? 0.7 : 1) + k * 90);
        x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  };
  requestAnimationFrame(draw);
}
