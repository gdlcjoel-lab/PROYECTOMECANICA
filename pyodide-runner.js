/**
 * pyodide-runner.js — shared Pyodide execution engine for exercise pages
 *
 * Usage in each page:
 *   const runner = new PyodideRunner({ steps, simCode, plotCode, colabUrl });
 *   runner.init();  // call on DOMContentLoaded
 */

const RAW_BASE = 'https://raw.githubusercontent.com/gdlcjoel-lab/PROYECTOMECANICA/main/Definitivos/';

class PyodideRunner {
  constructor({ steps, simCode, plotCode, extraFetch = [] }) {
    this.steps      = steps;      // string[] — step labels
    this.simCode    = simCode;    // Python string — simulation
    this.plotCode   = plotCode;   // Python string — plot + capture
    this.extraFetch = extraFetch; // extra .py files to fetch besides rocket_simulation.py
    this.pyodide    = null;
    this.done       = false;
    this.t0         = null;
  }

  init() {
    // Ensure returning to index.html skips the intro sequence
    sessionStorage.setItem('introSeen', '1');
    document.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
  }

  // ── DOM helpers ─────────────────────────────────────────────────
  _step(idx, state) {
    const n = this.steps.length;
    for (let i = 0; i < n; i++) {
      const el = document.getElementById('step-' + i);
      if (!el) continue;
      el.classList.remove('active', 'done');
      if (i < idx)     el.classList.add('done');
      if (i === idx)   el.classList.add(state);
    }
    const pct = Math.round((idx / (n - 1)) * 90) + (state === 'done' ? 10 : 0);
    const bar = document.getElementById('progressBar');
    if (bar) bar.style.width = Math.min(pct, 100) + '%';
    this._tick();
  }

  _tick() {
    const el = document.getElementById('progressTime');
    if (el && this.t0) el.textContent = ((performance.now() - this.t0) / 1000).toFixed(1) + 's elapsed';
  }

  _setBtn(html, disabled = false, colorGreen = false) {
    const btn = document.getElementById('runBtn');
    if (!btn) return;
    btn.innerHTML = html;
    btn.disabled = disabled;
    if (colorGreen) btn.style.background = 'linear-gradient(135deg,#22D3A0,#22D3A0)';
  }

  // ── Main entry ───────────────────────────────────────────────────
  async run() {
    if (this.done) {
      document.getElementById('plotResult')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      return;
    }
    this.t0 = performance.now();
    this._setBtn('<span>⏳</span> Running…', true);

    const pb  = document.getElementById('progressBox');
    const err = document.getElementById('errorBox');
    if (pb)  pb.classList.add('visible');
    if (err) err.classList.remove('visible');

    const n = this.steps.length;

    try {
      // Step 0 — Load Pyodide
      this._step(0, 'active');
      if (!this.pyodide) {
        this.pyodide = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/' });
      }
      this._step(0, 'done');

      // Step 1 — Packages
      this._step(1, 'active');
      await this.pyodide.loadPackage(['numpy', 'matplotlib']);
      this._step(1, 'done');

      // Step 2 — Download .py files
      this._step(2, 'active');
      const files = ['rocket_simulation.py', ...this.extraFetch];
      for (const f of files) {
        const src = await fetch(RAW_BASE + f).then(r => r.text());
        this.pyodide.runPython(src);
      }
      this._step(2, 'done');

      // Step 3 — Run simulation
      this._step(3, 'active');
      this.pyodide.runPython(`
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
` + this.simCode);
      this._step(3, 'done');

      // Step 4 — Render plot
      this._step(4, 'active');
      this.pyodide.runPython(`
import io, base64
` + this.plotCode + `
_buf = io.BytesIO()
plt.savefig(_buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
_buf.seek(0)
_img_b64 = base64.b64encode(_buf.read()).decode()
plt.close('all')
`);
      document.getElementById('progressBar').style.width = '100%';
      this._step(n - 1, 'done');

      // Show image
      const img64 = this.pyodide.globals.get('_img_b64');
      const imgEl = document.getElementById('plotImg');
      if (imgEl) imgEl.src = 'data:image/png;base64,' + img64;
      const pr = document.getElementById('plotResult');
      if (pr) { pr.classList.add('visible'); pr.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }

      const secs = ((performance.now() - this.t0) / 1000).toFixed(1);
      document.getElementById('progressTime').textContent = `Completed in ${secs}s`;
      this._setBtn(`<span>✓</span> Done · ${secs}s`, false, true);
      this.done = true;

    } catch (e) {
      if (err) { err.textContent = 'Error: ' + e.message; err.classList.add('visible'); }
      this._setBtn('<span>▶</span> Retry', false);
    }
  }
}

// ── Global helpers called by inline onclick ───────────────────────
function toggleCode() {
  const panel = document.getElementById('codePanel');
  const txt   = document.getElementById('codeBtnTxt');
  if (!panel) return;
  const open = panel.classList.toggle('open');
  if (txt) txt.textContent = open ? 'Hide Code' : 'View Code';
  if (open) panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function copyCode() {
  const src = document.getElementById('codeBlock')?.textContent ?? '';
  navigator.clipboard.writeText(src).then(() => {
    const btn = document.querySelector('.code-panel-bar button');
    if (btn) { btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy', 2000); }
  });
}

function downloadPlot() {
  const img = document.getElementById('plotImg');
  if (!img?.src) return;
  const a = document.createElement('a');
  a.href = img.src;
  a.download = 'simulation_plot.png';
  a.click();
}
