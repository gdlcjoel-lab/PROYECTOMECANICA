"""
================================================================================
 plot_utils.py — Shared interactive comparison plot for Tasks 4.1–4.4
================================================================================
 Import this after importing rocket_simulation so rcParams are already applied.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from matplotlib.lines import Line2D

from rocket_simulation import C_NAVY, C_GRID, _style_fig, _style_ax


def plot_comparison(panels, labels, colors, linestyles, suptitle, annotation=None):
    """
    Interactive 2×2 comparison figure with time slider.

    Parameters
    ----------
    panels : list of 4 dicts, each containing:
        't_arrays'     – list of 1-D time arrays, one per curve
        'x_arrays'     – list of 1-D x-data arrays, one per curve
        'y_arrays'     – list of 1-D y-data arrays, one per curve
        'xlabel'       – x-axis label string
        'ylabel'       – y-axis label string
        'ylabel_short' – short label shown in the inline y-value textbox
        'title'        – panel title
        'time_based'   – bool; True when x-axis is time
                         (draws a vertical crosshair + floating x-label)
    labels     : legend label per curve
    colors     : matplotlib colour per curve
    linestyles : matplotlib linestyle per curve
    suptitle   : figure suptitle
    annotation : optional str displayed inside the first panel (key numbers)

    Interaction
    -----------
    • Slider        — scrub time
    • Time textbox  — type a time [s] and press Enter
    • Y textbox     — one per panel, for the PRIMARY curve (curve 0); type a
                      y-value and press Enter to jump to the nearest point
    • Click         — click anywhere on a panel to jump the marker there;
                      time-based panels use x-coordinate, spatial panels use
                      normalised 2-D distance (primary curve)
    """
    n_curves = len(labels)
    t_min = 0.0
    t_max = float(max(p['t_arrays'][ci][-1]
                      for p in panels
                      for ci in range(n_curves)))

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(2, 2,
                           left=0.09, right=0.96,
                           bottom=0.22, top=0.93,
                           hspace=0.50, wspace=0.40)
    axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    _style_fig(fig, suptitle)

    # ── Static curves ──────────────────────────────────────────────────────────
    for ax, p in zip(axs, panels):
        for ci in range(n_curves):
            ax.plot(p['x_arrays'][ci], p['y_arrays'][ci],
                    color=colors[ci], linestyle=linestyles[ci],
                    linewidth=2.0, label=labels[ci], zorder=2)
        _style_ax(ax, p['title'], p['xlabel'], p['ylabel'])
        ax.legend(fontsize=9, framealpha=0.95)
        if not p['time_based']:
            ax.set_aspect('equal', adjustable='datalim')

    fig.canvas.draw()

    # ── Separator line above slider zone ──────────────────────────────────────
    fig.add_artist(Line2D([0.04, 0.96], [0.128, 0.128],
                          transform=fig.transFigure,
                          color=C_GRID, linewidth=0.9, zorder=0))

    # ── Optional annotation (shown in panel 0) ─────────────────────────────────
    if annotation:
        axs[0].text(0.97, 0.97, annotation,
                    transform=axs[0].transAxes,
                    fontsize=8.5, fontfamily='serif', color=C_NAVY,
                    va='top', ha='right', linespacing=1.7,
                    bbox=dict(boxstyle='round,pad=0.45',
                              fc='white', ec=C_NAVY, alpha=0.92, linewidth=0.8))

    # ── Dynamic overlay: dots + vertical crosshair (time-based panels) ─────────
    CH_KW = dict(color='#C0C0C0', linewidth=0.8, linestyle='-', zorder=3)
    LBL_X = dict(fontsize=8, color='#333333', clip_on=False, fontfamily='serif',
                 ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

    dots   = [[None] * n_curves for _ in range(4)]
    vlines = [None] * 4
    xlbls  = [None] * 4

    for pi, (ax, p) in enumerate(zip(axs, panels)):
        ylim = ax.get_ylim()
        for ci in range(n_curves):
            xi, yi = p['x_arrays'][ci][0], p['y_arrays'][ci][0]
            dot, = ax.plot([xi], [yi], 'o',
                           color=colors[ci], markersize=6,
                           markeredgewidth=0.6, markeredgecolor='white',
                           zorder=6)
            dots[pi][ci] = dot
        if p['time_based']:
            xi0 = p['x_arrays'][0][0]
            vl, = ax.plot([xi0, xi0], [ylim[0], ylim[1]], **CH_KW)
            lx  = ax.text(xi0, ylim[0], f'{xi0:.4g}', **LBL_X)
            vlines[pi] = vl
            xlbls[pi]  = lx

    hidden_ticks = []

    def _hide_overlapping(ax, txt, which):
        renderer = fig.canvas.get_renderer()
        try:
            bb = txt.get_window_extent(renderer).expanded(1.1, 1.6)
        except Exception:
            return []
        ticks = ax.get_xticklabels() if which == 'x' else ax.get_yticklabels()
        hidden = []
        for tl in ticks:
            if not tl.get_visible():
                continue
            if bb.overlaps(tl.get_window_extent(renderer)):
                tl.set_visible(False)
                hidden.append(tl)
        return hidden

    # ── TextBox factory ────────────────────────────────────────────────────────
    def _tb(pos, initial):
        ax_tb = fig.add_axes(pos)
        ax_tb.set_facecolor('white')
        for sp in ax_tb.spines.values():
            sp.set_edgecolor(C_NAVY); sp.set_linewidth(0.9)
        tb = TextBox(ax_tb, '', initial=str(initial))
        tb.text_disp.set_fontfamily('serif')
        tb.text_disp.set_color(C_NAVY)
        tb.text_disp.set_fontsize(9)
        return tb

    # ── Slider ─────────────────────────────────────────────────────────────────
    ax_sl = fig.add_axes([0.09, 0.090, 0.57, 0.022])
    ax_sl.set_facecolor('white')
    for sp in ax_sl.spines.values(): sp.set_edgecolor(C_GRID)
    slider = Slider(ax_sl, 'Time  [s]', t_min, t_max,
                    valinit=t_min, color=C_NAVY)
    slider.label.set_fontfamily('serif'); slider.label.set_color(C_NAVY)
    slider.valtext.set_visible(False)

    # ── Time TextBox (right of slider) ─────────────────────────────────────────
    fig.text(0.678, 0.101, 't [s]:', ha='right', va='center',
             fontfamily='serif', fontsize=9, color=C_NAVY)
    t_box = _tb([0.682, 0.084, 0.090, 0.033], f'{t_min:.4g}')

    # ── Y-value TextBoxes (primary curve only, inline with x-axis label) ───────
    renderer = fig.canvas.get_renderer()
    fig_inv  = fig.transFigure.inverted()
    BOX_H = 0.024; BOX_W_FRAC = 0.20; Y_OFFSET = 0.058

    y_boxes = []
    for ax, p in zip(axs, panels):
        pos     = ax.get_position()
        xlbl_bb = ax.xaxis.label.get_window_extent(renderer)
        lbl_cy  = pos.y0 - Y_OFFSET
        lbl_x   = fig_inv.transform((xlbl_bb.x1, 0))[0] + 0.025
        by  = lbl_cy - BOX_H / 2
        bw  = pos.width * BOX_W_FRAC
        bx  = pos.x1 - bw - 0.006
        fig.text(lbl_x, lbl_cy, p['ylabel_short'],
                 ha='left', va='center',
                 fontfamily='serif', fontsize=8.5, color=C_NAVY, fontweight='bold')
        y_boxes.append(_tb([bx, by, bw, BOX_H], f'{p["y_arrays"][0][0]:.4g}'))

    # ── Re-entrant guard ───────────────────────────────────────────────────────
    _lock = [False]

    # ── Core update ────────────────────────────────────────────────────────────
    def update(_):
        if _lock[0]:
            return
        _lock[0] = True
        try:
            for tl in hidden_ticks:
                tl.set_visible(True)
            hidden_ticks.clear()

            t_val = slider.val
            for pi, (ax, p) in enumerate(zip(axs, panels)):
                ylim = ax.get_ylim()
                for ci in range(n_curves):
                    t_c = p['t_arrays'][ci]
                    x_c = p['x_arrays'][ci]
                    y_c = p['y_arrays'][ci]
                    idx_c = int(np.argmin(np.abs(t_c - t_val)))
                    dots[pi][ci].set_data([x_c[idx_c]], [y_c[idx_c]])

                # Primary-curve index (for crosshair and textbox)
                t_0   = p['t_arrays'][0]
                x_0   = p['x_arrays'][0]
                y_0   = p['y_arrays'][0]
                idx_0 = int(np.argmin(np.abs(t_0 - t_val)))

                if p['time_based'] and vlines[pi] is not None:
                    xi0 = x_0[idx_0]
                    vlines[pi].set_data([xi0, xi0], [ylim[0], ylim[1]])
                    xlbls[pi].set_position((xi0, ylim[0]))
                    xlbls[pi].set_text(f'{xi0:.4g}')

                y_boxes[pi].set_val(f'{y_0[idx_0]:.4g}')

            t_box.set_val(f'{t_val:.4g}')
            fig.canvas.draw()
            for pi, ax in enumerate(axs):
                if xlbls[pi] is not None:
                    hidden_ticks.extend(_hide_overlapping(ax, xlbls[pi], 'x'))
            fig.canvas.draw_idle()
        finally:
            _lock[0] = False

    # ── Callbacks ──────────────────────────────────────────────────────────────
    def on_time_submit(text):
        try:
            t_val = float(text.strip())
            slider.set_val(np.clip(t_val, t_min, t_max))
        except ValueError:
            if not _lock[0]:
                t_box.set_val(f'{slider.val:.4g}')

    def on_y_submit(text, pi):
        y_0 = panels[pi]['y_arrays'][0]
        t_0 = panels[pi]['t_arrays'][0]
        try:
            y_val = float(text.strip())
            idx   = int(np.argmin(np.abs(y_0 - y_val)))
            slider.set_val(float(t_0[idx]))
        except ValueError:
            if not _lock[0]:
                cur = int(np.argmin(np.abs(t_0 - slider.val)))
                y_boxes[pi].set_val(f'{y_0[cur]:.4g}')

    def on_click(event):
        if event.button != 1 or event.inaxes is None:
            return
        for pi, (ax, p) in enumerate(zip(axs, panels)):
            if event.inaxes is ax:
                t_0 = p['t_arrays'][0]
                x_0 = p['x_arrays'][0]
                y_0 = p['y_arrays'][0]
                if p['time_based']:
                    idx = int(np.argmin(np.abs(x_0 - event.xdata)))
                else:
                    xr  = max(float(np.ptp(x_0)), 1e-10)
                    yr  = max(float(np.ptp(y_0)), 1e-10)
                    d2  = ((x_0 - event.xdata) / xr) ** 2 + \
                          ((y_0 - event.ydata) / yr) ** 2
                    idx = int(np.argmin(d2))
                slider.set_val(float(t_0[idx]))
                break

    t_box.on_submit(on_time_submit)
    for pi, yb in enumerate(y_boxes):
        yb.on_submit(lambda text, pi=pi: on_y_submit(text, pi))
    fig.canvas.mpl_connect('button_press_event', on_click)
    slider.on_changed(update)

    update(None)
    plt.show()
