"""
bonus_plot.py — Interactive plots for the 3-stage mission.

Produces two figures:
  Figure 1 (4-panel timeline):  altitude, speed, mass, drag-free indicator
  Figure 2 (orbital view):      2-D ECI x-y plane showing trajectory + ISS orbit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

# Colour palette (consistent with rocket_simulation.py)
C_NAVY   = '#0D2B55'
C_BLUE   = '#2A6496'
C_LTBLUE = '#5B9BD5'
C_GOLD   = '#B07D2A'
C_RED    = '#C0392B'
C_GREEN  = '#27AE60'
C_GRID   = '#D0D8E4'

# Phase colour mapping
PHASE_COLORS = {
    1: C_LTBLUE,   # Stage 1
    2: C_BLUE,     # Stage 2
    3: C_NAVY,     # Coast (suborbital)
    4: C_GOLD,     # Circularisation
    5: '#8E44AD',  # Hohmann transfer
    6: C_GREEN,    # ISS orbit / Rendezvous
}
PHASE_NAMES = {
    1: 'Stage 1 burn',
    2: 'Stage 2 burn',
    3: 'Coast',
    4: 'Circularisation',
    5: 'Hohmann transfer',
    6: 'ISS orbit / Rendezvous',
}


def _style_fig(fig):
    fig.patch.set_facecolor('white')


def _style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor('#F8FAFD')
    ax.tick_params(labelsize=9, colors=C_NAVY)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(C_GRID)
    ax.grid(True, color=C_GRID, linewidth=0.7, linestyle='--')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=C_NAVY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=C_NAVY)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', color=C_NAVY)


def _phase_segments(t, y, phase):
    """
    Split y-array into coloured segments by phase for multi-colour line plots.
    Returns list of (t_seg, y_seg, color) tuples.
    """
    segments = []
    phases   = np.unique(phase)
    for p in phases:
        mask = phase == p
        # Find contiguous blocks
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            continue
        # Split at gaps
        gaps = np.where(np.diff(idx) > 1)[0] + 1
        blocks = np.split(idx, gaps)
        for blk in blocks:
            if len(blk) < 2:
                continue
            # include one point before block for continuity
            start = max(blk[0] - 1, 0)
            sl    = slice(start, blk[-1] + 1)
            segments.append((t[sl], y[sl], PHASE_COLORS[p]))
    return segments


def plot_mission(data):
    """
    4-panel interactive figure:
      [0,0] Altitude [km] vs time
      [0,1] Speed [m/s] vs time
      [1,0] Mass [kg] vs time
      [1,1] Altitude [km] vs speed [m/s]  (phase portrait)
    """
    t     = data['t']
    h     = data['h']           # km
    speed = data['speed']       # m/s
    m     = data['m']           # kg
    phase = data['phase']
    events = data['events']

    fig = plt.figure(figsize=(14, 11))
    _style_fig(fig)
    fig.suptitle('3-Stage Rocket Mission — Timeline', fontsize=13,
                 fontweight='bold', color=C_NAVY, y=0.97)

    gs = fig.add_gridspec(2, 2,
                          left=0.09, right=0.96,
                          bottom=0.22, top=0.93,
                          hspace=0.48, wspace=0.38)

    ax_h   = fig.add_subplot(gs[0, 0])
    ax_v   = fig.add_subplot(gs[0, 1])
    ax_m   = fig.add_subplot(gs[1, 0])
    ax_hv  = fig.add_subplot(gs[1, 1])

    _style_ax(ax_h,  't [s]', 'Altitude [km]',    'Altitude vs Time')
    _style_ax(ax_v,  't [s]', 'Speed [m/s]',       'Speed vs Time')
    _style_ax(ax_m,  't [s]', 'Mass [kg]',         'Mass vs Time')
    _style_ax(ax_hv, 'Speed [m/s]', 'Altitude [km]', 'Altitude vs Speed')

    # ── Coloured phase segments ────────────────────────────────────────────
    for ax, ydata in [(ax_h, h), (ax_v, speed), (ax_m, m)]:
        segs = _phase_segments(t, ydata, phase)
        for ts, ys, col in segs:
            ax.plot(ts, ys, color=col, linewidth=1.5)

    # Phase portrait (not time-based)
    segs_hv = _phase_segments(speed, h, phase)
    for xs, ys, col in segs_hv:
        ax_hv.plot(xs, ys, color=col, linewidth=1.5)

    # ── ISS altitude reference ─────────────────────────────────────────────
    from bonus_multistage import H_ISS
    h_iss_km = H_ISS / 1e3
    for ax in [ax_h]:
        ax.axhline(h_iss_km, color=C_GREEN, linewidth=1.0, linestyle=':',
                   label=f'ISS ({h_iss_km:.0f} km)')
        ax.legend(fontsize=8, loc='upper left')

    # ── Event markers ─────────────────────────────────────────────────────
    for ev_t, ev_label in events:
        for ax, ydata in [(ax_h, h), (ax_v, speed), (ax_m, m)]:
            idx = np.searchsorted(t, ev_t)
            idx = min(idx, len(t) - 1)
            ax.axvline(ev_t, color='#999999', linewidth=0.7, linestyle=':')

    # ── Phase legend (shared) ─────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=PHASE_COLORS[p], label=PHASE_NAMES[p])
        for p in sorted(PHASE_COLORS)
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=5, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    # ── Separator line above legend ───────────────────────────────────────
    fig.add_artist(Line2D([0.04, 0.96], [0.13, 0.13],
                          transform=fig.transFigure,
                          color=C_GRID, linewidth=1.0))

    # ── Interactive slider ────────────────────────────────────────────────
    ax_sl = fig.add_axes([0.09, 0.090, 0.57, 0.022])
    slider = Slider(ax_sl, '', t[0], t[-1], valinit=t[0],
                    color=C_LTBLUE, track_color=C_GRID)
    slider.valtext.set_visible(False)

    fig.text(0.68, 0.101, 't [s]:', fontsize=9, color=C_NAVY,
             ha='right', va='center')

    # Dots for each time-based panel
    dots = {}
    for ax, ydata, key in [(ax_h, h, 'h'), (ax_v, speed, 'v'), (ax_m, m, 'm')]:
        d, = ax.plot([], [], 'o', color=C_NAVY, markersize=5, zorder=5)
        dots[key] = (d, ydata)

    _lock = [False]

    def update(val):
        if _lock[0]:
            return
        _lock[0] = True
        ti = float(slider.val)
        idx = np.searchsorted(t, ti)
        idx = min(idx, len(t) - 1)
        for key, (dot, ydata) in dots.items():
            dot.set_data([t[idx]], [ydata[idx]])
        fig.canvas.draw_idle()
        _lock[0] = False

    slider.on_changed(update)
    update(t[0])

    plt.show(block=False)
    return fig


def plot_orbit(data):
    """
    Figure 2: 2-D ECI orbital view (x-y equatorial plane).
    Shows rocket trajectory, Earth disc, and ISS target orbit.
    """
    from bonus_multistage import R_EARTH, H_ISS, GM

    r     = data['r']
    phase = data['phase']

    fig, ax = plt.subplots(figsize=(8, 8))
    _style_fig(fig)
    _style_ax(ax, 'ECI x [km]', 'ECI y [km]', '3-Stage Trajectory — Orbital View (ECI)')
    fig.suptitle('Orbital View', fontsize=13, fontweight='bold', color=C_NAVY, y=0.97)

    # Earth disc
    theta_earth = np.linspace(0, 2 * np.pi, 360)
    ax.fill(R_EARTH / 1e3 * np.cos(theta_earth),
            R_EARTH / 1e3 * np.sin(theta_earth),
            color='#DDEEFF', zorder=1)
    ax.plot(R_EARTH / 1e3 * np.cos(theta_earth),
            R_EARTH / 1e3 * np.sin(theta_earth),
            color=C_NAVY, linewidth=1.0, zorder=2, label='Earth surface')

    # ISS circular orbit
    r_iss = (R_EARTH + H_ISS) / 1e3   # km
    ax.plot(r_iss * np.cos(theta_earth),
            r_iss * np.sin(theta_earth),
            color=C_GREEN, linewidth=1.2, linestyle='--', zorder=3, label='ISS orbit (408 km)')

    # Trajectory, coloured by phase
    x_km = r[:, 0] / 1e3
    y_km = r[:, 1] / 1e3
    t_dummy = np.arange(len(x_km), dtype=float)
    segs = _phase_segments(t_dummy, y_km, phase)
    # re-plot as x-y using stored phase mask
    phases_uniq = np.unique(phase)
    for p in phases_uniq:
        mask = phase == p
        ax.plot(x_km[mask], y_km[mask], color=PHASE_COLORS[p],
                linewidth=1.4, label=PHASE_NAMES[p], zorder=4)

    # Launch marker
    ax.plot(R_EARTH / 1e3, 0, '*', color=C_GOLD, markersize=12,
            zorder=5, label='Launch site (equator)')

    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.show(block=False)
    return fig
