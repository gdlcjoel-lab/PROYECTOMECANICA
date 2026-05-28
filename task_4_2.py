"""
Task 4.2 — Remove Coriolis Force
Compare omega = OMEGA_EARTH (base) vs omega = 0 (Coriolis disabled)
Questions: trajectory change, which direction most affected
"""
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from rocket_simulation import (
    simulate_rocket, compute_metrics, print_metrics,
    OMEGA_EARTH, C_NAVY, C_GOLD,
)
from plot_utils import plot_comparison

# ── Parameters ───────────────────────────────────────────────────────────────
OMEGA_CASES = [OMEGA_EARTH, 0.0]
LABELS      = ['Coriolis ON (base)', 'Coriolis OFF (omega=0)']
COLORS      = [C_NAVY, C_GOLD]
LINESTYLES  = ['--', '-']

# ── Run simulations ──────────────────────────────────────────────────────────
print("=" * 60)
print("  TASK 4.2 — CORIOLIS FORCE REMOVAL")
print("=" * 60)

results      = []
metrics_list = []
for omega, label in zip(OMEGA_CASES, LABELS):
    print(f"\n>>> Running: {label}  (omega = {omega:.2e} rad/s)")
    t, r, v, drag, mass, accel = simulate_rocket(omega=omega)
    m_obj = compute_metrics(t, r, v)
    results.append((t, r, v, drag, mass, accel))
    metrics_list.append(m_obj)
    print_metrics(m_obj, label)

# ── Console answers ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 4.2 — ANSWERS")
print("=" * 60)

m_on  = metrics_list[0]
m_off = metrics_list[1]

print("\n1. HOW DOES THE TRAJECTORY CHANGE?")
print(f"   Coriolis ON:  landing x (East)  = {m_on['x_land']:+.4f} m")
print(f"   Coriolis ON:  landing y (North) = {m_on['y_land']:+.4f} m")
print(f"   Coriolis OFF: landing x (East)  = {m_off['x_land']:+.4f} m")
print(f"   Coriolis OFF: landing y (North) = {m_off['y_land']:+.4f} m")
print(f"   Horizontal range with Coriolis:    {m_on['range']:.4f} m")
print(f"   Horizontal range without Coriolis: {m_off['range']:.4f} m")
print("   The Coriolis force deflects the rocket WESTWARD (-x) during ascent.")
print("   On descent (vz < 0), it deflects EASTWARD, partially cancelling.")
print("   Net effect: small westward shift at the landing site.")

print("\n2. WHICH DIRECTION IS MOST AFFECTED?")
print("   At the equator, the Coriolis acceleration is:")
print("     a_Cx = -2 * omega * vz   [East-West component]")
print("     a_Cy =  0                [North-South — zero at equator]")
print("     a_Cz = +2 * omega * vx   [Vertical — small, since vx remains small]")
print("   -> EAST-WEST (x) direction is most affected.")
print("   -> Rising rocket (vz > 0) deflects west; descending (vz < 0) deflects east.")
print(f"   Net west deflection: {abs(m_on['x_land']):.4f} m")
print(f"   North deviation:     {abs(m_on['y_land']):.4e} m  (effectively zero)")

# ── Annotation for panel 0 ───────────────────────────────────────────────────
annotation = (
    f"Landing displacement:\n"
    f"  Coriolis ON:  x={m_on['x_land']:+.4f}m  y={m_on['y_land']:+.2e}m\n"
    f"  Coriolis OFF: x={m_off['x_land']:+.4f}m  y={m_off['y_land']:+.2e}m\n"
    f"Most affected: East-West (x)\n"
    f"  a_Cx = -2*omega*vz  (west when rising)"
)

# ── Build panels ─────────────────────────────────────────────────────────────
t0, r0 = results[0][0], results[0][1]
t1, r1 = results[1][0], results[1][1]

panels = [
    {
        't_arrays':     [t0, t1],
        'x_arrays':     [t0, t1],
        'y_arrays':     [r0[:, 2], r1[:, 2]],
        'xlabel':       'Time [s]',
        'ylabel':       'Altitude [m]',
        'ylabel_short': 'z [m]:',
        'title':        'Altitude vs Time',
        'time_based':   True,
    },
    {
        't_arrays':     [t0, t1],
        'x_arrays':     [t0, t1],
        'y_arrays':     [r0[:, 0], r1[:, 0]],
        'xlabel':       'Time [s]',
        'ylabel':       'East Displacement x [m]',
        'ylabel_short': 'x [m]:',
        'title':        'East Displacement vs Time',
        'time_based':   True,
    },
    {
        't_arrays':     [t0, t1],
        'x_arrays':     [t0, t1],
        'y_arrays':     [r0[:, 1], r1[:, 1]],
        'xlabel':       'Time [s]',
        'ylabel':       'North Displacement y [m]',
        'ylabel_short': 'y [m]:',
        'title':        'North Displacement vs Time',
        'time_based':   True,
    },
    {
        # Horizontal projection — non-time-based; click uses 2D distance
        't_arrays':     [t0, t1],
        'x_arrays':     [r0[:, 0], r1[:, 0]],
        'y_arrays':     [r0[:, 1], r1[:, 1]],
        'xlabel':       'East x [m]',
        'ylabel':       'North y [m]',
        'ylabel_short': 'y [m]:',
        'title':        'Horizontal Projection (y vs x)',
        'time_based':   False,
    },
]

plot_comparison(
    panels, LABELS, COLORS, LINESTYLES,
    suptitle='Task 4.2 — Effect of Coriolis Force on Trajectory',
    annotation=annotation,
)
