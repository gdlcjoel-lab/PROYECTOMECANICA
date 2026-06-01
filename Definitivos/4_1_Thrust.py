"""
Task 4.1 — Thrust Variation
Compare T = 1000 N, 1500 N (base), 3000 N
"""
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from rocket_simulation import (
    simulate_rocket, compute_metrics, print_metrics,
    G0, ISP, MP,
    C_NAVY, C_LTBLUE,
)
from plot_utils import plot_comparison

# three thrust levels; same propellant mass in all cases
THRUST_CASES = [1000.0, 1500.0, 3000.0]
LABELS       = ['T = 1000 N', 'T = 1500 N (base)', 'T = 3000 N']
COLORS       = [C_LTBLUE, C_NAVY, '#C0392B']
LINESTYLES   = ['-', '--', ':']

PROP_MASS = 90.0   # kg — same propellant for all cases

print("=" * 60)
print("  TASK 4.1 — THRUST VARIATION")
print("=" * 60)

results      = []
metrics_list = []
for T in THRUST_CASES:
    mdot   = T / (G0 * ISP)
    t_burn = PROP_MASS / mdot
    print(f"\n>>> T = {T:.0f} N  (mdot = {mdot:.4f} kg/s, t_burn = {t_burn:.2f} s)")
    t, r, v, drag, mass, accel = simulate_rocket(thrust=T, m_propellant=PROP_MASS)
    m_obj = compute_metrics(t, r, v)
    results.append((t, r, v, drag, mass, accel))
    metrics_list.append(m_obj)
    print_metrics(m_obj, f"T = {T:.0f} N  (t_burn = {t_burn:.2f} s)")

print("\n" + "=" * 60)
print("  TASK 4.1 — ANSWERS")
print("=" * 60)

print("\n1. HOW DOES THE MAXIMUM ALTITUDE CHANGE?")
for T, m_obj in zip(THRUST_CASES, metrics_list):
    mdot   = T / (G0 * ISP)
    t_burn = PROP_MASS / mdot
    print(f"   T = {T:5.0f} N  ->  z_max = {m_obj['z_max']/1000:7.3f} km  "
          f"(t_burn = {t_burn:.2f} s)")
print()
print("   NOTE: all three cases have the same propellant mass (90 kg),")
print("   so the total impulse I = T * t_burn = Isp * g0 * m_prop is IDENTICAL.")
print("   Yet T=3000N reaches LESS altitude than T=1500N. Why?")
print("   -> Higher thrust -> higher speed at LOW altitude (dense air)")
print("      -> more aerodynamic drag loss (F_D ~ v^2)")
print("   -> Lower thrust -> slower ascent -> more time fighting gravity (gravity loss)")
print("   There is an optimal thrust. For this rocket, T=1500N is near-optimal;")
print("   T=3000N wastes energy to drag; T=1000N wastes energy to gravity.")

print("\n2. DOES THE ROCKET ESCAPE THE DENSE ATMOSPHERE (troposphere > 11 km)?")
for T, m_obj in zip(THRUST_CASES, metrics_list):
    esc = "YES" if m_obj['z_max'] > 11000 else "NO "
    print(f"   T = {T:5.0f} N  ->  z_max = {m_obj['z_max']/1000:.3f} km  ->  {esc}")

print("\n3. HOW DOES BURNOUT TIME VARY WITH THRUST?")
print("   t_burn = m_prop / mdot = m_prop * g0 * Isp / T")
print("   -> Higher thrust means shorter burnout time (more propellant consumed per second).")
for T in THRUST_CASES:
    mdot   = T / (G0 * ISP)
    t_burn = PROP_MASS / mdot
    print(f"   T = {T:5.0f} N  ->  t_burn = {t_burn:.2f} s  (mdot = {mdot:.4f} kg/s)")

# text box shown in the first plot panel
ann_lines = ["Max altitude & burnout time:"]
for T, m_obj in zip(THRUST_CASES, metrics_list):
    mdot   = T / (G0 * ISP)
    t_burn = PROP_MASS / mdot
    ann_lines.append(f"  T={T:.0f}N: z_max={m_obj['z_max']/1000:.2f}km  t_b={t_burn:.1f}s")
ann_lines.append("Troposphere limit: 11 km")
annotation = "\n".join(ann_lines)

# four panels: altitude, speed, drag, mass
panels = [
    {
        't_arrays':     [results[i][0] for i in range(3)],
        'x_arrays':     [results[i][0] for i in range(3)],
        'y_arrays':     [results[i][1][:, 2] for i in range(3)],
        'xlabel':       'Time [s]',
        'ylabel':       'Altitude [m]',
        'ylabel_short': 'z [m]:',
        'title':        'Altitude vs Time',
        'time_based':   True,
    },
    {
        't_arrays':     [results[i][0] for i in range(3)],
        'x_arrays':     [results[i][0] for i in range(3)],
        'y_arrays':     [np.linalg.norm(results[i][2], axis=1) for i in range(3)],
        'xlabel':       'Time [s]',
        'ylabel':       'Speed [m/s]',
        'ylabel_short': 'v [m/s]:',
        'title':        'Speed vs Time',
        'time_based':   True,
    },
    {
        't_arrays':     [results[i][0] for i in range(3)],
        'x_arrays':     [results[i][0] for i in range(3)],
        'y_arrays':     [results[i][3] for i in range(3)],
        'xlabel':       'Time [s]',
        'ylabel':       'Drag Force [N]',
        'ylabel_short': 'D [N]:',
        'title':        'Aerodynamic Drag vs Time',
        'time_based':   True,
    },
    {
        't_arrays':     [results[i][0] for i in range(3)],
        'x_arrays':     [results[i][0] for i in range(3)],
        'y_arrays':     [results[i][4] for i in range(3)],
        'xlabel':       'Time [s]',
        'ylabel':       'Mass [kg]',
        'ylabel_short': 'm [kg]:',
        'title':        'Mass vs Time',
        'time_based':   True,
    },
]

plot_comparison(
    panels, LABELS, COLORS, LINESTYLES,
    suptitle='Task 4.1 — Effect of Thrust on Rocket Trajectory',
    annotation=annotation,
)