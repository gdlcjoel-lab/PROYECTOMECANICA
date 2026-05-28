"""
Task 4.4 — Propellant Mass Variation
Change propellant mass: 10 kg -> 50 kg -> 90 kg
Questions: how does acceleration change, how does burnout altitude change
"""
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from rocket_simulation import (
    simulate_rocket, compute_metrics, print_metrics,
    T0, ISP, MP, G0,
    C_NAVY, C_BLUE, C_LTBLUE,
)
from plot_utils import plot_comparison

# ── Parameters ───────────────────────────────────────────────────────────────
PROP_MASSES = [10.0, 50.0, 90.0]
LABELS      = ['m_prop = 10 kg', 'm_prop = 50 kg', 'm_prop = 90 kg']
COLORS      = [C_LTBLUE, C_BLUE, C_NAVY]
LINESTYLES  = ['-', '--', ':']

MDOT = T0 / (G0 * ISP)   # mass flow rate [kg/s] — same for all cases (fixed thrust)

# ── Run simulations ──────────────────────────────────────────────────────────
print("=" * 60)
print("  TASK 4.4 — PROPELLANT MASS VARIATION")
print("=" * 60)
print(f"  Fixed thrust  : T   = {T0:.0f} N")
print(f"  Fixed Isp     : Isp = {ISP:.0f} s")
print(f"  Mass flow rate: mdot = T/(g0*Isp) = {MDOT:.4f} kg/s  (same for all cases)")
print(f"  Payload mass  : MP  = {MP:.0f} kg")

results      = []
metrics_list = []
for mp_mass in PROP_MASSES:
    m_initial = MP + mp_mass
    t_burn    = mp_mass / MDOT
    a0        = T0 / m_initial - G0   # net initial upward acceleration
    print(f"\n>>> m_prop = {mp_mass:.0f} kg  (m0 = {m_initial:.0f} kg, "
          f"t_burn = {t_burn:.2f} s, a0 = {a0:.2f} m/s2)")
    t, r, v, drag, mass, accel = simulate_rocket(m_propellant=mp_mass)
    m_obj = compute_metrics(t, r, v)
    results.append((t, r, v, drag, mass, accel))
    metrics_list.append(m_obj)
    print_metrics(m_obj, f"m_prop = {mp_mass:.0f} kg  (m0 = {m_initial:.0f} kg)")

# ── Console answers ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 4.4 — ANSWERS")
print("=" * 60)

print("\n1. HOW DOES ACCELERATION CHANGE?")
print("   Net initial acceleration: a0 = T/m0 - g0")
print("   Smaller propellant mass -> lighter rocket -> higher initial a0.")
print("   BUT total impulse = T * t_burn is lower -> less delta-v overall.")
for mp_mass, m_obj in zip(PROP_MASSES, metrics_list):
    m_initial = MP + mp_mass
    t_burn    = mp_mass / MDOT
    a0        = T0 / m_initial - G0
    print(f"   m_prop = {mp_mass:2.0f} kg  ->  m0 = {m_initial:3.0f} kg  "
          f"a0 = {a0:6.2f} m/s2  t_burn = {t_burn:.2f} s  "
          f"z_max = {m_obj['z_max']/1000:.2f} km")

print("\n2. HOW DOES BURNOUT ALTITUDE CHANGE?")
print("   Burnout altitude = altitude at t = t_burn (when propellant is exhausted).")
for mp_mass, result in zip(PROP_MASSES, results):
    t_arr, r_arr = result[0], result[1]
    t_burn   = mp_mass / MDOT
    idx_burn = int(np.argmin(np.abs(t_arr - t_burn)))
    z_burn   = r_arr[idx_burn, 2]
    print(f"   m_prop = {mp_mass:2.0f} kg  ->  t_burn = {t_burn:.2f} s  "
          f"z_burn = {z_burn:.1f} m")
print("   More propellant -> longer burn -> higher burnout altitude,")
print("   even though initial acceleration is lower (heavier rocket).")
print("   After burnout the rocket coasts ballistically to apogee.")

# ── Annotation for panel 0 ───────────────────────────────────────────────────
ann_lines = ["Initial acc. & max altitude:"]
for mp_mass, m_obj in zip(PROP_MASSES, metrics_list):
    m_initial = MP + mp_mass
    a0        = T0 / m_initial - G0
    ann_lines.append(
        f"  {mp_mass:.0f}kg: a0={a0:.1f}m/s2  z_max={m_obj['z_max']/1000:.2f}km"
    )
annotation = "\n".join(ann_lines)

# ── Build panels ─────────────────────────────────────────────────────────────
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
        'y_arrays':     [results[i][4] for i in range(3)],
        'xlabel':       'Time [s]',
        'ylabel':       'Mass [kg]',
        'ylabel_short': 'm [kg]:',
        'title':        'Mass vs Time',
        'time_based':   True,
    },
    {
        't_arrays':     [results[i][0] for i in range(3)],
        'x_arrays':     [results[i][0] for i in range(3)],
        'y_arrays':     [results[i][5] for i in range(3)],
        'xlabel':       'Time [s]',
        'ylabel':       'Acceleration [m/s2]',
        'ylabel_short': 'a [m/s2]:',
        'title':        'Total Acceleration vs Time',
        'time_based':   True,
    },
]

plot_comparison(
    panels, LABELS, COLORS, LINESTYLES,
    suptitle='Task 4.4 — Effect of Propellant Mass on Trajectory',
    annotation=annotation,
)
