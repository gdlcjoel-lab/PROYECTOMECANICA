"""
Task 4.3 — Simplified Atmosphere
Compare ISA multi-layer model vs single-exponential rho(z) = rho0 * exp(-z/H)
Questions: max altitude, drag evolution, which model is more realistic
"""
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from rocket_simulation import (
    simulate_rocket, compute_metrics, print_metrics,
    density_isa, density_exponential,
    RHO0, H_SCALE, C_NAVY, C_GOLD,
)
from plot_utils import plot_comparison

# ── Parameters ───────────────────────────────────────────────────────────────
DENSITY_CASES = [density_isa, density_exponential]
LABELS        = ['ISA (multi-layer)', 'Exponential (approx)']
COLORS        = [C_NAVY, C_GOLD]
LINESTYLES    = ['--', '-']

# ── Run simulations ──────────────────────────────────────────────────────────
print("=" * 60)
print("  TASK 4.3 — ATMOSPHERE MODEL COMPARISON")
print("=" * 60)

results      = []
metrics_list = []
for density_func, label in zip(DENSITY_CASES, LABELS):
    print(f"\n>>> Running: {label} ...")
    t, r, v, drag, mass, accel = simulate_rocket(density_func=density_func)
    m_obj = compute_metrics(t, r, v)
    results.append((t, r, v, drag, mass, accel))
    metrics_list.append(m_obj)
    print_metrics(m_obj, label)

# ── Density comparison table ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DENSITY COMPARISON AT KEY ALTITUDES")
print("=" * 60)
print(f"  {'Altitude':>12}  {'ISA [kg/m3]':>14}  {'Exp [kg/m3]':>14}  {'Exp/ISA':>10}")
print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*10}")
for alt_km in [0, 2, 5, 8, 11, 15, 20, 30]:
    alt_m   = alt_km * 1000.0
    rho_isa = density_isa(alt_m)
    rho_exp = density_exponential(alt_m)
    ratio   = rho_exp / rho_isa
    print(f"  {alt_km:>10} km  {rho_isa:>14.4e}  {rho_exp:>14.4e}  {ratio:>10.4f}")

# ── Console answers ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 4.3 — ANSWERS")
print("=" * 60)

m_isa = metrics_list[0]
m_exp = metrics_list[1]
delta_z = m_exp['z_max'] - m_isa['z_max']

print("\n1. COMPARE MAXIMUM ALTITUDE:")
print(f"   ISA (multi-layer): z_max = {m_isa['z_max']/1000:.3f} km")
print(f"   Exponential:       z_max = {m_exp['z_max']/1000:.3f} km")
print(f"   Difference:        Delta = {delta_z/1000:+.3f} km")
if delta_z > 0:
    print("   Exponential gives higher apogee: it predicts lower density above ~5 km,")
    print("   so the rocket experiences less drag and reaches a higher altitude.")
else:
    print("   ISA gives higher apogee: the exponential model over-predicts density")
    print("   at mid-altitudes, causing more drag and a lower apogee.")

print("\n2. COMPARE DRAG EVOLUTION:")
print("   Both models agree at sea level: rho(0) = 1.225 kg/m3.")
print("   ISA has a kink at 11 km (tropopause): temperature becomes isothermal,")
print("   so density drops more slowly — the ISA is denser than exponential above ~7 km.")
print("   Exponential decreases monotonically; its derivative is everywhere continuous.")
print("   Consequence: with ISA the rocket faces more drag in the 5-15 km range.")

print("\n3. WHICH MODEL IS MORE REALISTIC?")
print("   The ISA multi-layer model is more realistic.")
print("   - It captures the temperature inversion at the tropopause (11 km).")
print("   - It models stratospheric warming correctly above 20 km.")
print("   - It matches measured density profiles to < 1% up to 85 km.")
print("   The exponential model (scale height H = 8500 m) is a useful")
print("   approximation for rough calculations, but underestimates density")
print("   in the lower stratosphere and overestimates it above ~15 km.")

# ── Annotation for panel 0 ───────────────────────────────────────────────────
annotation = (
    f"Max altitude comparison:\n"
    f"  ISA:  {m_isa['z_max']/1000:.3f} km\n"
    f"  Exp:  {m_exp['z_max']/1000:.3f} km\n"
    f"  Delta = {delta_z/1000:+.3f} km\n"
    f"rho0 = {RHO0:.3f} kg/m3   H = {H_SCALE:.0f} m"
)

# ── Density arrays along each trajectory ─────────────────────────────────────
t0, r0 = results[0][0], results[0][1]
t1, r1 = results[1][0], results[1][1]

rho_isa_arr = np.array([density_isa(max(float(z), 0.0))         for z in r0[:, 2]])
rho_exp_arr = np.array([density_exponential(max(float(z), 0.0)) for z in r1[:, 2]])

# ── Build panels ─────────────────────────────────────────────────────────────
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
        'y_arrays':     [results[0][3], results[1][3]],
        'xlabel':       'Time [s]',
        'ylabel':       'Drag Force [N]',
        'ylabel_short': 'D [N]:',
        'title':        'Aerodynamic Drag vs Time',
        'time_based':   True,
    },
    {
        't_arrays':     [t0, t1],
        'x_arrays':     [t0, t1],
        'y_arrays':     [np.linalg.norm(results[0][2], axis=1),
                         np.linalg.norm(results[1][2], axis=1)],
        'xlabel':       'Time [s]',
        'ylabel':       'Speed [m/s]',
        'ylabel_short': 'v [m/s]:',
        'title':        'Speed vs Time',
        'time_based':   True,
    },
    {
        # Density experienced along the trajectory of each respective case
        't_arrays':     [t0, t1],
        'x_arrays':     [t0, t1],
        'y_arrays':     [rho_isa_arr, rho_exp_arr],
        'xlabel':       'Time [s]',
        'ylabel':       'Air Density [kg/m3]',
        'ylabel_short': 'rho:',
        'title':        'Air Density Experienced vs Time',
        'time_based':   True,
    },
]

plot_comparison(
    panels, LABELS, COLORS, LINESTYLES,
    suptitle='Task 4.3 — ISA vs Exponential Atmosphere Model',
    annotation=annotation,
)
