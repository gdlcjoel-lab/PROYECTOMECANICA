"""
================================================================================
 3D ROCKET TRAJECTORY SIMULATION — Euler Integration
 Classical Mechanics — Project 2

 Authors: Rubén Praena, Claudia Ros, Antoni Ruiz, Miguel Sanz,
          Elishka Mrázek, Joel García, Giulia Latorre
================================================================================

 COORDINATE SYSTEM  (local inertial frame fixed to Earth's surface at equator)
   x = East,  y = North,  z = Up (radial)
   ω⃗ = ω ŷ   (Earth's rotation axis points North at the equator)

 FORCES ACTING ON THE ROCKET
   (1) Gravity     : g(z) = −GM / (R + z)²  ẑ    [variable with altitude]
   (2) Drag        : F⃗_D = −½ Cd A ρ(z) |v⃗| v⃗   [opposes velocity]
   (3) Thrust      : F⃗_T = T ẑ               [only while propellant remains]
   (4) Coriolis    : a⃗_C = −2 ω⃗ × v⃗         [rotating-frame pseudo-force]

 EULER INTEGRATION SCHEME
   r⃗_{n+1} = r⃗_n + v⃗_n · Δt          (position updated with OLD velocity)
   v⃗_{n+1} = v⃗_n + a⃗_n · Δt          (velocity updated with acceleration at t_n)
   m_{n+1} = m_n  − ṁ · Δt           (mass decreases at constant rate during burn)

 FILE STRUCTURE
   Section 1 — Physical constants
   Section 2 — Default rocket parameters
   Section 3 — Atmospheric density models (ISA + exponential)
   Section 4 — Euler simulation core
   Section 5 — Post-processing utilities
   Section 6 — Plotting functions
   Section 7 — Main execution (base case + 4 additional tasks)
================================================================================
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Force UTF-8 on stdout/stderr so matplotlib font warnings don't crash on Windows
# (the default Windows console uses cp1252 which cannot encode characters like em-dash)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ================================================================================
# SECTION 1: PHYSICAL AND EARTH CONSTANTS
# ================================================================================

GM          = 3.99e14     # m³/s²   — Earth's gravitational parameter
R_EARTH     = 6.378e6    # m       — Earth's mean radius
G0          = 9.80665    # m/s²    — standard gravity at surface (used for mass flow rate)
OMEGA_EARTH = 7.27e-5    # rad/s   — Earth's angular velocity


# ================================================================================
# SECTION 2: DEFAULT ROCKET PARAMETERS
# ================================================================================

M0   = 100.0   # kg   — initial total mass  (structure + propellant + payload)
MP   = 10.0    # kg   — payload/structural mass that remains after burnout
T0   = 1500.0  # N    — nominal thrust
ISP  = 100.0   # s    — specific impulse  (low value → low fuel efficiency)
CD   = 0.5     # —    — drag coefficient  (dimensionless)
AREF = 0.1     # m²   — rocket cross-sectional reference area


# ================================================================================
# SECTION 3: ATMOSPHERIC DENSITY MODELS
# ================================================================================
#
#  Two density models are provided:
#    a) density_isa(z)          — International Standard Atmosphere (ISA)
#                                 multi-layer model. Used by default.
#    b) density_exponential(z)  — simplified single-exponential model.
#                                 Used in Task 3 for comparison.
#
#  The ISA model is adapted from the reference file air_density.py
#  (that file is NOT modified). Here the function is simplified to return
#  a scalar ρ [kg/m³] instead of a dictionary.
#
#  FUTURE IMPROVEMENT: at each layer boundary the ISA profile has a kink
#  (the lapse rate changes abruptly), causing a discontinuous derivative
#  in ρ(z). A smoother model can be built by applying cubic-spline
#  interpolation across those boundary altitudes.
#

# ISA atmospheric constants
RHO0    = 1.225    # kg/m³     — sea-level density
H_SCALE = 8500.0   # m         — exponential scale height (Task 3)
T0_ISA  = 288.15   # K         — sea-level temperature
L_TROP  = 0.0065   # K/m       — troposphere lapse rate (temperature drop per metre)
R_AIR   = 287.05   # J/(kg·K)  — specific gas constant for dry air


def density_isa(z):
    """
    ISA air density [kg/m³] at geometric altitude z [m] above sea level.

    Each atmospheric layer has a reference pressure at its base and either
    a constant temperature (isothermal) or a linear lapse rate. Density is
    obtained from the ideal gas law:  ρ = p / (R_air · T).

    Layers:
      0 – 11 km   Troposphere           linear cooling  (−6.5 K/km)
      11 – 20 km  Lower stratosphere    isothermal      (216.65 K)
      20 – 32 km  Upper stratosphere    linear warming  (+1 K/km)
      32 – 47 km  Lower mesosphere      linear warming  (+2.8 K/km)
      47 – 51 km  Upper mesosphere      isothermal      (270.65 K)
      51 – 71 km  Lower thermosphere    linear warming  (+2.8 K/km)
      71 – 85 km  Upper thermosphere    linear warming  (+2.0 K/km)
      > 85 km     Exponential fallback  (very low density regime)

    Adapted from air_density.py — original file left unmodified.
    """
    z_km = z / 1000.0

    if z_km < 11:                              # Troposphere
        T = T0_ISA - L_TROP * z
        p = 101325.0 * (T / T0_ISA) ** (G0 / (L_TROP * R_AIR))

    elif z_km < 20:                            # Lower stratosphere (isothermal)
        T = 216.65
        p = 22632.1 * np.exp(-G0 * (z - 11000) / (R_AIR * T))

    elif z_km < 32:                            # Upper stratosphere
        T     = 216.65 + 0.001 * (z - 20000)
        T_avg = (216.65 + T) / 2.0            # average T over the layer (for integration)
        p     = 5474.89 * np.exp(-G0 * (z - 20000) / (R_AIR * T_avg))

    elif z_km < 47:                            # Lower mesosphere
        T     = 228.65 + 0.0028 * (z - 32000)
        T_avg = (228.65 + T) / 2.0
        p     = 868.02 * np.exp(-G0 * (z - 32000) / (R_AIR * T_avg))

    elif z_km < 51:                            # Upper mesosphere (isothermal)
        T = 270.65
        p = 110.91 * np.exp(-G0 * (z - 47000) / (R_AIR * T))

    elif z_km < 71:                            # Lower thermosphere
        T     = 270.65 + 0.0028 * (z - 51000)
        T_avg = (270.65 + T) / 2.0
        p     = 66.939 * np.exp(-G0 * (z - 51000) / (R_AIR * T_avg))

    elif z_km < 85:                            # Upper thermosphere
        T     = 270.65 + 0.002 * (z - 71000)
        T_avg = (270.65 + T) / 2.0
        p     = 3.9564 * np.exp(-G0 * (z - 71000) / (R_AIR * T_avg))

    else:                                      # Above 85 km: exponential falloff
        return max(1.5e-5 * np.exp(-(z - 85000) / 6000.0), 1e-20)

    rho = p / (R_AIR * T)
    return max(rho, 1e-20)                     # never return exactly zero


def density_exponential(z):
    """
    Simplified exponential atmosphere (used in Task 3):

        ρ(z) = ρ₀ · exp(−z / H)

    where  ρ₀ = 1.225 kg/m³  (sea-level density)
    and    H  = 8500 m        (constant scale height).

    This is a single-layer approximation that is easier to analyse
    analytically but less accurate than the ISA at high altitudes.
    """
    return RHO0 * np.exp(-z / H_SCALE)


# ================================================================================
# SECTION 4: EULER INTEGRATION — CORE SIMULATION
# ================================================================================

def simulate_rocket(
    thrust        = T0,
    omega         = OMEGA_EARTH,
    density_func  = density_isa,
    m_propellant  = M0 - MP,    # kg — propellant mass (default: 90 kg)
    dt            = 0.1,         # s  — Euler time step
):
    """
    Simulate the 3D rocket trajectory using explicit Euler integration.

    The rocket is launched vertically from rest at the Earth's equator
    (origin of the coordinate system). Simulation ends when the rocket
    returns to z = 0 (ground impact).

    Parameters
    ----------
    thrust        : float     — engine thrust [N]
    omega         : float     — Earth angular velocity [rad/s].
                                Set to 0 to disable the Coriolis effect.
    density_func  : callable  — atmospheric density model: ρ(z) → [kg/m³].
                                Pass density_isa or density_exponential.
    m_propellant  : float     — propellant mass loaded at launch [kg].
                                Total initial mass = MP + m_propellant.
    dt            : float     — integration time step [s].

    Returns
    -------
    t    : ndarray (N,)   — time array [s]
    r    : ndarray (N,3)  — position  [m],   columns → (x_East, y_North, z_Up)
    v    : ndarray (N,3)  — velocity  [m/s], columns → (vx, vy, vz)
    drag : ndarray (N,)   — aerodynamic drag magnitude [N]
    mass : ndarray (N,)   — rocket mass at each step [kg]
    accel: ndarray (N,)   — total acceleration magnitude [m/s²]
    """

    # --- derived burn parameters ---
    m_initial = MP + m_propellant              # total initial mass [kg]
    mdot      = thrust / (G0 * ISP)           # mass flow rate  ṁ = T / (g₀ · Isp)  [kg/s]
    t_burn    = m_propellant / mdot            # burn-out time  [s]

    # --- Earth's rotation vector: ω⃗ = ω ŷ at the equator ---
    #     (the rotation axis points toward geographic North = +y direction)
    omega_vec = np.array([0.0, omega, 0.0])

    # --- initial state ---
    r = np.array([0.0, 0.0, 0.0])    # position [m]
    v = np.array([0.0, 0.0, 0.0])    # velocity [m/s]
    m = m_initial                      # current mass [kg]
    t = 0.0                            # elapsed time [s]

    # --- history lists (converted to arrays on return) ---
    t_arr     = [t]
    r_arr     = [r.copy()]
    v_arr     = [v.copy()]
    drag_arr  = [0.0]
    mass_arr  = [m]
    accel_arr = [0.0]

    launched = False    # becomes True once z > 0 for the first time

    # =====================================================================
    # EULER LOOP
    # =====================================================================
    while True:
        z = r[2]   # current altitude [m]

        # --- termination: rocket has hit the ground after liftoff ---
        if launched and z < 0.0:
            break

        # Clamp altitude for physics evaluation (density/gravity below ground = sea level)
        z_eff = max(z, 0.0)

        # -----------------------------------------------------------------
        # Force 1: Gravity
        #   g(z) = −GM / (R + z)²   (magnitude decreases with altitude)
        # -----------------------------------------------------------------
        g_mag  = GM / (R_EARTH + z_eff) ** 2
        a_grav = np.array([0.0, 0.0, -g_mag])

        # -----------------------------------------------------------------
        # Force 2: Thrust  (vertical, upward, active only while t < t_burn)
        # -----------------------------------------------------------------
        if t < t_burn:
            a_thrust = np.array([0.0, 0.0, thrust / m])
            m_next   = max(m - mdot * dt, MP)   # clamp so mass never drops below MP
        else:
            a_thrust = np.zeros(3)
            m_next   = MP

        # -----------------------------------------------------------------
        # Force 3: Aerodynamic Drag
        #   F⃗_D = −½ Cd A ρ(z) |v⃗| v⃗   (always opposes velocity vector)
        # -----------------------------------------------------------------
        rho   = density_func(z_eff)
        v_mag = np.linalg.norm(v)

        if v_mag > 0.0:
            F_drag = -0.5 * CD * AREF * rho * v_mag * v
        else:
            F_drag = np.zeros(3)

        a_drag = F_drag / m

        # -----------------------------------------------------------------
        # Force 4: Coriolis Acceleration  (rotating-frame pseudo-force)
        #   a⃗_C = −2 (ω⃗ × v⃗)
        #
        #   With ω⃗ = ω ŷ  and  v⃗ = (vx, vy, vz):
        #       ω⃗ × v⃗ = (ω·vz,  0,  −ω·vx)
        #   →  a⃗_C   = (−2ω·vz,  0,  +2ω·vx)
        #
        #   Physical interpretation at equator:
        #     A rocket going UP  (vz > 0) is deflected WEST  (−x direction).
        #     A rocket going EAST(vx > 0) is deflected UP/DOWN (+z).
        # -----------------------------------------------------------------
        a_coriolis = -2.0 * np.cross(omega_vec, v)

        # -----------------------------------------------------------------
        # Total acceleration — Newton's 2nd law in the rotating frame
        # -----------------------------------------------------------------
        a = a_grav + a_thrust + a_drag + a_coriolis

        # -----------------------------------------------------------------
        # Explicit Euler step
        #   r uses the OLD velocity (before the velocity update)
        # -----------------------------------------------------------------
        r_new = r + v * dt          # position updated with v at time t_n
        v_new = v + a * dt          # velocity updated with a at time t_n

        # --- advance state ---
        r = r_new
        v = v_new
        m = m_next
        t += dt

        # --- detect liftoff ---
        if not launched and r[2] > 0.0:
            launched = True

        # --- store step ---
        t_arr.append(t)
        r_arr.append(r.copy())
        v_arr.append(v.copy())
        drag_arr.append(np.linalg.norm(F_drag))
        mass_arr.append(m)
        accel_arr.append(np.linalg.norm(a))

    return (
        np.array(t_arr),
        np.array(r_arr),
        np.array(v_arr),
        np.array(drag_arr),
        np.array(mass_arr),
        np.array(accel_arr),
    )


# ================================================================================
# SECTION 5: POST-PROCESSING UTILITIES
# ================================================================================

def compute_metrics(t, r, v):
    """
    Extract key flight performance metrics from simulation arrays.

    Parameters
    ----------
    t : (N,)   time array [s]
    r : (N,3)  position array [m]
    v : (N,3)  velocity array [m/s]

    Returns
    -------
    dict with keys:
      'z_max'     — maximum altitude reached [m]
      't_zmax'    — time at which z_max is reached [s]
      't_flight'  — total flight duration [s]
      'x_land'    — East displacement at landing [m]
      'y_land'    — North displacement at landing [m]
      'range'     — total horizontal distance from launch [m]
      'v_max'     — maximum speed reached [m/s]
    """
    z     = r[:, 2]
    speed = np.linalg.norm(v, axis=1)

    idx_max   = int(np.argmax(z))
    z_max     = float(z[idx_max])
    t_zmax    = float(t[idx_max])
    t_flight  = float(t[-1])
    x_land    = float(r[-1, 0])
    y_land    = float(r[-1, 1])
    range_val = float(np.sqrt(x_land**2 + y_land**2))
    v_max     = float(np.max(speed))

    return {
        'z_max'    : z_max,
        't_zmax'   : t_zmax,
        't_flight' : t_flight,
        'x_land'   : x_land,
        'y_land'   : y_land,
        'range'    : range_val,
        'v_max'    : v_max,
    }


def print_metrics(metrics, label=""):
    """Print flight metrics in a formatted block."""
    sep = "=" * 55
    print(f"\n{sep}")
    if label:
        print(f"  {label}")
        print(sep)
    print(f"  Max altitude       : {metrics['z_max']:>10.2f} m  "
          f"({metrics['z_max']/1000:.3f} km)")
    print(f"  Time at apogee     : {metrics['t_zmax']:>10.2f} s")
    print(f"  Max speed          : {metrics['v_max']:>10.2f} m/s")
    print(f"  Total flight time  : {metrics['t_flight']:>10.2f} s")
    print(f"  Landing  x (East)  : {metrics['x_land']:>10.4f} m")
    print(f"  Landing  y (North) : {metrics['y_land']:>10.4f} m")
    print(f"  Horizontal range   : {metrics['range']:>10.4f} m")
    print(sep)


# ================================================================================
# SECTION 6: PLOTTING STYLE AND FUNCTIONS
# ================================================================================

# ─── Colour palette — white + navy blue theme ────────────────────────────────
C_NAVY   = '#0D2B55'   # deep navy  — primary series / all labels and spines
C_BLUE   = '#2A6496'   # mid blue   — secondary series in comparison plots
C_LTBLUE = '#5B9BD5'   # light blue — tertiary series in comparison plots
C_GOLD   = '#B07D2A'   # amber gold — accent used in two-series comparisons
C_GRID   = '#D0D8E4'   # pale blue-grey — grid lines

# ─── Global rcParams — applied once at import time ───────────────────────────
plt.rcParams.update({
    # Typography
    'font.family'           : 'serif',
    'font.serif'            : ['Times New Roman', 'DejaVu Serif'],
    'font.size'             : 11,
    'axes.titlesize'        : 12,
    'axes.titleweight'      : 'bold',
    'axes.labelsize'        : 11,
    'axes.titlecolor'       : C_NAVY,
    'axes.labelcolor'       : C_NAVY,
    # Axes frame and background
    'axes.linewidth'        : 1.2,
    'axes.edgecolor'        : C_NAVY,
    'axes.facecolor'        : 'white',
    'figure.facecolor'      : 'white',
    # Grid
    'axes.grid'             : True,
    'grid.color'            : C_GRID,
    'grid.linewidth'        : 0.7,
    'grid.alpha'            : 1.0,
    # Tick marks (inward, navy coloured)
    'xtick.color'           : C_NAVY,
    'ytick.color'           : C_NAVY,
    'xtick.labelsize'       : 10,
    'ytick.labelsize'       : 10,
    'xtick.direction'       : 'in',
    'ytick.direction'       : 'in',
    'xtick.major.width'     : 1.0,
    'ytick.major.width'     : 1.0,
    # Legend
    'legend.framealpha'     : 0.95,
    'legend.edgecolor'      : C_GRID,
    'legend.fontsize'       : 10,
    'legend.title_fontsize' : 10,
    # Lines
    'lines.linewidth'       : 2.0,
    # Output resolution
    'figure.dpi'            : 110,
})


def _style_fig(fig, title):
    """Uniform suptitle style: bold, navy, slightly above the axes."""
    fig.suptitle(title, fontsize=14, fontweight='bold', color=C_NAVY)


def _style_ax(ax, title, xlabel, ylabel):
    """Axis-level styling: title, labels, and spine colours."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_NAVY)
        spine.set_linewidth(1.2)
    ax.tick_params(colors=C_NAVY, which='both')


def plot_base_results(t, r, v, drag, title="Base Case"):
    """
    Interactive four-panel figure.  Three ways to place the marker:

      1. Slider    — drag to any time instant.
      2. Click     — click anywhere on a panel; marker jumps to nearest curve point.
      3. Text box  — type a value and press Enter:
           • Right of slider  → time [s]  (same as dragging the slider).
           • Below each panel → y-axis value; finds the closest point on the curve.

    All three inputs stay synchronised at all times.
    Tick labels that overlap with floating value labels are temporarily hidden.
      • A floating text label at the x-axis showing x★ (4 sig. figs.).
      • A floating text label at the y-axis showing y★ (4 sig. figs.).
      • Any fixed tick label that overlaps a floating label is hidden while
        it would overlap, then restored automatically when it no longer does.

    Panel 2 (horizontal projection) uses position axes (East, North);
    for a click on panel 2 the nearest curve point in normalised 2-D space is used.
    """
    from matplotlib.widgets import TextBox
    from matplotlib.lines  import Line2D

    z     = r[:, 2] / 1000.0
    xp    = r[:, 0]
    yp    = r[:, 1]
    speed = np.linalg.norm(v, axis=1)

    # (xdata, ydata, x-label, y-label, panel title, y-box label)
    pdata = [
        (t,  z,     "Time [s]",      "Altitude [km]",  "Altitude vs Time",               "Altitude [km]"),
        (xp, yp,    "x — East [m]", "y — North [m]", "Horizontal Projection (y vs x)", "y North [m]"),
        (t,  speed, "Time [s]",      "Speed [m/s]",    "Speed vs Time",                  "Speed [m/s]"),
        (t,  drag,  "Time [s]",      "Drag force [N]", "Aerodynamic Drag vs Time",       "Drag [N]"),
    ]
    time_based = [True, False, True, True]   # True  → x-axis is time; find by x-coord
                                              # False → 2-D curve;      find by distance

    # ── Figure layout ──────────────────────────────────────────────────────────
    # bottom=0.22 only needs to clear the slider; text boxes live inline with
    # the x-axis labels so no extra vertical space is needed below the panels.
    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(2, 2,
                           left=0.09, right=0.96,
                           bottom=0.22, top=0.93,
                           hspace=0.50, wspace=0.40)
    axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    _style_fig(fig, title)

    # ── Static curves ─────────────────────────────────────────────────────────
    for ax, (xd, yd, xl, yl, ttl, _) in zip(axs, pdata):
        ax.plot(xd, yd, color=C_NAVY, zorder=2)
        _style_ax(ax, ttl, xl, yl)
    axs[1].set_aspect('equal', adjustable='datalim')
    fig.canvas.draw()    # finalise axis limits before placing dynamic elements

    # ── Thin separator line above the slider zone ─────────────────────────────
    fig.add_artist(Line2D([0.04, 0.96], [0.128, 0.128],
                           transform=fig.transFigure,
                           color=C_GRID, linewidth=0.9, zorder=0))

    # ── Dynamic overlay (dot + crosshair + floating value labels) ─────────────
    CH_KW = dict(color='#B8B8B8', linewidth=0.8, linestyle='-', zorder=3, clip_on=True)
    LBL_X = dict(fontsize=8.5, color='#222222', clip_on=False, fontfamily='serif',
                 ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.92))
    LBL_Y = dict(fontsize=8.5, color='#222222', clip_on=False, fontfamily='serif',
                 ha='right', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.92))

    dots, vls, hls, xlbls, ylbls = [], [], [], [], []
    for ax, (xd, yd, *_) in zip(axs, pdata):
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        xi, yi = xd[0], yd[0]
        dot, = ax.plot([xi], [yi], 'o', color=C_NAVY, markersize=6, zorder=6)
        vl,  = ax.plot([xi, xi],      [ylim[0], yi], **CH_KW)
        hl,  = ax.plot([xlim[0], xi], [yi,      yi], **CH_KW)
        lx   = ax.text(xi,      ylim[0], f'{xi:.4g}', **LBL_X)
        ly   = ax.text(xlim[0], yi,      f'{yi:.4g}', **LBL_Y)
        dots.append(dot); vls.append(vl); hls.append(hl)
        xlbls.append(lx); ylbls.append(ly)

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
            if not tl.get_visible(): continue
            if bb.overlaps(tl.get_window_extent(renderer)):
                tl.set_visible(False); hidden.append(tl)
        return hidden

    # ── TextBox factory (styled to match the navy theme) ──────────────────────
    def _textbox(pos, initial):
        ax_tb = fig.add_axes(pos)
        ax_tb.set_facecolor('white')
        for sp in ax_tb.spines.values():
            sp.set_edgecolor(C_NAVY); sp.set_linewidth(0.9)
        tb = TextBox(ax_tb, '', initial=str(initial))
        tb.text_disp.set_fontfamily('serif')
        tb.text_disp.set_color(C_NAVY)
        tb.text_disp.set_fontsize(9)
        return tb

    # ── Slider ────────────────────────────────────────────────────────────────
    ax_sl = fig.add_axes([0.09, 0.090, 0.57, 0.022])
    ax_sl.set_facecolor('white')
    for sp in ax_sl.spines.values(): sp.set_edgecolor(C_GRID)
    slider = Slider(ax_sl, 'Time  [s]', float(t[0]), float(t[-1]),
                    valinit=float(t[0]), color=C_NAVY)
    slider.label.set_fontfamily('serif'); slider.label.set_color(C_NAVY)
    slider.valtext.set_visible(False)   # value shown in the TextBox instead

    # ── Time TextBox  (right of the slider) ───────────────────────────────────
    fig.text(0.678, 0.101, 't [s]:', ha='right', va='center',
             fontfamily='serif', fontsize=9, color=C_NAVY)
    t_box = _textbox([0.682, 0.084, 0.090, 0.033], f'{t[0]:.4g}')

    # ── Y-value TextBoxes  (inline with the x-axis label, on the right side) ───
    # Use a fixed offset of 0.058 below the panel frame bottom (pos.y0).
    # This is DPI-independent in figure-fraction coordinates and consistently
    # clears tick labels + the xlabel regardless of the rendering backend.
    # Horizontal positions come from the renderer (xlabel right edge) so the
    # quantity label never collides with the matplotlib xlabel.
    renderer = fig.canvas.get_renderer()
    fig_inv  = fig.transFigure.inverted()
    BOX_H = 0.024;  BOX_W_FRAC = 0.20;  Y_OFFSET = 0.058
    y_boxes = []
    for i, (ax, (_, yd, *_)) in enumerate(zip(axs, pdata)):
        pos     = ax.get_position()
        xlbl_bb = ax.xaxis.label.get_window_extent(renderer)
        # Fixed vertical centre below the panel frame (DPI-safe)
        lbl_cy  = pos.y0 - Y_OFFSET
        # Quantity label starts 0.025 to the right of the xlabel's right edge
        lbl_x   = fig_inv.transform((xlbl_bb.x1, 0))[0] + 0.025
        by  = lbl_cy - BOX_H / 2
        bw  = pos.width * BOX_W_FRAC
        bx  = pos.x1 - bw - 0.006           # flush to panel right edge
        fig.text(lbl_x, lbl_cy, pdata[i][5], ha='left', va='center',
                 fontfamily='serif', fontsize=8.5, color=C_NAVY, fontweight='bold')
        y_boxes.append(_textbox([bx, by, bw, BOX_H], f'{yd[0]:.4g}'))

    # ── Re-entrant guard (prevents feedback loops when set_val triggers submit) ─
    _lock = [False]

    # ── Core update  (called by every input method via slider.set_val) ─────────
    def update(_):
        if _lock[0]: return
        _lock[0] = True
        try:
            for tl in hidden_ticks: tl.set_visible(True)
            hidden_ticks.clear()

            idx = int(np.argmin(np.abs(t - slider.val)))

            for i, (ax, (xd, yd, *_)) in enumerate(zip(axs, pdata)):
                xlim = ax.get_xlim(); ylim = ax.get_ylim()
                xi, yi = xd[idx], yd[idx]
                dots[i].set_data([xi], [yi])
                vls[i].set_data([xi, xi],      [ylim[0], yi])
                hls[i].set_data([xlim[0], xi], [yi,      yi])
                xlbls[i].set_position((xi,      ylim[0])); xlbls[i].set_text(f'{xi:.4g}')
                ylbls[i].set_position((xlim[0], yi));      ylbls[i].set_text(f'{yi:.4g}')

            # Keep all text inputs in sync with the current time index
            t_box.set_val(f'{t[idx]:.4g}')
            for i, (yb, (_, yd, *_)) in enumerate(zip(y_boxes, pdata)):
                yb.set_val(f'{yd[idx]:.4g}')

            fig.canvas.draw()
            for i, ax in enumerate(axs):
                hidden_ticks.extend(_hide_overlapping(ax, xlbls[i], 'x'))
                hidden_ticks.extend(_hide_overlapping(ax, ylbls[i], 'y'))
            fig.canvas.draw_idle()
        finally:
            _lock[0] = False

    # ── Callback: time TextBox ─────────────────────────────────────────────────
    def on_time_submit(text):
        try:
            t_val = float(text.strip())
            slider.set_val(np.clip(t_val, float(t[0]), float(t[-1])))
        except ValueError:
            if not _lock[0]: t_box.set_val(f'{slider.val:.4g}')  # restore valid

    # ── Callback: y-value TextBox ──────────────────────────────────────────────
    def on_y_submit(text, panel_idx):
        _, yd = pdata[panel_idx][:2]
        try:
            y_val = float(text.strip())
            idx   = int(np.argmin(np.abs(yd - y_val)))
            slider.set_val(float(t[idx]))
        except ValueError:
            if not _lock[0]:
                cur = int(np.argmin(np.abs(t - slider.val)))
                y_boxes[panel_idx].set_val(f'{yd[cur]:.4g}')   # restore valid

    # ── Callback: click on a panel ─────────────────────────────────────────────
    def on_click(event):
        if event.button != 1 or event.inaxes is None: return
        for i, ax in enumerate(axs):
            if event.inaxes is ax:
                xd, yd = pdata[i][:2]
                if not time_based[i]:
                    # 2-D trajectory: nearest point in normalised coordinates
                    xr = max(float(np.ptp(xd)), 1e-10)
                    yr = max(float(np.ptp(yd)), 1e-10)
                    dist2 = ((xd - event.xdata) / xr) ** 2 + \
                            ((yd - event.ydata) / yr) ** 2
                    idx = int(np.argmin(dist2))
                else:
                    # Time-based panel: nearest time to click x-coordinate
                    idx = int(np.argmin(np.abs(xd - event.xdata)))
                slider.set_val(float(t[idx]))
                break

    # ── Wire all callbacks ─────────────────────────────────────────────────────
    t_box.on_submit(on_time_submit)
    for i, yb in enumerate(y_boxes):
        yb.on_submit(lambda text, i=i: on_y_submit(text, i))
    fig.canvas.mpl_connect('button_press_event', on_click)
    slider.on_changed(update)

    update(None)
    plt.show()


# ================================================================================
# SECTION 7: MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":

    # ============================================================
    # 7.0  BASE CASE
    # ------------------------------------------------------------
    #  Default parameters from the problem statement:
    #    T   = 1500 N     Isp = 100 s     ω = 7.27e-5 rad/s
    #    m0  = 100 kg     mp  = 10 kg     propellant = 90 kg
    #    Cd  = 0.5        Aref = 0.1 m²   ISA atmosphere
    #
    #  Derived burn quantities (analytical):
    #    ṁ      = T / (g₀·Isp) = 1500 / (9.80665·100) ≈ 1.5296 kg/s
    #    t_burn = m_prop / ṁ   = 90 / 1.5296          ≈ 58.84 s
    # ============================================================

    print("\n" + "=" * 55)
    print("  3D ROCKET TRAJECTORY SIMULATION")
    print("=" * 55)
    print(f"  Thrust        : {T0} N")
    print(f"  Isp           : {ISP} s")
    print(f"  m0            : {M0} kg   (mp = {MP} kg payload)")
    print(f"  Propellant    : {M0 - MP:.1f} kg")
    mdot_base = T0 / (G0 * ISP)
    tburn_base = (M0 - MP) / mdot_base
    print(f"  Mass flow     : {mdot_base:.4f} kg/s")
    print(f"  Burn time     : {tburn_base:.2f} s")
    print(f"  Atmosphere    : ISA (multi-layer)")
    print(f"  Earth omega   : {OMEGA_EARTH:.2e} rad/s")
    print("=" * 55)

    print("\n>>> Running base case simulation...")
    t0, r0, v0, drag0, mass0, accel0 = simulate_rocket()

    metrics0 = compute_metrics(t0, r0, v0)
    print_metrics(metrics0, "BASE CASE  (T=1500 N, ISA, Coriolis ON)")
    plot_base_results(t0, r0, v0, drag0, title="Base Case — T=1500 N, ISA Atmosphere, Coriolis ON")


    # ============================================================
    # 7.1  TASK 1: THRUST VARIATION
    # ------------------------------------------------------------
    #  Run with T = 1000 N, 1500 N (base), 3000 N.
    #  Note: changing T also changes both ṁ and t_burn.
    #
    #  Questions to answer:
    #    - How does maximum altitude change with thrust?
    #    - Does the rocket escape the dense atmosphere (> ~12 km)?
    #    - How does burnout time vary?
    # ============================================================

    print("\n" + "─" * 55)
    print("  TASK 1: Thrust Variation  (T = 1000, 1500, 3000 N)")
    print("─" * 55)

    thrust_cases  = [1000.0, 1500.0, 3000.0]
    colors_task1  = ['royalblue', 'darkorange', 'crimson']
    results_task1 = []

    for T_val in thrust_cases:
        t_s, r_s, v_s, d_s, m_s, a_s = simulate_rocket(thrust=T_val)
        results_task1.append((t_s, r_s, v_s, d_s, m_s, a_s))
        m_t1 = compute_metrics(t_s, r_s, v_s)
        # also print the corresponding burn time for this thrust
        mdot_t1  = T_val / (G0 * ISP)
        tburn_t1 = (M0 - MP) / mdot_t1
        print_metrics(m_t1, f"Task 1 — T = {T_val:.0f} N  (t_burn = {tburn_t1:.2f} s)")

    # Comparison plot: altitude and speed for the three thrust values
    t1_colors     = [C_NAVY, C_BLUE, C_LTBLUE]
    t1_linestyles = ['-', '--', ':']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _style_fig(fig, "Task 1 — Effect of Thrust Variation")

    for (t_s, r_s, v_s, d_s, m_s, a_s), T_val, col, ls in zip(
            results_task1, thrust_cases, t1_colors, t1_linestyles):
        label   = f"T = {T_val:.0f} N"
        speed_s = np.linalg.norm(v_s, axis=1)
        axes[0].plot(t_s, r_s[:, 2] / 1000.0, label=label, color=col, linestyle=ls)
        axes[1].plot(t_s, speed_s,              label=label, color=col, linestyle=ls)

    _style_ax(axes[0], "Altitude vs Time",  "Time [s]", "Altitude [km]")
    _style_ax(axes[1], "Speed vs Time",     "Time [s]", "Speed [m/s]")
    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 7.2  TASK 2: REMOVE CORIOLIS FORCE
    # ------------------------------------------------------------
    #  Compare the trajectory with ω = 7.27e-5 rad/s (base)
    #  against ω = 0  (Coriolis disabled).
    #
    #  Questions to answer:
    #    - How does the trajectory change?
    #    - Which direction (x_East or y_North) is most affected?
    # ============================================================

    print("\n" + "─" * 55)
    print("  TASK 2: Removing Coriolis Force  (ω = 0)")
    print("─" * 55)

    t_nc, r_nc, v_nc, drag_nc, mass_nc, accel_nc = simulate_rocket(omega=0.0)
    metrics_nc = compute_metrics(t_nc, r_nc, v_nc)
    print_metrics(metrics_nc, "Task 2 — No Coriolis  (ω = 0)")

    # Print difference in landing position
    print(f"  Δx (East)  with/without Coriolis: "
          f"{metrics0['x_land']:.4f} m  vs  {metrics_nc['x_land']:.4f} m")
    print(f"  Δy (North) with/without Coriolis: "
          f"{metrics0['y_land']:.4f} m  vs  {metrics_nc['y_land']:.4f} m")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _style_fig(fig, "Task 2 — Coriolis Effect ON vs OFF")

    # Left panel: horizontal trajectory
    axes[0].plot(r0[:, 0],   r0[:, 1],   color=C_NAVY, linestyle='-',  label="With Coriolis")
    axes[0].plot(r_nc[:, 0], r_nc[:, 1], color=C_GOLD, linestyle='--', label="No Coriolis")
    _style_ax(axes[0], "Horizontal Trajectory (y vs x)",
              "x — East [m]", "y — North [m]")
    axes[0].legend()
    axes[0].set_aspect('equal', adjustable='datalim')

    # Right panel: East and North displacement vs time
    # Solid = Coriolis ON, dashed = Coriolis OFF; navy = East, blue = North
    axes[1].plot(t0,   r0[:, 0],   color=C_NAVY, linestyle='-',  label="x East  (Coriolis ON)")
    axes[1].plot(t_nc, r_nc[:, 0], color=C_NAVY, linestyle='--', label="x East  (Coriolis OFF)")
    axes[1].plot(t0,   r0[:, 1],   color=C_BLUE, linestyle='-',  label="y North (Coriolis ON)")
    axes[1].plot(t_nc, r_nc[:, 1], color=C_BLUE, linestyle='--', label="y North (Coriolis OFF)")
    _style_ax(axes[1], "East / North Displacement vs Time",
              "Time [s]", "Horizontal displacement [m]")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 7.3  TASK 3: SIMPLIFIED ATMOSPHERE
    # ------------------------------------------------------------
    #  Replace ISA with:  ρ(z) = ρ₀ · exp(−z / H),  H = 8500 m
    #  Keep all other parameters identical to the base case.
    #
    #  Questions to answer:
    #    - Compare maximum altitude (ISA vs exponential).
    #    - Compare drag evolution over time.
    #    - Which model is more physically realistic?
    # ============================================================

    print("\n" + "─" * 55)
    print("  TASK 3: Simplified Atmosphere  ρ(z) = ρ₀ exp(−z/H)")
    print("─" * 55)

    t_exp, r_exp, v_exp, drag_exp, mass_exp, accel_exp = simulate_rocket(
        density_func=density_exponential
    )
    metrics_exp = compute_metrics(t_exp, r_exp, v_exp)
    print_metrics(metrics_exp, "Task 3 — Exponential atmosphere  ρ(z) = ρ₀ exp(−z/H)")

    # Print comparison summary
    print(f"\n  Altitude comparison:")
    print(f"    ISA max altitude   : {metrics0['z_max']/1000:.3f} km")
    print(f"    Exp max altitude   : {metrics_exp['z_max']/1000:.3f} km")
    print(f"    Difference         : {(metrics_exp['z_max']-metrics0['z_max'])/1000:.3f} km")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _style_fig(fig, "Task 3 — ISA vs Exponential Atmosphere")

    # Left: altitude vs time
    axes[0].plot(t0,    r0[:, 2]    / 1000.0, color=C_NAVY, linestyle='-',  label="ISA")
    axes[0].plot(t_exp, r_exp[:, 2] / 1000.0, color=C_GOLD, linestyle='--', label="Exponential")
    _style_ax(axes[0], "Altitude vs Time", "Time [s]", "Altitude [km]")
    axes[0].legend()

    # Right: drag vs time
    axes[1].plot(t0,    drag0,    color=C_NAVY, linestyle='-',  label="ISA")
    axes[1].plot(t_exp, drag_exp, color=C_GOLD, linestyle='--', label="Exponential")
    _style_ax(axes[1], "Aerodynamic Drag vs Time", "Time [s]", "Drag force [N]")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 7.4  TASK 4: PROPELLANT MASS VARIATION
    # ------------------------------------------------------------
    #  Change the propellant mass:  10 kg → 50 kg → 90 kg (base).
    #  The payload mass MP = 10 kg remains constant.
    #  Total initial mass = MP + m_propellant varies with each case.
    #
    #  Questions to answer:
    #    - How does the acceleration profile change?
    #    - How does the burnout altitude change?
    # ============================================================

    print("\n" + "─" * 55)
    print("  TASK 4: Propellant Mass Variation  (10, 50, 90 kg)")
    print("─" * 55)

    prop_cases  = [10.0, 50.0, 90.0]
    colors_t4   = ['royalblue', 'darkorange', 'crimson']
    results_t4  = []

    mdot_t4 = T0 / (G0 * ISP)   # mass flow rate is the same for all (same T, Isp)

    for mp_val in prop_cases:
        t_s, r_s, v_s, d_s, m_s, a_s = simulate_rocket(m_propellant=mp_val)
        results_t4.append((t_s, r_s, v_s, d_s, m_s, a_s))
        m_t4   = compute_metrics(t_s, r_s, v_s)
        tburn4 = mp_val / mdot_t4
        print_metrics(m_t4, f"Task 4 — propellant = {mp_val:.0f} kg  (t_burn = {tburn4:.2f} s)")

    t4_colors     = [C_NAVY, C_BLUE, C_LTBLUE]
    t4_linestyles = ['-', '--', ':']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _style_fig(fig, "Task 4 — Propellant Mass Variation")

    for (t_s, r_s, v_s, d_s, m_s, a_s), mp_val, col, ls in zip(
            results_t4, prop_cases, t4_colors, t4_linestyles):
        tburn4 = mp_val / mdot_t4
        label  = f"prop = {mp_val:.0f} kg"

        # Left: altitude vs time — vertical dotted line marks burnout
        axes[0].plot(t_s, r_s[:, 2] / 1000.0, label=label, color=col, linestyle=ls)
        axes[0].axvline(tburn4, color=col, linestyle=':', linewidth=1.0, alpha=0.6)

        # Right: acceleration magnitude vs time
        axes[1].plot(t_s, a_s, label=label, color=col, linestyle=ls)
        axes[1].axvline(tburn4, color=col, linestyle=':', linewidth=1.0, alpha=0.6)

    _style_ax(axes[0], "Altitude vs Time\n(dotted line = burnout)",
              "Time [s]", "Altitude [km]")
    _style_ax(axes[1], "Acceleration vs Time\n(dotted line = burnout)",
              "Time [s]", "Acceleration magnitude [m/s\xb2]")
    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print("\n>>> All simulations complete.")
