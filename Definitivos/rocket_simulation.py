"""
3D rocket trajectory simulation — explicit Euler integration.
Authors: Rubén Praena, Claudia Ros, Antoni Ruiz, Miguel Sanz,
         Elishka Mrázek, Joel García, Giulia Latorre

Frame: x = East, y = North, z = Up (equatorial surface, local inertial).
Forces: gravity g(z)=GM/(R+z)², drag F_D=-½CdAρ|v|v, thrust T·ẑ, Coriolis -2ω×v.
Euler: r_{n+1} = r_n + v_n·dt,  v_{n+1} = v_n + a_n·dt,  m_{n+1} = m_n - ṁ·dt
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



# SECTION 1: PHYSICAL AND EARTH CONSTANTS

GM          = 3.99e14     # m³/s²   — Earth's gravitational parameter
R_EARTH     = 6.378e6    # m       — Earth's mean radius
G0          = 9.80665    # m/s²    — standard gravity at surface (used for mass flow rate)
OMEGA_EARTH = 7.27e-5    # rad/s   — Earth's angular velocity



# SECTION 2: DEFAULT ROCKET PARAMETERS

M0   = 100.0   # kg   — initial total mass  (structure + propellant + payload)
MP   = 10.0    # kg   — payload/structural mass that remains after burnout
T0   = 1500.0  # N    — nominal thrust
ISP  = 100.0   # s    — specific impulse  (low value → low fuel efficiency)
CD   = 0.5     # —    — drag coefficient  (dimensionless)
AREF = 0.1     # m²   — rocket cross-sectional reference area


# SECTION 3: ATMOSPHERIC DENSITY MODELS

# density_isa: multi-layer ISA model, used by default.
# density_exponential: ρ₀·exp(-z/H), used in 4_3_Drag.py for comparison.
# Note: ISA has a slope discontinuity at each layer boundary (kink in the lapse rate).


# ISA atmospheric constants
RHO0    = 1.225    # kg/m³     — sea-level density
H_SCALE = 8500.0   # m         — exponential scale height
T0_ISA  = 288.15   # K         — sea-level temperature
L_TROP  = 0.0065   # K/m       — troposphere lapse rate (temperature drop per metre)
R_AIR   = 287.05   # J/(kg·K)  — specific gas constant for dry air


def density_isa(z):
    """
    ISA air density [kg/m³] at altitude z [m].
    Layers: troposphere 0-11 km, lower stratosphere 11-20 km (isothermal at 216.65 K),
    upper stratosphere 20-32 km, mesosphere 32-51 km, thermosphere 51-85 km,
    exponential fallback above 85 km. Returns at least 1e-20.
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
    """Single-exponential model: ρ(z) = ρ₀·exp(−z/H), H = 8500 m."""
    return RHO0 * np.exp(-z / H_SCALE)


# SECTION 4: EULER INTEGRATION

def simulate_rocket(
    thrust        = T0,
    omega         = OMEGA_EARTH,
    density_func  = density_isa,
    m_propellant  = M0 - MP,    # kg — propellant mass (default: 90 kg)
    dt            = 0.1,         # s  — Euler time step
):


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

    while True:
        z = r[2]   # current altitude [m]

        # --- termination: rocket has hit the ground after liftoff ---
        if launched and z < 0.0:
            break

        # Clamp altitude for physics evaluation (density/gravity below ground = sea level)
        z_eff = max(z, 0.0)

        # gravity — decreases with altitude
        g_mag  = GM / (R_EARTH + z_eff) ** 2
        a_grav = np.array([0.0, 0.0, -g_mag])

        # thrust — vertical, active while propellant remains
        if t < t_burn:
            a_thrust = np.array([0.0, 0.0, thrust / m])
            m_next   = max(m - mdot * dt, MP)   # clamp so mass never drops below MP
        else:
            a_thrust = np.zeros(3)
            m_next   = MP

        # drag — F_D = -½ Cd A ρ |v| v
        rho   = density_func(z_eff)
        v_mag = np.linalg.norm(v)

        if v_mag > 0.0:
            F_drag = -0.5 * CD * AREF * rho * v_mag * v
        else:
            F_drag = np.zeros(3)

        a_drag = F_drag / m

        # Coriolis: -2ω×v; upward motion deflects west at the equator
        a_coriolis = -2.0 * np.cross(omega_vec, v)

        a = a_grav + a_thrust + a_drag + a_coriolis

        # Euler: r uses velocity from the current step, not the updated one
        r_new = r + v * dt
        v_new = v + a * dt

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


# SECTION 5: POST-PROCESSING UTILITIES

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


# SECTION 6: PLOTTING STYLE AND FUNCTIONS

# colour palette
C_NAVY   = '#0D2B55'   # dark navy — main colour
C_BLUE   = '#2A6496'   # mid blue
C_LTBLUE = '#5B9BD5'   # light blue
C_GOLD   = '#B07D2A'   # amber gold — accent for two-series plots
C_GRID   = '#D0D8E4'   # grid lines

# global rcParams
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
    """Set figure suptitle."""
    fig.suptitle(title, fontsize=14, fontweight='bold', color=C_NAVY)


def _style_ax(ax, title, xlabel, ylabel):
    """Apply title, axis labels and spine colour."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_NAVY)
        spine.set_linewidth(1.2)
    ax.tick_params(colors=C_NAVY, which='both')


def plot_base_results(t, r, v, drag, title="Base Case"):
    """
    4-panel interactive figure: altitude, horizontal projection, speed, drag.
    Controls: time slider, click on any panel, or type a value in the text boxes.
    All inputs stay in sync. Tick labels that overlap the floating value label
    are temporarily hidden.
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

    # bottom=0.22 clears the slider; text boxes sit inline with the x-axis labels
    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(2, 2,
                           left=0.09, right=0.96,
                           bottom=0.22, top=0.93,
                           hspace=0.50, wspace=0.40)
    axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    _style_fig(fig, title)

    # static curves
    for ax, (xd, yd, xl, yl, ttl, _) in zip(axs, pdata):
        ax.plot(xd, yd, color=C_NAVY, zorder=2)
        _style_ax(ax, ttl, xl, yl)
    axs[1].set_aspect('equal', adjustable='datalim')
    fig.canvas.draw()    # finalise axis limits before placing dynamic elements

    # separator above slider
    fig.add_artist(Line2D([0.04, 0.96], [0.128, 0.128],
                           transform=fig.transFigure,
                           color=C_GRID, linewidth=0.9, zorder=0))

    # dynamic overlay: dot + crosshair + floating value labels
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

    # styled text-box factory
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

    # slider
    ax_sl = fig.add_axes([0.09, 0.090, 0.57, 0.022])
    ax_sl.set_facecolor('white')
    for sp in ax_sl.spines.values(): sp.set_edgecolor(C_GRID)
    slider = Slider(ax_sl, 'Time  [s]', float(t[0]), float(t[-1]),
                    valinit=float(t[0]), color=C_NAVY)
    slider.label.set_fontfamily('serif'); slider.label.set_color(C_NAVY)
    slider.valtext.set_visible(False)   # value shown in the TextBox instead

    # time text box (right of slider)
    fig.text(0.678, 0.101, 't [s]:', ha='right', va='center',
             fontfamily='serif', fontsize=9, color=C_NAVY)
    t_box = _textbox([0.682, 0.084, 0.090, 0.033], f'{t[0]:.4g}')

    # y-value text boxes, one per panel (inline with x-axis label)
    renderer = fig.canvas.get_renderer()
    fig_inv  = fig.transFigure.inverted()
    BOX_H = 0.024;  BOX_W_FRAC = 0.20;  Y_OFFSET = 0.058
    y_boxes = []
    for i, (ax, (_, yd, *_)) in enumerate(zip(axs, pdata)):
        pos     = ax.get_position()
        xlbl_bb = ax.xaxis.label.get_window_extent(renderer)
        lbl_cy  = pos.y0 - Y_OFFSET
        lbl_x   = fig_inv.transform((xlbl_bb.x1, 0))[0] + 0.025
        by  = lbl_cy - BOX_H / 2
        bw  = pos.width * BOX_W_FRAC
        bx  = pos.x1 - bw - 0.006           # flush to panel right edge
        fig.text(lbl_x, lbl_cy, pdata[i][5], ha='left', va='center',
                 fontfamily='serif', fontsize=8.5, color=C_NAVY, fontweight='bold')
        y_boxes.append(_textbox([bx, by, bw, BOX_H], f'{yd[0]:.4g}'))

    # re-entrant guard: set_val can trigger submit, which would loop
    _lock = [False]

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

    def on_time_submit(text):
        try:
            t_val = float(text.strip())
            slider.set_val(np.clip(t_val, float(t[0]), float(t[-1])))
        except ValueError:
            if not _lock[0]: t_box.set_val(f'{slider.val:.4g}')  # restore on bad input

    def on_y_submit(text, panel_idx):
        _, yd = pdata[panel_idx][:2]
        try:
            y_val = float(text.strip())
            idx   = int(np.argmin(np.abs(yd - y_val)))
            slider.set_val(float(t[idx]))
        except ValueError:
            if not _lock[0]:
                cur = int(np.argmin(np.abs(t - slider.val)))
                y_boxes[panel_idx].set_val(f'{yd[cur]:.4g}')  # restore on bad input

    def on_click(event):
        if event.button != 1 or event.inaxes is None: return
        for i, ax in enumerate(axs):
            if event.inaxes is ax:
                xd, yd = pdata[i][:2]
                if not time_based[i]:
                    # spatial panel: nearest point by normalised 2D distance
                    xr = max(float(np.ptp(xd)), 1e-10)
                    yr = max(float(np.ptp(yd)), 1e-10)
                    dist2 = ((xd - event.xdata) / xr) ** 2 + \
                            ((yd - event.ydata) / yr) ** 2
                    idx = int(np.argmin(dist2))
                else:
                    idx = int(np.argmin(np.abs(xd - event.xdata)))
                slider.set_val(float(t[idx]))
                break

    t_box.on_submit(on_time_submit)
    for i, yb in enumerate(y_boxes):
        yb.on_submit(lambda text, i=i: on_y_submit(text, i))
    fig.canvas.mpl_connect('button_press_event', on_click)
    slider.on_changed(update)

    update(None)
    plt.show()


# SECTION 7: MAIN EXECUTION

if __name__ == "__main__":

    # 7.0 base case — default parameters, ISA atmosphere, Coriolis ON

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
