"""
3-stage rocket simulation to ISS orbit, ECI frame.
Authors: Rubén Praena, Claudia Ros, Antoni Ruiz, Miguel Sanz,
         Elishka Mrázek, Joel García, Giulia Latorre

ECI: origin at Earth's centre, x toward vernal equinox, z toward North Pole.
Launch from equator: r0=(R_EARTH,0,0), v0=(0,Ω·R_EARTH,0).
Integrator: explicit Euler (dt=0.1 s in atmosphere, 5 s in vacuum),
            leapfrog for coast/orbital phases to avoid energy drift.

Known simplifications vs. a real mission:
  1. Equatorial orbit — ISS is at 51.6°; a real launch needs a plane change.
  2. Pre-programmed gravity turn, no active guidance.
  3. Attitude changes modelled as thrust direction changes, no rigid-body dynamics.
  4. Rendezvous = matching orbital speed at ISS altitude; proximity ops not simulated.
  5. Stage 2 ignites immediately after Stage 1 separation.
  6. Single Cd·A value for all phases.
"""

import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# SECTION 1 — PHYSICAL CONSTANTS

GM          = 3.99e14      # m³/s²   gravitational parameter
R_EARTH     = 6.378e6      # m       mean radius
G0          = 9.80665      # m/s²    standard gravity
OMEGA_EARTH = 7.27e-5      # rad/s   Earth's angular velocity

# ISA atmospheric constants (same as rocket_simulation.py)
RHO0    = 1.225
T0_ISA  = 288.15
L_TROP  = 0.0065
R_AIR   = 287.05

H_DRAG_CUTOFF = 100e3      # m   — no drag above the Kármán line
H_ISS         = 408e3      # m   — ISS target altitude

# Gravity-turn pitch profile (altitude limits)
H_PITCH_START = 3e3        # m   — begin pitching at 3 km
H_PITCH_END   = 80e3       # m   — fully horizontal at 80 km


# SECTION 2 — ROCKET PARAMETERS
# Soyuz-type architecture scaled for 500 kg payload.
# Structural fraction ≈ 10% of stage propellant mass.
# Stage 1 — booster, kerosene/LOX type
S1_THRUST  = 800e3    # N      (800 kN)
S1_ISP     = 290      # s
S1_M_PROP  = 40000    # kg
S1_M_STRUCT = 4000    # kg     (tank + engine, jettisoned at separation)

# Stage 2 — upper stage, LH2/LOX type
S2_THRUST  = 200e3    # N      (200 kN)
S2_ISP     = 348      # s
S2_M_PROP  = 8000     # kg
S2_M_STRUCT = 1000    # kg

# Stage 3 — maneuvering / RCS (remains attached to spacecraft)
S3_THRUST  = 2e3      # N      (2 kN)
S3_ISP     = 300      # s
S3_M_PROP  = 500      # kg
S3_M_STRUCT = 200     # kg     (spacecraft bus, stays with payload)

M_PAYLOAD   = 500     # kg

# Aerodynamics (constant across all phases — simplification 6)
CD   = 0.30
AREF = 4.0            # m²   (wider body than the toy rocket in main project)

M_INITIAL = (S1_M_PROP + S1_M_STRUCT
             + S2_M_PROP + S2_M_STRUCT
             + S3_M_PROP + S3_M_STRUCT
             + M_PAYLOAD)   # 54 200 kg total


# SECTION 3 — ATMOSPHERIC DENSITY (ISA, adapted)

def density_isa(z):
    """ISA air density [kg/m³] at altitude z [m]. Returns 0 above 100 km."""
    if z >= H_DRAG_CUTOFF:
        return 0.0
    z_km = z / 1000.0
    if z_km < 11:
        T = T0_ISA - L_TROP * z
        p = 101325.0 * (T / T0_ISA) ** (G0 / (L_TROP * R_AIR))
    elif z_km < 20:
        T = 216.65
        p = 22632.1 * np.exp(-G0 * (z - 11000) / (R_AIR * T))
    elif z_km < 32:
        T     = 216.65 + 0.001 * (z - 20000)
        T_avg = (216.65 + T) / 2.0
        p     = 5474.89 * np.exp(-G0 * (z - 20000) / (R_AIR * T_avg))
    elif z_km < 47:
        T     = 228.65 + 0.0028 * (z - 32000)
        T_avg = (228.65 + T) / 2.0
        p     = 868.02 * np.exp(-G0 * (z - 32000) / (R_AIR * T_avg))
    elif z_km < 51:
        T = 270.65
        p = 110.91 * np.exp(-G0 * (z - 47000) / (R_AIR * T))
    elif z_km < 71:
        T     = 270.65 + 0.0028 * (z - 51000)
        T_avg = (270.65 + T) / 2.0
        p     = 66.939 * np.exp(-G0 * (z - 51000) / (R_AIR * T_avg))
    elif z_km < 85:
        T     = 270.65 + 0.002 * (z - 71000)
        T_avg = (270.65 + T) / 2.0
        p     = 3.9564 * np.exp(-G0 * (z - 71000) / (R_AIR * T_avg))
    else:
        return max(1.5e-5 * np.exp(-(z - 85000) / 6000.0), 0.0)
    return max(p / (R_AIR * T), 0.0)


# SECTION 4 — GEOMETRY HELPERS

def prograde_hat(r):
    """
    Unit prograde vector in the orbital plane (perpendicular to r, eastward).
    For an equatorial launch from +x axis: p_hat = ẑ × r_hat.
    Works as long as the trajectory stays near the equatorial plane.
    """
    z_pole = np.array([0.0, 0.0, 1.0])
    p = np.cross(z_pole, r)
    norm = np.linalg.norm(p)
    if norm < 1e-10:
        return np.array([0.0, 1.0, 0.0])
    return p / norm


def pitch_angle(h):
    """
    Pre-programmed gravity turn: pitch angle α from radial (vertical).
      h < H_PITCH_START  → α = 0     (pure vertical)
      H_PITCH_START ≤ h ≤ H_PITCH_END → α linearly from 0 to π/2
      h > H_PITCH_END    → α = π/2   (pure horizontal / prograde)
    """
    if h <= H_PITCH_START:
        return 0.0
    if h >= H_PITCH_END:
        return np.pi / 2.0
    frac = (h - H_PITCH_START) / (H_PITCH_END - H_PITCH_START)
    return frac * np.pi / 2.0


def thrust_hat(r, h, mode='gravity_turn'):
    """
    Thrust direction unit vector in ECI frame.

    mode = 'gravity_turn'  : pitch angle from vertical based on altitude
    mode = 'prograde'      : purely prograde (for circularisation)
    mode = 'radial'        : purely radial/vertical (Stage 1 initial)
    """
    r_hat = r / np.linalg.norm(r)
    t_hat = prograde_hat(r)

    if mode == 'radial':
        return r_hat
    elif mode == 'prograde':
        return t_hat
    else:  # gravity_turn
        alpha = pitch_angle(h)
        return np.cos(alpha) * r_hat + np.sin(alpha) * t_hat


def orbital_elements(r, v):
    """
    Returns (altitude_km, speed_ms, ecc, sma_km) for the current state.
    sma = semi-major axis.
    """
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h_km   = (r_norm - R_EARTH) / 1e3

    eps = v_norm**2 / 2 - GM / r_norm
    sma = -GM / (2 * eps) if abs(eps) > 1 else float('inf')

    e_vec = (v_norm**2 / GM - 1 / r_norm) * r - np.dot(r, v) / GM * v
    ecc   = np.linalg.norm(e_vec)

    return h_km, v_norm, ecc, sma / 1e3


def _orbit_apogee_perigee(r, v):
    """
    Returns (apogee_altitude_m, perigee_altitude_m) of the Keplerian orbit.
    Returns (inf, -inf) for escape trajectories.
    """
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    eps    = v_norm**2 / 2 - GM / r_norm
    if eps >= 0:
        return float('inf'), float('-inf')
    sma   = -GM / (2 * eps)
    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)
    ecc   = np.sqrt(max(1 + 2 * eps * h_mag**2 / GM**2, 0.0))
    return sma * (1 + ecc) - R_EARTH, sma * (1 - ecc) - R_EARTH


def apogee_altitude(r, v):
    """Predicted apogee altitude [m]. Returns inf for escape trajectory."""
    return _orbit_apogee_perigee(r, v)[0]


# SECTION 5 — SIMULATION

# Phase labels (used in output arrays)
PHASE_S1   = 1   # Stage 1 burning
PHASE_S2   = 2   # Stage 2 burning
PHASE_COAS = 3   # Coast (suborbital) to first apogee
PHASE_S3C  = 4   # Stage 3 circularisation at parking orbit
PHASE_HOH  = 5   # Hohmann coast from parking orbit to ISS altitude
PHASE_ORB  = 6   # ISS circular orbit + rendezvous

PHASE_NAMES = {
    PHASE_S1:   'Stage 1 burn',
    PHASE_S2:   'Stage 2 burn',
    PHASE_COAS: 'Coast',
    PHASE_S3C:  'Circularisation',
    PHASE_HOH:  'Hohmann transfer',
    PHASE_ORB:  'ISS orbit / Rendezvous',
}


def simulate_multistage(dt_atm=0.1, dt_space=5.0):
    """
    Simulate the full 3-stage mission to ISS using explicit Euler integration.

    Two time steps are used:
      dt_atm   — inside atmosphere (h < 100 km), default 0.1 s
      dt_space — in vacuum / orbit, default 5.0 s

    Returns
    -------
    dict with keys:
      't'      : (N,)   time [s]
      'r'      : (N,3)  ECI position [m]
      'v'      : (N,3)  ECI velocity [m/s]
      'm'      : (N,)   total mass [kg]
      'phase'  : (N,)   int phase label (see PHASE_* constants)
      'events' : list of (t, label) tuples at key mission events
      'h'      : (N,)   altitude [km]
      'speed'  : (N,)   speed [m/s]
    """
    # initial state
    r = np.array([R_EARTH, 0.0, 0.0], dtype=float)
    v = np.array([0.0, OMEGA_EARTH * R_EARTH, 0.0], dtype=float)
    m = float(M_INITIAL)

    prop1 = float(S1_M_PROP)
    prop2 = float(S2_M_PROP)
    prop3 = float(S3_M_PROP)

    phase = PHASE_S1
    t     = 0.0

    t_list     = []
    r_list     = []
    v_list     = []
    m_list     = []
    phase_list = []
    events     = []

    t_circ_done = None
    T_MAX = 25000.0  # s — well beyond 2 orbital periods

    # main integration loop
    while t < T_MAX:

        r_norm = np.linalg.norm(r)
        h      = r_norm - R_EARTH
        r_hat  = r / r_norm

        t_list.append(t)
        r_list.append(r.copy())
        v_list.append(v.copy())
        m_list.append(m)
        phase_list.append(phase)

        # time step
        if h < H_DRAG_CUTOFF:
            dt = dt_atm
        elif phase in (PHASE_HOH, PHASE_ORB):
            dt = 1.0
        else:
            dt = dt_space

        # atmospheric drag (v_rel = velocity relative to co-rotating atmosphere)
        if h < H_DRAG_CUTOFF:
            rho   = density_isa(h)
            v_atm = np.array([-OMEGA_EARTH * r[1], OMEGA_EARTH * r[0], 0.0])
            v_rel = v - v_atm
            vr    = np.linalg.norm(v_rel)
            a_drag = -0.5 * CD * AREF * rho * vr / m * v_rel
        else:
            a_drag = np.zeros(3)

        # thrust
        a_thrust = np.zeros(3)

        if phase == PHASE_S1 and prop1 > 0:
            T_cur   = S1_THRUST
            mdot    = T_cur / (S1_ISP * G0)
            dm      = mdot * dt
            if dm > prop1:
                dm = prop1
            t_hat_v  = thrust_hat(r, h, mode='gravity_turn')
            a_thrust = (T_cur / m) * t_hat_v
            prop1   -= dm
            m       -= dm

        elif phase == PHASE_S2 and prop2 > 0:
            T_cur   = S2_THRUST
            mdot    = T_cur / (S2_ISP * G0)
            dm      = min(mdot * dt, prop2)
            t_hat_v  = thrust_hat(r, h, mode='gravity_turn')
            a_thrust = (T_cur / m) * t_hat_v
            prop2   -= dm
            m       -= dm

        elif phase == PHASE_S3C and prop3 > 0:
            v_radial_now = np.dot(v, r_hat)
            v_tang_now   = np.sqrt(max(np.linalg.norm(v)**2 - v_radial_now**2, 0.0))
            v_circ_now   = np.sqrt(GM / r_norm)
            if v_tang_now >= v_circ_now:
                pass
            else:
                T_cur   = S3_THRUST
                mdot    = T_cur / (S3_ISP * G0)
                dm      = min(mdot * dt, prop3)
                t_hat_v  = thrust_hat(r, h, mode='prograde')
                a_thrust = (T_cur / m) * t_hat_v
                prop3   -= dm
                m       -= dm

        # coast/orbit phases: leapfrog (symplectic, avoids Euler energy drift)
        if phase in (PHASE_COAS, PHASE_HOH, PHASE_ORB) and np.all(a_drag == 0) and np.all(a_thrust == 0):
            # kick-drift-kick leapfrog
            a_r    = -(GM / r_norm**2) * r_hat
            v_half = v + a_r * (dt / 2)
            r      = r + v_half * dt
            r2     = np.linalg.norm(r)
            a_r2   = -(GM / r2**2) * (r / r2)
            v      = v_half + a_r2 * (dt / 2)
        else:
            a_grav  = -(GM / r_norm**2) * r_hat
            a_total = a_grav + a_drag + a_thrust
            r = r + v * dt
            v = v + a_total * dt
        t = t + dt

        # phase transitions

        if phase == PHASE_S1:
            if prop1 <= 0:
                phase = PHASE_S2
                m    -= S1_M_STRUCT
                events.append((t, 'Stage 1 separation'))

        elif phase == PHASE_S2:
            if prop2 <= 0:
                phase = PHASE_COAS
                m    -= S2_M_STRUCT
                h_sep = (np.linalg.norm(r) - R_EARTH) / 1e3
                events.append((t, f'Stage 2 burnout + separation at {h_sep:.1f} km — coasting to apogee'))

        elif phase == PHASE_COAS:
            v_radial = np.dot(v, r / np.linalg.norm(r))
            if v_radial < 0 and t > 50:
                phase = PHASE_S3C
                events.append((t, f'Apogee at {(np.linalg.norm(r)-R_EARTH)/1e3:.1f} km — Stage 3 ignition'))

        elif phase == PHASE_S3C:
            v_radial_now = np.dot(v, r_hat)
            v_tang_now   = np.sqrt(max(np.linalg.norm(v)**2 - v_radial_now**2, 0.0))
            v_circ_now   = np.sqrt(GM / r_norm)
            h_circ       = (r_norm - R_EARTH) / 1e3
            _, _, ecc_now, _ = orbital_elements(r, v)
            if v_tang_now >= v_circ_now:
                r_park  = r_norm
                r_iss   = R_EARTH + H_ISS
                sma_hoh = (r_park + r_iss) / 2
                v_hoh1  = np.sqrt(GM * (2 / r_park - 1 / sma_hoh))
                dv_hoh1 = v_hoh1 - v_circ_now
                t_hat_now = prograde_hat(r)
                v    += dv_hoh1 * t_hat_now
                dm_hoh1 = m * (1 - np.exp(-abs(dv_hoh1) / (S3_ISP * G0)))
                m    -= dm_hoh1
                prop3 = max(prop3 - dm_hoh1, 0.0)
                phase = PHASE_HOH
                events.append((t, f'Parking orbit {h_circ:.0f} km — Hohmann burn 1 ({dv_hoh1:.1f} m/s)'))
            elif prop3 <= 0:
                phase       = PHASE_ORB
                t_circ_done = t
                events.append((t, f'Stage 3 propellant depleted — ecc={ecc_now:.4f}, h={h_circ:.1f} km'))

        elif phase == PHASE_HOH:
            v_radial_hoh = np.dot(v, r_hat)
            h_now = (r_norm - R_EARTH) / 1e3
            if v_radial_hoh < 0 and h_now > H_ISS / 1e3 * 0.95:
                v_circ_iss  = np.sqrt(GM / r_norm)
                v_tang_hoh  = np.sqrt(max(np.linalg.norm(v)**2 - v_radial_hoh**2, 0.0))
                dv_hoh2     = v_circ_iss - v_tang_hoh
                if dv_hoh2 > 0 and prop3 > 0:
                    t_hat_now = prograde_hat(r)
                    v    += dv_hoh2 * t_hat_now
                    dm_hoh2 = m * (1 - np.exp(-abs(dv_hoh2) / (S3_ISP * G0)))
                    m    -= dm_hoh2
                    prop3 = max(prop3 - dm_hoh2, 0.0)
                _, _, ecc_iss, _ = orbital_elements(r, v)
                h_iss_now = (np.linalg.norm(r) - R_EARTH) / 1e3
                events.append((t, f'ISS orbit {h_iss_now:.0f} km — Hohmann burn 2 ({dv_hoh2:.1f} m/s), ecc={ecc_iss:.4f}'))
                events.append((t, 'Attitude manoeuvres (pitch/yaw/roll) — ISS docking approach'))
                phase       = PHASE_ORB
                t_circ_done = t

        elif phase == PHASE_ORB:
            T_actual = 2 * np.pi * np.sqrt(r_norm**3 / GM)
            if t_circ_done is not None and t - t_circ_done > T_actual:
                events.append((t, 'ISS docking complete — mission success'))
                t_list.append(t)
                r_list.append(r.copy())
                v_list.append(v.copy())
                m_list.append(m)
                phase_list.append(phase)
                break

        if np.linalg.norm(r) < R_EARTH - 1000:
            events.append((t, 'CRASH — rocket fell back to Earth'))
            break

    # pack results
    t_arr     = np.array(t_list)
    r_arr     = np.array(r_list)
    v_arr     = np.array(v_list)
    m_arr     = np.array(m_list)
    phase_arr = np.array(phase_list, dtype=int)

    h_arr     = (np.linalg.norm(r_arr, axis=1) - R_EARTH) / 1e3
    speed_arr = np.linalg.norm(v_arr, axis=1)

    return {
        't':      t_arr,
        'r':      r_arr,
        'v':      v_arr,
        'm':      m_arr,
        'phase':  phase_arr,
        'events': events,
        'h':      h_arr,
        'speed':  speed_arr,
    }


# SECTION 6 — METRICS

def print_mission_summary(data):
    """Print a formatted summary of key mission milestones."""
    t     = data['t']
    h     = data['h']
    speed = data['speed']
    m     = data['m']
    phase = data['phase']

    print('=' * 62)
    print('  3-STAGE ROCKET MISSION SUMMARY')
    print('=' * 62)
    print(f'  Initial mass      : {M_INITIAL/1e3:.2f} t')
    print(f'  Payload           : {M_PAYLOAD:.0f} kg')
    print()

    for ev_t, ev_label in data['events']:
        print(f'  t = {ev_t:7.1f} s  |  {ev_label}')
    print()

    print(f'  Max altitude      : {h.max():.2f} km')
    print(f'  Max speed         : {speed.max():.1f} m/s')
    print()

    mask_orb = phase == PHASE_ORB
    if mask_orb.any():
        r_orb   = data['r'][mask_orb]
        v_orb   = data['v'][mask_orb]
        idx_mid = len(r_orb) // 2
        h_o, v_o, ecc, sma = orbital_elements(r_orb[idx_mid], v_orb[idx_mid])
        print(f'  Final orbit altitude : {h_o:.1f} km')
        print(f'  Orbital speed        : {v_o:.1f} m/s')
        print(f'  Eccentricity         : {ecc:.4f}')
        print(f'  Semi-major axis      : {sma:.1f} km')
        T_orb = 2 * np.pi * np.sqrt((sma * 1e3)**3 / GM)
        print(f'  Orbital period       : {T_orb/60:.1f} min')

    print()
    print('  Δv budget (Tsiolkovsky, ideal):')
    m0_s1 = M_INITIAL
    mf_s1 = M_INITIAL - S1_M_PROP
    dv1   = S1_ISP * G0 * np.log(m0_s1 / mf_s1)
    m0_s2 = mf_s1 - S1_M_STRUCT
    mf_s2 = m0_s2 - S2_M_PROP
    dv2   = S2_ISP * G0 * np.log(m0_s2 / mf_s2)
    m0_s3 = mf_s2 - S2_M_STRUCT
    mf_s3 = m0_s3 - S3_M_PROP
    dv3   = S3_ISP * G0 * np.log(m0_s3 / mf_s3)
    print(f'    Stage 1  : {dv1:.0f} m/s')
    print(f'    Stage 2  : {dv2:.0f} m/s')
    print(f'    Stage 3  : {dv3:.0f} m/s')
    print(f'    Total    : {dv1+dv2+dv3:.0f} m/s')
    print(f'    Needed   : ~9 400 m/s  (LEO losses included)')
    print('=' * 62)
