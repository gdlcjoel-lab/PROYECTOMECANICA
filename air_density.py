import numpy as np

# Atmospheric constants (International Standard Atmosphere)
rho0 = 1.225       # kg/m^3, sea level density
H = 8500.0         # m, scale height for exponential atmosphere
T0 = 288.15        # K, sea level temperature
L = 0.0065         # K/m, temperature lapse rate (up to 11 km)
R_air = 287.05     # J/(kg*K), specific gas constant for air
g0 = 9.80665       # m/s^2, standard gravity

def density_at_altitude(z):
    """
    Calculate air density using International Standard Atmosphere model.
    Returns density and atmospheric layer for altitude z in meters.
    """
    z_km = z / 1000.0

    if z_km < 11:
        layer = "Troposphere"
        T = T0 - L * z
        p = 101325 * (T / T0) ** (g0 / (L * R_air))
        rho = p / (R_air * T)

    elif z_km < 20:
        layer = "Lower Stratosphere"
        T = 216.65
        p_11 = 22632.1
        h = z - 11000
        p = p_11 * np.exp(-g0 * h / (R_air * T))
        rho = p / (R_air * T)

    elif z_km < 32:
        layer = "Upper Stratosphere"
        T = 216.65 + 0.001 * (z - 20000)
        p_20 = 5474.89
        h = z - 20000
        T_avg = (216.65 + T) / 2
        p = p_20 * np.exp(-g0 * h / (R_air * T_avg))
        rho = p / (R_air * T)

    elif z_km < 47:
        layer = "Mesosphere"
        T = 228.65 + 0.0028 * (z - 32000)
        p_32 = 868.02
        h = z - 32000
        T_avg = (228.65 + T) / 2
        p = p_32 * np.exp(-g0 * h / (R_air * T_avg))
        rho = p / (R_air * T)

    elif z_km < 51:
        layer = "Upper Mesosphere"
        T = 270.65
        p_47 = 110.91
        h = z - 47000
        p = p_47 * np.exp(-g0 * h / (R_air * T))
        rho = p / (R_air * T)

    elif z_km < 71:
        layer = "Lower Thermosphere"
        T = 270.65 + 0.0028 * (z - 51000)
        p_51 = 66.939
        h = z - 51000
        T_avg = (270.65 + T) / 2
        p = p_51 * np.exp(-g0 * h / (R_air * T_avg))
        rho = p / (R_air * T)

    elif z_km < 85:
        layer = "Upper Thermosphere"
        T = 270.65 + 0.002 * (z - 71000)
        p_71 = 3.9564
        h = z - 71000
        T_avg = (270.65 + T) / 2
        p = p_71 * np.exp(-g0 * h / (R_air * T_avg))
        rho = p / (R_air * T)

    else:
        layer = "Exosphere / Very Low Density Region"
        rho = 1.5e-5 * np.exp(-(z - 85000) / 6000)

    rho = max(rho, 1e-20)
    return {
        "altitude_m": z,
        "altitude_km": z_km,
        "layer": layer,
        "density_kg_m3": rho
    }


# Rangos definidos en el código, con inicio, fin y altitud de muestra (punto medio)
layers_info = [
    ("Troposphere",                    0,     11000,  5500),
    ("Lower Stratosphere",         11000,     20000, 15500),
    ("Upper Stratosphere",         20000,     32000, 26000),
    ("Mesosphere",                 32000,     47000, 39500),
    ("Upper Mesosphere",           47000,     51000, 49000),
    ("Lower Thermosphere",         51000,     71000, 61000),
    ("Upper Thermosphere",         71000,     85000, 78000),
    ("Exosphere / Very Low Density Region", 85000, None, 90000),
]

print(f"{'Rango (km)':>14} | {'Capa':<35} | {'Muestra (km)':>12} | {'Densidad (kg/m³)':>18}")
print("-" * 90)
for layer_name, z_start, z_end, z_sample in layers_info:
    r = density_at_altitude(z_sample)
    rango = f"{z_start/1000:.0f} - {z_end/1000:.0f}" if z_end else f"{z_start/1000:.0f} - inf"
    print(f"{rango:>14} | {layer_name:<35} | {z_sample/1000:>12.1f} | {r['density_kg_m3']:>18.6e}")