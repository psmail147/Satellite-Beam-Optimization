import numpy as np

def synthetic_pass(duration_s=720, dt=10.0, max_el_deg=45.0, seed=0):
    """
    Generate a smooth synthetic pass with az/el over time.
    Returns t, az_rad[t], el_rad[t].
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s+1e-9, dt)
    # Azimuth: sweep across sky
    az0 = rng.uniform(0, 2*np.pi)
    az1 = (az0 + np.pi + rng.uniform(-0.4, 0.4)) % (2*np.pi)
    az = np.linspace(az0, az1, t.size)
    # Elevation: bell-shaped (0 at ends, max at middle)
    mid = t.size//2
    el_peak = np.deg2rad(max_el_deg)
    el = el_peak * np.exp(-0.5 * ((np.arange(t.size)-mid)/(0.35*t.size))**2)
    return t, az, el

def interferers(num=1):
    """
    Place static interferers near horizon at random azimuths.
    Returns list of (az_rad, el_rad).
    """
    rng = np.random.default_rng(123)
    out = []
    for _ in range(num):
        az = rng.uniform(0, 2*np.pi)
        el = rng.uniform(np.deg2rad(0.0), np.deg2rad(10.0))
        out.append((az, el))
    return out
