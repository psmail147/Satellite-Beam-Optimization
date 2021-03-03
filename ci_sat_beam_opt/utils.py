import numpy as np

def db(x, floor_db=-200.0):
    """Convert linear power to dB with a floor."""
    x = np.maximum(x, 10**(floor_db/10.0))
    return 10*np.log10(x)

def lin_from_db(x_db):
    return 10**(x_db/10.0)

def wrap_phase(x):
    """Wrap radians to [-pi, pi)."""
    return (x + np.pi) % (2*np.pi) - np.pi

def quantize_phase(phases_rad, bits):
    levels = 2**bits
    q = np.round((phases_rad%(2*np.pi)) / (2*np.pi/levels)) % levels
    return q * (2*np.pi/levels)

def seed_all(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)

def radec_to_unit(theta_rad, phi_rad):
    """Spherical to unit vector."""
    st = np.sin(theta_rad); ct = np.cos(theta_rad)
    sp = np.sin(phi_rad);   cp = np.cos(phi_rad)
    return np.array([st*cp, st*sp, ct])
