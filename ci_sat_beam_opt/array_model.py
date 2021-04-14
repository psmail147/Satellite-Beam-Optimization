import numpy as np
from .utils import wrap_phase, quantize_phase

class PlanarArray:
    """
    Rectangular planar array with optional amplitude control and quantized phase shifters.
    Coordinates: x (columns), y (rows), element (i,j) at r_ij = [i*dx, j*dy, 0].
    """

    def __init__(self, nx=4, ny=4, fc=435e6, dx=None, dy=None, c=299792458.0,
                 phase_bits=3, amp_levels=1, elem_gain_db=0.0,
                 sigma_phase_deg=0.0, sigma_gain_db=0.0, failed_mask=None):
        self.nx = int(nx)
        self.ny = int(ny)
        self.N = self.nx * self.ny
        self.fc = float(fc)
        self.c = float(c)
        self.k = 2 * np.pi * self.fc / self.c
        lam = self.c / self.fc
        self.dx = float(dx) if dx is not None else 0.5 * lam
        self.dy = float(dy) if dy is not None else 0.5 * lam
        self.phase_bits = int(phase_bits)
        self.amp_levels = int(amp_levels)
        self.elem_gain_db = float(elem_gain_db)
        self.sigma_phase = np.deg2rad(float(sigma_phase_deg))
        self.sigma_gain = float(sigma_gain_db)

        # element positions
        xs = np.arange(nx) - (nx-1)/2
        ys = np.arange(ny) - (ny-1)/2
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        self.r = np.stack([X.flatten()*self.dx, Y.flatten()*self.dy, np.zeros(self.N)], axis=1)

        # failures
        if failed_mask is None:
            self.failed = np.zeros(self.N, dtype=bool)
        else:
            self.failed = failed_mask.astype(bool).flatten()
            assert self.failed.size == self.N

        # amplitude codebook (uniform)
        if amp_levels <= 1:
            self.amp_codebook = np.array([1.0])
        else:
            self.amp_codebook = np.linspace(0.5, 1.0, amp_levels)

    def steering_vector(self, az_rad, el_rad):
        """
        Unit direction from az/el (azimuth from x-axis in xy-plane, elevation from horizon).
        Returns exp(j k r·s).
        """
        ca, sa = np.cos(az_rad), np.sin(az_rad)
        ce, se = np.cos(el_rad), np.sin(el_rad)
        s = np.array([ce*ca, ce*sa, se])  # (3,)
        phase = self.k * (self.r @ s)     # (N,)
        return np.exp(1j*phase)           # (N,)

    def apply_errors(self, weights):
        """Apply per-element random phase and gain errors (one draw per call)."""
        if self.sigma_phase > 0.0:
            dphi = np.random.normal(0.0, self.sigma_phase, size=self.N)
        else:
            dphi = 0.0
        if self.sigma_gain != 0.0:
            g_db = np.random.normal(0.0, self.sigma_gain, size=self.N)
            g = 10**(g_db/20.0)
        else:
            g = 1.0
        err = g*np.exp(1j*dphi)
        return weights * err

    def quantize_weights(self, w_complex):
        """Quantize phases to phase_bits and map amplitudes to nearest codebook."""
        amps = np.abs(w_complex)
        phases = np.angle(w_complex) % (2*np.pi)
        qph = quantize_phase(phases, self.phase_bits)
        if self.amp_levels > 1:
            idx = np.argmin(np.abs(amps[:,None] - self.amp_codebook[None,:]), axis=1)
            qamp = self.amp_codebook[idx]
        else:
            qamp = np.ones_like(amps)
        return qamp * np.exp(1j*qph)

    def array_factor(self, az_rad, el_rad, weights, apply_element_gain=True, apply_random_errors=True):
        """
        Compute complex array factor at (az,el) for given (possibly quantized) weights.
        """
        w = weights.copy()
        if apply_random_errors:
            w = self.apply_errors(w)
        sv = self.steering_vector(az_rad, el_rad)
        af = np.vdot(w, sv) / self.N  # normalized
        if apply_element_gain and self.elem_gain_db != 0.0:
            af = af * 10**(self.elem_gain_db/20.0)
        return af

    def steer_ideal(self, az_rad, el_rad):
        """Ideal continuous weights for beam pointing to (az,el): conjugate of steering vector."""
        sv = self.steering_vector(az_rad, el_rad)
        w = np.conj(sv)
        return w / np.linalg.norm(w)
