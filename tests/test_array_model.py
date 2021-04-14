import numpy as np
from ci_sat_beam_opt.array_model import PlanarArray

def test_steer_and_quantize():
    arr = PlanarArray(nx=2, ny=2, fc=1e9, phase_bits=2)
    w = arr.steer_ideal(0.0, np.deg2rad(30.0))
    wq = arr.quantize_weights(w)
    assert wq.shape[0] == 4
    assert np.all(np.isfinite(wq))
