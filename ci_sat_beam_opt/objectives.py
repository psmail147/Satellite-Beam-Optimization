import numpy as np
from .utils import db, lin_from_db

def snr_time_series(array, az_series, el_series, weights_series,
                    noise_power_dbm=-100.0, interferers=None,
                    interferer_lin_power=1e-4, apply_errors=True):
    """
    Compute SNR(t) from |AF|^2 against AWGN + simple interferer model.
    - interferers: list of (az,el); each contributes power scaled by sidelobe response.
    """
    T = len(az_series)
    snr = np.zeros(T)
    N0_lin = 1e-3 * 10**(noise_power_dbm/10.0)  # W -> arbitrary units
    for k in range(T):
        w = weights_series[k]
        af_sig = array.array_factor(az_series[k], el_series[k], w, apply_random_errors=apply_errors)
        p_sig = np.abs(af_sig)**2

        p_int = 0.0
        if interferers:
            for (iaz, iel) in interferers:
                af_i = array.array_factor(iaz, iel, w, apply_random_errors=apply_errors)
                p_int += interferer_lin_power * (np.abs(af_i)**2)

        snr[k] = p_sig / (N0_lin + p_int)
    return snr

def sidelobe_metric(array, weights, forbidden_dirs, apply_errors=False):
    """
    Max sidelobe level (linear) across a set of forbidden directions (az,el).
    """
    if not forbidden_dirs:
        return 0.0
    vals = []
    for (az, el) in forbidden_dirs:
        af = array.array_factor(az, el, weights, apply_random_errors=apply_errors)
        vals.append(np.abs(af)**2)
    return float(np.max(vals))

def retune_cost(weights_series):
    """
    Penalize large changes between consecutive weights (phase difference).
    """
    if len(weights_series) < 2:
        return 0.0
    cost = 0.0
    for k in range(1, len(weights_series)):
        w0 = weights_series[k-1] / (np.linalg.norm(weights_series[k-1]) + 1e-12)
        w1 = weights_series[k]   / (np.linalg.norm(weights_series[k]) + 1e-12)
        d = np.linalg.norm(w1 - w0)
        cost += d
    return cost / (len(weights_series)-1)

def aggregate_objective(snr_series, sll_series, retune, alpha=1.0, beta=1.0, gamma=0.1):
    """
    Higher is better. We subtract penalties (beta * SLL, gamma * retune).
    """
    snr_avg = float(np.mean(snr_series))
    sll_max = float(np.max(sll_series)) if len(sll_series) > 0 else 0.0
    return alpha*snr_avg - beta*sll_max - gamma*retune, snr_avg, sll_max
