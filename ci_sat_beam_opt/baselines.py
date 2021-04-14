import numpy as np

def static_beam(array, az0, el0, T):
    w = array.steer_ideal(az0, el0)
    wq = array.quantize_weights(w)
    return [wq for _ in range(T)]

def naive_per_snapshot(array, az_series, el_series):
    out = []
    for az, el in zip(az_series, el_series):
        w = array.steer_ideal(az, el)
        out.append(array.quantize_weights(w))
    return out
