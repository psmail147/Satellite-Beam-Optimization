import numpy as np
from types import SimpleNamespace
from mealpy.utils.space import FloatVar
from mealpy import Problem                         # 3.0.x has Problem
from mealpy.evolutionary_based import GA, DE       # standard module paths in 3.0.x
from mealpy.swarm_based import PSO

def _decode_candidate(x, N, amp_levels, amp_codebook, phase_bits):
    B = phase_bits
    L = amp_levels
    x = np.asarray(x)
    ptr = 0
    phase_codes = (x[ptr:ptr+N] * (2**B)).astype(int) % (2**B); ptr += N
    if L > 1:
        amp_codes = (x[ptr:ptr+N] * L).astype(int) % L; ptr += N
        amp = amp_codebook[amp_codes]
    else:
        amp = np.ones(N)
    phases = phase_codes * (2*np.pi/(2**B))
    return amp * np.exp(1j*phases)

def make_fitness(array, T, az_series, el_series, forbidden_dirs, alpha, beta, gamma,
                 noise_power_dbm, interferers, interferer_lin_power,
                 apply_errors, retune_weight=True):
    N = array.N
    L = array.amp_levels
    amp_codebook = array.amp_codebook
    genome_len_per_t = N*(1 + (1 if L>1 else 0))

    def f(x):
        x = np.asarray(x)
        assert x.size == genome_len_per_t*T
        weights_series = []
        ptr = 0
        for _ in range(T):
            genes = x[ptr:ptr+genome_len_per_t]
            w = _decode_candidate(genes, N, L, amp_codebook, array.phase_bits)
            w = w.copy()
            w[array.failed] = 0.0
            weights_series.append(w)
            ptr += genome_len_per_t

        from ..objectives import snr_time_series, sidelobe_metric, retune_cost, aggregate_objective
        snr_series = snr_time_series(array, az_series, el_series, weights_series,
                                     noise_power_dbm=noise_power_dbm,
                                     interferers=interferers,
                                     interferer_lin_power=interferer_lin_power,
                                     apply_errors=apply_errors)
        sll_series = [sidelobe_metric(array, w, forbidden_dirs, apply_errors=False) for w in weights_series]
        rcost = retune_cost(weights_series) if retune_weight else 0.0
        obj, _, _ = aggregate_objective(snr_series, sll_series, rcost, alpha=alpha, beta=beta, gamma=gamma)
        return -obj

    return f

def run_mealpy(optimizer_name, fitness, lb, ub, dims, pop=32, iters=100, seed=42):
    bounds = [FloatVar(lb=0.0, ub=1.0) for _ in range(dims)]

    # Build Problem with typed bounds
    prob = Problem(obj_func=fitness, bounds=bounds, minmax="min", name="BeamOpt", log_to=None, seed=seed)

    name = optimizer_name.upper()
    if name == "GA":
        Model = GA.BaseGA
    elif name == "DE":
        Model = DE.OriginalDE#DE.BaseDE
    elif name == "PSO":
        Model = PSO.OriginalPSO#PSO.BasePSO
    else:
        raise ValueError("Unknown optimizer: " + optimizer_name)

    opt = Model(epoch=iters, pop_size=pop, seed=seed)
    best = opt.solve(problem=prob)   # explicitly pass Problem

    from types import SimpleNamespace
    sol = getattr(best, "solution", getattr(best, "position", None))
    if sol is None:
        gb = getattr(opt, "g_best", getattr(opt, "gbest", None))
        sol = getattr(gb, "solution", getattr(gb, "position", None)) if gb is not None else None
    if sol is None:
        raise RuntimeError("Could not extract best solution from Mealpy result.")
    return opt, SimpleNamespace(solution=np.asarray(sol, dtype=float))
