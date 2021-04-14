import os, json, argparse, yaml
import numpy as np
import matplotlib.pyplot as plt

from ci_sat_beam_opt.array_model import PlanarArray
from ci_sat_beam_opt.pass_generator import synthetic_pass, interferers as gen_interferers
from ci_sat_beam_opt.baselines import static_beam, naive_per_snapshot
from ci_sat_beam_opt.objectives import snr_time_series, sidelobe_metric, retune_cost, aggregate_objective
from ci_sat_beam_opt.optimizers.mealpy_wrappers import make_fitness, run_mealpy, _decode_candidate

def plot_snr(t, snr_opt, snr_naive, snr_static, out_png):
    plt.figure()
    plt.plot(t, 10*np.log10(snr_opt+1e-20), label="Optimized")
    plt.plot(t, 10*np.log10(snr_naive+1e-20), label="Naive per-snapshot")
    plt.plot(t, 10*np.log10(snr_static+1e-20), label="Static beam")
    plt.xlabel("Time (s)")
    plt.ylabel("SNR (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def beampattern_slice(array, weights, out_png, az_samples=181, el_deg=10.0):
    az = np.linspace(0, 2*np.pi, az_samples)
    el = np.deg2rad(el_deg) * np.ones_like(az)
    p = []
    for a in az:
        af = array.array_factor(a, el[0], weights, apply_random_errors=False)
        p.append(np.abs(af)**2)
    p = np.array(p)
    plt.figure()
    plt.plot(np.rad2deg(az), 10*np.log10(p+1e-20))
    plt.xlabel("Azimuth (deg) at El={}°".format(int(el_deg)))
    plt.ylabel("Normalized Power (dB)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg.get("seed", 42))

    arr = PlanarArray(
        nx=cfg["array"]["nx"],
        ny=cfg["array"]["ny"],
        fc=cfg["array"]["fc"],
        dx=cfg["array"].get("dx", None),
        dy=cfg["array"].get("dy", None),
        phase_bits=cfg["array"]["phase_bits"],
        amp_levels=cfg["array"].get("amp_levels", 1),
        elem_gain_db=cfg["array"].get("elem_gain_db", 0.0),
        sigma_phase_deg=cfg["array"].get("sigma_phase_deg", 0.0),
        sigma_gain_db=cfg["array"].get("sigma_gain_db", 0.0),
    )

    t, az, el = synthetic_pass(duration_s=cfg["pass"]["duration_s"],
                               dt=cfg["pass"]["dt"],
                               max_el_deg=cfg["pass"]["max_el_deg"],
                               seed=cfg.get("seed", 42))

    forb = gen_interferers(cfg["interferers"]["num"]) if cfg["interferers"]["num"] > 0 else []

    # Baselines
    w_static = static_beam(arr, az0=az[len(az)//2], el0=el[len(el)//2], T=len(t))
    w_naive  = naive_per_snapshot(arr, az, el)

    # Genome / bounds
    T = len(t)
    N = arr.N
    L = arr.amp_levels
    genome_len_per_t = N*(1 + (1 if L>1 else 0))
    dims = genome_len_per_t * T
    lb = [0.0]*dims
    ub = [1.0]*dims

    # Fitness
    fitness = make_fitness(
        arr, T, az, el, forbidden_dirs=forb,
        alpha=cfg["objective"]["alpha"],
        beta=cfg["objective"]["beta"],
        gamma=cfg["objective"]["gamma"],
        noise_power_dbm=cfg["channel"]["noise_power_dbm"],
        interferers=forb,
        interferer_lin_power=10**(cfg["channel"]["interferer_power_dbm"]/10.0) * 1e-3,
        apply_errors=cfg["channel"].get("apply_errors", True)
    )

    # Optimize
    opt_name = cfg["optimizer"]["name"]
    pop = cfg["optimizer"]["pop"]
    iters = cfg["optimizer"]["iters"]
    seed = cfg.get("seed", 42)

    opt, gbest = run_mealpy(opt_name, fitness, lb, ub, dims, pop=pop, iters=iters, seed=seed)

    # Decode best
    x = np.array(gbest.solution)
    weights_opt = []
    ptr = 0
    for _ in range(T):
        genes = x[ptr:ptr+genome_len_per_t]
        w = _decode_candidate(genes, N, arr.amp_levels, arr.amp_codebook, arr.phase_bits)
        w[arr.failed] = 0.0
        weights_opt.append(w)
        ptr += genome_len_per_t

    # Evaluate
    snr_opt = snr_time_series(arr, az, el, weights_opt,
                              noise_power_dbm=cfg["channel"]["noise_power_dbm"],
                              interferers=forb,
                              interferer_lin_power=10**(cfg["channel"]["interferer_power_dbm"]/10.0) * 1e-3,
                              apply_errors=False)
    snr_naive = snr_time_series(arr, az, el, w_naive,
                                noise_power_dbm=cfg["channel"]["noise_power_dbm"],
                                interferers=forb,
                                interferer_lin_power=10**(cfg["channel"]["interferer_power_dbm"]/10.0) * 1e-3,
                                apply_errors=False)
    snr_static = snr_time_series(arr, az, el, w_static,
                                 noise_power_dbm=cfg["channel"]["noise_power_dbm"],
                                 interferers=forb,
                                 interferer_lin_power=10**(cfg["channel"]["interferer_power_dbm"]/10.0) * 1e-3,
                                 apply_errors=False)

    sll_series_opt = [sidelobe_metric(arr, w, forb, apply_errors=False) for w in weights_opt]
    rcost_opt = retune_cost(weights_opt)
    obj_opt, snr_avg_opt, sll_max_opt = aggregate_objective(snr_opt, sll_series_opt, rcost_opt,
                                                            alpha=cfg["objective"]["alpha"],
                                                            beta=cfg["objective"]["beta"],
                                                            gamma=cfg["objective"]["gamma"])

    # Save
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    tag = cfg.get("tag", opt_name.lower())
    summary = {
        "optimizer": opt_name,
        "dims": dims, "pop": pop, "iters": iters,
        "snr_avg_opt_lin": float(np.mean(snr_opt)),
        "snr_avg_opt_db": float(10*np.log10(np.mean(snr_opt)+1e-20)),
        "sll_max_opt_lin": float(np.max(sll_series_opt)),
        "retune_cost": float(rcost_opt),
        "objective_value": float(obj_opt),
    }
    with open(os.path.join("results", f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_snr(t, snr_opt, snr_naive, snr_static, out_png=os.path.join("plots", f"snr_{tag}.png"))
    beampattern_slice(arr, weights_opt[len(weights_opt)//2], out_png=os.path.join("plots", f"beampattern_{tag}.png"))

    print("Saved results to results/ and plots/.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/example_ga.yaml",
                    help="Path to YAML config (default: configs/example_ga.yaml)")
    args = ap.parse_args()
    if not os.path.exists(args.config):
        raise SystemExit(f"Config not found: {args.config}")
    main(args.config)
