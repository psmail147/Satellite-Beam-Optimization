import os
import json
import math
import yaml
from run_experiment import main as run_one  # reuses your existing pipeline

def ensure_dirs():
    os.makedirs("configs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def load_summary(tag):
    """Read one run's JSON summary and add derived SLL in dB."""
    path = os.path.join("results", f"summary_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        d = json.load(f)
    d["sll_max_opt_db"] = 10.0 * math.log10(float(d["sll_max_opt_lin"]) + 1e-20)
    d["tag"] = tag
    return d

def save_csv(rows, path="results/aggregate.csv"):
    """Write a simple CSV with the most useful fields."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k, "")
                vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
            f.write(",".join(vals) + "\n")

def print_table(rows, cols):
    """Print a readable fixed-width table into the Run console."""
    widths = [max(len(str(r.get(c, ""))) for r in rows + [{c: c}]) for c in cols]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))

# Base configuration (copied and tweaked to create multiple runs)
BASE = {
    "seed": 42,
    "tag": "default",
    "array": {
        "nx": 4,
        "ny": 4,
        "fc": 435000000.0,  # float to avoid YAML issues
        "phase_bits": 3,
        "amp_levels": 1,
        "sigma_phase_deg": 0.0,
        "sigma_gain_db": 0.0
    },
    "pass": {"duration_s": 480, "dt": 30.0, "max_el_deg": 45.0},
    "interferers": {"num": 2},
    "channel": {"noise_power_dbm": -100, "interferer_power_dbm": -80, "apply_errors": False},
    "objective": {"alpha": 1.0, "beta": 0.5, "gamma": 0.1},
    "optimizer": {"name": "GA", "pop": 24, "iters": 50}
}

def make_config(tag, **overrides):
    """Create a config by copying BASE and applying nested overrides."""
    cfg = json.loads(json.dumps(BASE))  # deep copy
    for k, v in overrides.items():
        if "__" in k:             # nested override: e.g., "array__phase_bits" -> cfg["array"]["phase_bits"]
            top, sub = k.split("__", 1)
            cfg[top][sub] = v
        else:
            cfg[k] = v
    cfg["tag"] = tag
    return cfg

def build_suite():
    """Define a compact set of experiments worth discussing."""
    suite = []

    # Quantization sweep
    suite += [
        make_config("ga_bits2", array__phase_bits=2),
        make_config("ga_bits3", array__phase_bits=3),
        make_config("ga_bits4", array__phase_bits=4),
    ]

    # Sidelobe penalty sweep
    suite += [
        make_config("beta0",  objective__beta=0.0),
        make_config("beta05", objective__beta=0.5),
        make_config("beta1",  objective__beta=1.0),
        make_config("beta2",  objective__beta=2.0),
    ]

    # Robustness (hardware errors on/off)
    suite += [
        make_config("robust_clean", array__sigma_phase_deg=0.0, array__sigma_gain_db=0.0, channel__apply_errors=False),
        make_config("robust_noisy", array__sigma_phase_deg=2.0, array__sigma_gain_db=0.5, channel__apply_errors=True),
    ]

    # Optimizer comparison
    suite += [
        make_config("cmp_ga",  optimizer__name="GA"),
        make_config("cmp_de",  optimizer__name="DE"),
        make_config("cmp_pso", optimizer__name="PSO"),
    ]
    return suite

if __name__ == "__main__":
    ensure_dirs()

    # Build and write configs, then run each experiment (one by one)
    suite = build_suite()
    for cfg in suite:
        cfg_path = os.path.join("configs", f"{cfg['tag']}.yaml")
        write_yaml(cfg_path, cfg)
        print(f"\n=== Running {cfg['tag']} ===")
        run_one(cfg_path)

    # Aggregate the per-run JSON summaries into one table + CSV
    rows = []
    for cfg in suite:
        d = load_summary(cfg["tag"])
        if d:
            rows.append({
                "tag": d["tag"],
                "optimizer": d["optimizer"],
                "snr_avg_db": d["snr_avg_opt_db"],
                "sll_max_db": d["sll_max_opt_db"],
                "retune_cost": d["retune_cost"]
            })

    save_csv(rows, path="results/aggregate.csv")
    print("\n=== Aggregate (results/aggregate.csv) ===")
    print_table(rows, cols=["tag", "optimizer", "snr_avg_db", "sll_max_db", "retune_cost"])

    # Convenience: grouped views you can screenshot or share
    bits = [r for r in rows if r["tag"] in {"ga_bits2", "ga_bits3", "ga_bits4"}]
    if bits:
        print("\n--- Quantization sweep ---")
        print_table(sorted(bits, key=lambda r: r["tag"]),
                    cols=["tag", "snr_avg_db", "sll_max_db", "retune_cost"])

    beta = [r for r in rows if r["tag"] in {"beta0", "beta05", "beta1", "beta2"}]
    if beta:
        print("\n--- Sidelobe-penalty sweep ---")
        order = {"beta0":0, "beta05":1, "beta1":2, "beta2":3}
        beta = sorted(beta, key=lambda r: order.get(r["tag"], 9))
        print_table(beta, cols=["tag", "snr_avg_db", "sll_max_db", "retune_cost"])

    rob = [r for r in rows if r["tag"] in {"robust_clean", "robust_noisy"}]
    if rob:
        print("\n--- Robustness ---")
        print_table(sorted(rob, key=lambda r: r["tag"]),
                    cols=["tag", "snr_avg_db", "sll_max_db", "retune_cost"])

    cmpv = [r for r in rows if r["tag"] in {"cmp_ga", "cmp_de", "cmp_pso"}]
    if cmpv:
        print("\n--- Optimizers ---")
        order = {"cmp_ga":0, "cmp_de":1, "cmp_pso":2}
        cmpv = sorted(cmpv, key=lambda r: order.get(r["tag"], 9))
        print_table(cmpv, cols=["tag", "snr_avg_db", "sll_max_db", "retune_cost"])
