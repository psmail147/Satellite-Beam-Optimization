# CI-SAT-BEAM-OPT

Computational-Intelligence metaheuristic optimization (via **Mealpy**) for **quantized beamforming** of a small ground **phased array** tracking a **LEO satellite**.

- Electronics: phased array with B-bit phase shifters, optional amplitude control, element gain/phase errors.
- Satellites: time-varying az/el pass and simple link budget.
- Optimization: GA/DE/PSO (from **Mealpy**) to tune quantized weights over the pass.
- Python-first: NumPy/SciPy/Matplotlib, YAML-config experiments.

## Quick start

1. Create a Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run a toy experiment (4x4 array, 3-bit quantization, GA optimizer):
   ```bash
   python run_experiment.py --config configs/example_ga.yaml
   ```

3. Results:
   - JSON logs and best solutions in `results/`
   - Plots in `plots/`

> If you don't have *mealpy* yet, install it with `pip install mealpy`.
> Repo structure is config-driven; tweak YAMLs in `configs/`.

## Project layout

```
ci_sat_beam_opt/
  __init__.py
  array_model.py
  pass_generator.py
  objectives.py
  baselines.py
  utils.py
  optimizers/
    mealpy_wrappers.py
run_experiment.py
configs/
  example_ga.yaml
  example_de.yaml
  example_pso.yaml
plots/               # generated
results/             # generated
tests/
  test_array_model.py
```

## Research knobs

- Array size (N), spacing, element pattern
- Phase bits B (2–4), amplitude levels
- Interferer geometry
- Error models (gain/phase jitter, element failures)
- Objectives: SNR (avg/5th-percentile), sidelobe level, retune cost
- Optimizer choice + hyperparams

## License

MIT
