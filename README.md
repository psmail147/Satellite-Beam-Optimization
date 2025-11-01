# Satellite Beam Optimization

This project applies **Computational Intelligence Optimization** techniques to improve the signal performance of a simulated satellite antenna array.  
Developed in **Python** using the [Mealpy](https://github.com/thieu1995/mealpy) framework, it demonstrates how **evolutionary and swarm-based algorithms** can optimize beam direction for stronger signals and lower interference.

---

## Project Overview

Satellite antennas use multiple small transmitters whose combined signals form a directional beam.  
The goal of this project is to **maximize signal-to-noise ratio (SNR)** while **minimizing sidelobe interference**, by tuning the phase values of each transmitter.  

Three main optimizers were tested:
- **Genetic Algorithm (GA)**
- **Differential Evolution (DE)**
- **Particle Swarm Optimization (PSO)**

Each optimizer searches for the best vector of phase settings that produce the desired beam pattern.

---

## ⚙️ Methods

- Implemented in Python with modular design (`array_model`, `optimizers`, `run_experiment`).
- Parameter sweeps were conducted to study:
  - **Quantization levels** (2-bit, 3-bit, 4-bit)
  - **Sidelobe penalties (β = 0, 0.5, 1, 2)**
  - **Optimizer comparison** (GA, DE, PSO)
- Each configuration was run for 50 iterations with 24 agents.
- Results were automatically saved as `.json`, `.csv`, and `.png` plots.

---

## Results Summary

| Experiment | Optimizer | Avg. SNR (dB) | Max. Sidelobe (dB) | Retune Cost |
|-------------|------------|----------------|--------------------|--------------|
| GA (2-bit)  | GA | 116.73 | -6.24 | 1.42 |
| GA (3-bit)  | GA | 118.31 | -6.56 | 1.35 |
| GA (4-bit)  | GA | 117.67 | -5.81 | 1.37 |
| β = 0.5     | GA | **119.12** | **-7.49** | 1.40 |
| DE          | DE | 116.85 | -6.88 | 1.43 |
| PSO         | PSO | 116.81 | **-9.21** | 1.46 |

**Key findings:**
- Moderate sidelobe penalties (β ≈ 0.5) produced the best balance between beam strength and interference.
- 3-bit quantization achieved near-optimal performance with minimal cost.
- PSO achieved the strongest sidelobe suppression, while GA reached the highest overall SNR.

---

## Learning Outcome

This project demonstrates how metaheuristic optimization can efficiently handle nonlinear, high-dimensional search problems common in engineering.  
It also showcases practical use of Mealpy for rapid algorithm prototyping and comparative benchmarking.

---
## Project Context

This project was completed as part of a **Computational Intelligence Optimization** at DMU, illustrating the application of evolutionary algorithms to real-world signal optimization.


