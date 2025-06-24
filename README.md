# NitroFoodWeb – Isotope-based Reconstruction of Quantitative Food Webs

This repository contains a proof-of-concept pipeline for inferring a weighted
diet matrix **Q** (proportional feeding links) from

* a known binary food-web topology **A**  
* bulk δ¹⁵N values for every species.

The key idea is to search the space of admissible **Q** matrices so that the
fractional trophic levels implied by **Q** reproduce the observed isotope
pattern.

---

## Quick Start

1. **Clone the repo**

   ```bash
   git clone https://github.com/jdyeakel/NitroFoodWeb.git
   cd NitroFoodWeb
   # Launch Julia in project mode and instantiate
    julia --project -e '
        using Pkg
        Pkg.instantiate()     # downloads Distributions, DataFrames, Plots, …
    '
    julia --project nitrotest.jl
    ```


The script prints fit diagnostics and pops up three plots:

1. annealing error trace  
2. observed vs. estimated trophic levels  
3. true vs. estimated link weights (log–log).

---

## File overview

| file | what it does |
|------|--------------|
| **`nitrotest.jl`** | End-to-end demo: generates a synthetic niche-model web, assigns “true” diets, simulates δ¹⁵N, then recovers **Q̂** with simulated annealing and prints accuracy metrics. |
| **`src/estimate_Q_sa.jl`** | Simulated-annealing optimiser (adaptive *T₀* calibration, Dirichlet column moves, Metropolis acceptance). |
| **`src/evaluate_Q.jl`** | Calculates MAE, RMSE, Pearson *R*, mean K-L divergence, and per-consumer K-L scores between any two **Q** matrices. |
| **`src/nichemodelweb.jl`** | Generates a binary adjacency matrix via the classic Williams–Martinez niche model (`nichemodelweb(S, C)`). |
| **`src/quantitativeweb.jl`** | Converts a binary **A** into a weighted **Q** by drawing one Dirichlet distribution per consumer column (`quantitativeweb(A; alpha)`). |
| **`src/trophic.jl`** <br>(module *Trophic*) | • `InternalNetwork` – housekeeping for flow matrices <br>• `Diet` – converts flow matrix to column-normalised diets <br>• `TrophInd` – fractional TL + omnivory index following Pauly et al. (1998). |
| **`src/NitroFoodWeb.jl`** | Umbrella module re-exporting all public functions; makes `using NitroFoodWeb` work. |

---

## Workflow in *nitrotest.jl*

1. **Topology** – generate *(S = 100, C = 0.02)* niche model.  
2. **True diets** – `Q_true = quantitativeweb(A; alpha = 0.5)`

   The keyword `alpha` sets the concentration parameter of the symmetric
   Dirichlet used to draw each consumer’s diet column:

   | `alpha` value | diet breadth generated | ecological interpretation |
   |---------------|------------------------|---------------------------|
   | `alpha ≪ 1` (e.g. 0.05) | long-tailed: one or two prey dominate, many near-zero links | strong specialists |
   | `alpha = 1`  | “uninformative” prior—every possible composition is equally likely | mix of specialists & generalists |
   | `alpha > 1` (e.g. 5)  | weights cluster near `1 / (# prey)` | true generalists with even diets |

   In the demo we pick `alpha = 0.5`, so most consumers have a favourite prey
   but still include a few minor items in their diet.  
3. **Isotopes** – convert `TrophInd(Q_true)` to δ¹⁵N using Δ¹⁵N = 3.5 ‰.  
4. **Estimation** – call `estimate_Q_sa(A_bool, d15N_true)` to obtain **Q_est**.  
5. **Diagnostics** – plots + `evaluate_Q(Q_true, Q_est)` summary metrics.

Typical fit on the demo network (100 sp):

| metric | value |
|--------|-------|
| MAE (weights) | **0.014** |
| RMSE (weights)| 0.027 |
| Pearson *R*   | 0.98 |
| Mean KL-div.  | 0.032 |

---

## Requirements

* Julia ≥ 1.9  
* Packages: `NitroFoodWeb`, `Distributions`, `DataFrames`, `StatsBase`,
  `Plots`, `Random` (see `Project.toml`).

---

## Next steps

* Add measurement error to δ¹⁵N and test robustness.  
* Experiment with larger webs (S ≥ 300) or different Dirichlet priors (`alpha`).  
* Swap annealing for parallel tempering or CMA-ES (via *BlackBoxOptim.jl*).  
* Extend `evaluate_Q` with Jensen–Shannon divergence or χ² residuals.

Contributions and issue reports are welcome!  
