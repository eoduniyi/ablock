# A-Block: Clifford Neural Score Matching for Multi-Well Conformational Dynamics and Active Particle Sampling

[![ML4PS 2026](https://img.shields.io/badge/NeurIPS_2026-ML4PS_Accepted-059669.svg)](https://openreview.net/forum?id=t1UaaGEPJB)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg)](https://www.python.org/)

Official code, trained weights, interactive visualizers, and reproducibility repository for the paper:  
**"A-Block: Clifford Neural Score Matching for Multi-Well Conformational Dynamics and Active Particle Sampling"**  
**Erick Oduniyi** (*Massachusetts Institute of Technology*)  
*Accepted at the Machine Learning and the Physical Sciences (ML4PS) Workshop at NeurIPS 2026.*

---

## Overview

Sampling rugged, multi-basin conformational landscapes without suffering kinetic trapping is a central challenge in molecular dynamics and score-based generative modeling. Standard Euclidean Langevin samplers follow purely conservative gradients ($\mathbf{f} = -\nabla V_\theta$), frequently arresting in deep local wells or over-concentrating when score scales are uncalibrated.

**A-Block** (Algebraic Block) unifies **2D Clifford Geometric Algebra** $\text{Cl}(2,0)$, **non-reversible score-based SDEs**, and **active matter hydrodynamics** via a coupled multivector rotor-dilator drift operator:

$$
\mathbf{f}_i(\mathbf{x}_i) = (\sigma_i I - \kappa_i) \nabla V_\theta(\mathbf{x}_i), \qquad I = e_1 e_2 \; (I^2 = -1)
$$

- **Fokker–Planck Stationarity vs. Active Confinement** (Proposition 1):
  - *Measure-Preserving Non-Reversible Regime:* When $\kappa$ calibrates the dimensionless DSM score scale ($\kappa = \alpha_E^{-1} = k_B T$) and $\sigma = \sigma(V_\theta(\mathbf{x}))$, the transverse rotor current $\mathbf{J}_{\text{irr}} = \gamma^{-1}\sigma(V_\theta) I \nabla V_\theta\,\rho_{\text{B}}$ is strictly divergence-free ($\nabla \cdot \mathbf{J}_{\text{irr}}(\rho_{\text{B}}) = 0$), preserving the exact Boltzmann distribution while accelerating saddle traversal.
  - *Non-Equilibrium Steady State (NESS) Regime:* Driving $(\sigma, \kappa)$ outside fluctuation-dissipation creates an active steady state with persistent boundary circulation for programmable spatial confinement in non-convex geometries.
- **Linear Scaling** (Proposition 2): Replacing pairwise proximity graphs ($O(N^2)$) with empirical point measures $\mu_t$ yields $O(N)$ per-step scaling, while direct-sum planar dihedral algebras $\bigoplus_{k=1}^K \text{Cl}(2,0)$ scale linearly as $4K$ rather than $2^{2K}$.

---

## Repository Structure

```
ablock/
├── README.md                          # Overview, quickstart, and benchmark summary
├── PROVENANCE.md                      # Complete parameter, weight, and figure provenance
├── REPRODUCIBILITY.md                 # Detailed mathematical and empirical walkthrough
├── CITATION.cff                       # Citation metadata
├── requirements.txt                   # Minimal dependencies
├── setup.py                           # Installable Python package
├── reproduce_table1_and_figure1.py    # Reproduces Table 1, multi-seed ablations & Figure 1
├── reproduce_typography_figure3.py    # Reproduces standardized Figure 3 ('G' benchmark)
├── src/
│   ├── potentials.py                  # 3-Basin conformational potential & glyph surfaces
│   ├── models.py                      # Cl(2,0) CliffordScoreNet & rotor-dilator operator
│   ├── training.py                    # Denoising Score Matching (DSM) trainer
│   ├── samplers.py                    # Overdamped SDE integrators & Dilator-Only ablation
│   └── metrics.py                     # FES RMSE, Sliced W1, Basin Visitation & Precision
├── notebooks/
│   ├── 01_clifford_dsm_and_table1_tutorial.ipynb   # Tutorial 1: DSM, Proposition 1 & Table 1
│   └── 02_nonconvex_typography_navigation.ipynb    # Tutorial 2: Non-convex 'G' active matter
├── interactive/
│   ├── magic_apertures/               # p5.js multi-unit "MAGIC" simulation (Figure 2 source)
│   └── proposition1_visualizer/       # p5.js interactive O(N^2) vs O(N) visualizer
├── paper/
│   ├── main.tex                       # Camera-ready 5-page LaTeX manuscript
│   ├── main.pdf                       # Compiled camera-ready PDF
│   └── reference.bib                  # BibTeX bibliography
├── output/
│   ├── clifford_score_net.pt          # Trained CliffordScoreNet PyTorch checkpoint
│   ├── table1_results.json            # Full Table 1 & 5-seed ablation metrics
│   ├── typography_metrics.json        # Standardized Figure 3 typography metrics
│   ├── figure1_annotated_benchmark.pdf
│   └── figure3_typography_benchmark.pdf
└── tests/
    └── test_pipeline.py               # Automated unit test suite
```

---

## Quickstart & Installation

```bash
git clone https://github.com/eoduniyi/ablock.git
cd ablock
pip install -r requirements.txt
pip install -e .
```

### Run Unit Tests
```bash
pytest tests/ -v
```

---

## Reproducing Paper Results

All experiments run deterministically on a standard CPU in **under 90 seconds**.

### 1. Multi-Well Conformational Dynamics & Component Ablations (Table 1 & Figure 1)
Trains `CliffordScoreNet` via Denoising Score Matching (saving `output/clifford_score_net.pt`) and evaluates $N = 2,000$ particles over $T = 1,000$ steps ($\Delta t = 0.015$) across 5 seeds:

```bash
python3 reproduce_table1_and_figure1.py
```

| Sampling Method / Ablation | FES RMSE (kcal/mol) $\downarrow$ | $\mathcal{W}_1$ Distance $\downarrow$ | Term. B/C Occ. (%) | Cumul. B/C Visit. (%) $\uparrow$ | Transitions / Pt. $\uparrow$ | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Brownian Dynamics** ($\mathbf{f} = \mathbf{0}$) | 2.572 ± 0.136 | 3.127 ± 0.035 | 2.0 ± 0.4% | 56.4 ± 1.0% | 1.74 ± 0.06 | 0.13 |
| **Standard Langevin** ($\kappa = 1, \sigma = 0$) | 1.476 ± 0.069 | **0.066 ± 0.008** | **60.9 ± 1.6%** | **99.7 ± 0.1%** | 21.47 ± 0.13 | 2.31 |
| **Dilator-Only Ablation** ($\sigma = 0, \kappa$) | 1.064 ± 0.044 | 0.266 ± 0.026 | 44.2 ± 1.5% | 96.9 ± 0.3% | 13.86 ± 0.15 | 2.20 |
| **A-Block Rotor-Only** ($\kappa = 0, \sigma$) | 3.028 ± 0.194 | 4.243 ± 0.017 | 0.4 ± 0.2% | 70.1 ± 0.8% | 2.82 ± 0.04 | 2.38 |
| **A-Block Full Clifford (Ours)** | **1.059 ± 0.031** | 0.385 ± 0.016 | 37.4 ± 0.7% | **99.7 ± 0.1%** | **22.48 ± 0.15** | 2.23 |

### 2. Standardized Non-Convex Typography Navigation (Figure 3)
Simulates $N = 1,200$ particles for $T = 4,000$ steps ($\Delta t = 2.0$) initialized in the sharp inner spur of the letterform `'G'` under identical stochastic noise (`noise_amp = 0.85`) and circular dish clamping ($r_{\text{dish}} = 176\text{ px}$):

```bash
python3 reproduce_typography_figure3.py
```

| Method | In-Glyph Precision (%) $\uparrow$ | Cumulative Trajectory Visitation (%) $\uparrow$ | Terminal Coverage (%) $\uparrow$ |
| :--- | :---: | :---: | :---: |
| **Brownian Dynamics** ($\mathbf{f} = \mathbf{0}$) | 26.1% | 99.3% | 90.5% |
| **Standard Langevin** ($\sigma = 0, \kappa = 1.0$) | 98.9% | 33.2% | 33.7% |
| **Rotor-Only** ($\sigma = 11.0, \kappa = 0$) | 0.1% | 100.0% | 1.4% |
| **A-Block Full Clifford** ($\sigma = 11.0, \kappa = 1.8$) | **99.9%** | **97.8%** | **99.2%** |

---

## Interactive Tutorials & Web Visualizers

- **Jupyter Notebooks** (`notebooks/`):
  - `notebooks/01_clifford_dsm_and_table1_tutorial.ipynb`: Step-by-step tutorial on `CliffordScoreNet`, DSM energy scale calibration, Fokker–Planck stationarity, and multi-well ablations.
  - `notebooks/02_nonconvex_typography_navigation.ipynb`: Step-by-step tutorial on constructing non-convex morphological potentials and simulating active rotor-dilator confinement.
- **Interactive p5.js Sketches** (`interactive/`):
  - Open `interactive/magic_apertures/index.html` in any browser to run the 5-aperture `"MAGIC"` ensemble (Figure 2).
  - Open `interactive/proposition1_visualizer/index.html` to interactively toggle between $O(N^2)$ proximity graphs and $O(N)$ Clifford point measures.

---

## Citation

```bibtex
@inproceedings{oduniyi2026ablock,
  title={A-Block: Clifford Neural Score Matching for Multi-Well Conformational Dynamics and Active Particle Sampling},
  author={Oduniyi, Erick},
  booktitle={Machine Learning and the Physical Sciences (ML4PS) Workshop at NeurIPS},
  year={2026},
  url={https://openreview.net/forum?id=t1UaaGEPJB}
}
```

## License
Licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).