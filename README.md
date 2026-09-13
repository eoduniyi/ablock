# A-Block: Clifford Neural Score Matching for Multi-Well Conformational Dynamics and Active Particle Sampling

Official anonymous code repository for the paper:  
**"A-Block: Clifford Neural Score Matching for Multi-Well Conformational Dynamics and Active Particle Sampling"**  
*Submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026).*

---

## Overview

Sampling rugged, multi-basin conformational landscapes without falling into kinetic traps is a long-standing challenge in molecular dynamics and generative modeling. Standard Euclidean Langevin samplers rely purely on gradient descent ($\mathbf{f} = -\nabla V$), causing ensembles to suffer kinetic arrest in local potential wells.

**A-Block** introduces an algebraic framework uniting **Geometric Algebra** $\text{Cl}(2,0)$ with **Denoising Score Matching (DSM)** and active matter Langevin hydrodynamics:

$$
\mathbf{f}_i = (\sigma_i I - \kappa_i) \nabla V_\theta(\mathbf{x}_i)
$$

- **Contractive Dilator ($-\kappa \nabla V_\theta$):** Stabilizes particles onto learned score isolines.
- **Transverse Rotor ($\sigma I \nabla V_\theta$):** Injects non-conservative, orthogonal circulation along the energy contours to cross steric barriers without modifying the underlying energy landscape.
- **Linear Complexity ($O(N)$):** Evaluates independent local score fields per particle, replacing quadratic $O(N^2)$ pairwise proximity graphs.

---

## Repository Structure

```
ablock-reproducibility/
├── README.md                          # Complete replication guide
├── requirements.txt                   # Minimal pinned dependencies
├── setup.py                           # Package installation script
├── reproduce_table1_and_figure1.py    # Generates Table 1 & Figure 1
├── reproduce_typography_figure3.py    # Generates Figure 3 (Typography benchmark)
├── src/
│   ├── potentials.py                  # 3-Basin potential & non-convex glyph surfaces
│   ├── models.py                      # Cl(2,0) CliffordScoreNet & rotor-dilator operator
│   ├── training.py                    # Denoising Score Matching (DSM) trainer
│   ├── samplers.py                    # SDE integrators (Brownian, Langevin, Rotor, A-Block)
│   └── metrics.py                     # FES RMSE, Basin Coverage, In-glyph Precision
├── tests/
│   └── test_pipeline.py               # Unit tests verifying mathematical correctness
└── output/                            # Destination for generated benchmark figures and JSONs
```

---

## Installation & Setup

### Environment Setup
```bash
git clone <ANONYMOUS_REPO_URL>
cd ablock-reproducibility
pip install -r requirements.txt
pip install -e .
```

### Dependency Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.22.0
- SciPy >= 1.8.0
- Matplotlib >= 3.5.0
- Pillow >= 9.0.0
- PyTest >= 7.0.0

---

## Reproducing Paper Results

All benchmarks are fully deterministic and run on standard commodity CPUs without requiring GPU acceleration. Total reproduction time is **under 3 minutes**.

### 1. Reproducing Table 1 & Figure 1 (Multi-Well Conformational Dynamics)
Simulates $N=1{,}200$ particles over $T=6{,}000$ integration steps on a canonical 3-basin landscape (Basins A, B, C) seeded from a local kinetic trap in Basin A:
```bash
python reproduce_table1_and_figure1.py
```
**Expected Outputs:**
- `output/table1_results.json`: Quantitative benchmark table recording Free Energy Surface (FES) RMSE, basin coverage, and runtime.
- `output/figure1_annotated_benchmark.pdf` & `.png`: High-resolution 4-panel figure showing:
  - (a) Brownian Dynamics (f = 0): Unbounded isotropic diffusion.
  - (b) Standard Langevin ($f = -\nabla V$): Kinetic arrest in Basin A ($97.2\%$ trapping).
  - (c) Rotor-Only ($\kappa = 0$): Unstable centrifugal dispersion into high-energy regions.
  - (d) A-Block Full Clifford ($f = (\sigma I - \kappa)\nabla V$): Optimal thermodynamic fidelity ($\mathrm{RMSE} = 1.059\text{ kcal/mol}$) and complete multi-basin coverage.

### 2. Reproducing Figure 3 (Non-Convex Typography Kinetic Trap Benchmark)
Tests non-conservative rotor circulation in navigating high-curvature, non-convex steric funnels on the letterform `'G'` seeded in the sharp inner spur:
```bash
python reproduce_typography_figure3.py
```
**Expected Outputs:**
- `output/typography_metrics.json`: Records In-Glyph Precision (%) and Stroke Coverage (%).
- `output/figure3_typography_benchmark.pdf` & `.png`: 4-panel figure comparing:
  - Brownian: Precision 18.7%, Coverage 42.1% (unbounded diffusion into void).
  - Langevin: Precision 99.3%, Coverage 33.9% (severe kinetic arrest in spur).
  - Rotor-Only: Precision 0.2%, Coverage 2.9% (unstable centrifugal boundary leakage).
  - A-Block Full Clifford: Precision 99.8%, Coverage 97.8% (active confinement, robust non-convex exploration).

---

## Automated Verification Tests

Run the unit test suite to verify the algebraic rotor-dilator orthogonality, potential gradients, and metric calculations:
```bash
pytest tests/ -v
```

---

## Hardware & Computational Efficiency

| Benchmark | CPU Hardware | Core Count | Integration Steps | Total Runtime | Per-Step Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Multi-Well (Table 1) | Apple M-Series / x86_64 | 4 cores | 6,000 | ~45 sec | $O(N)$ Linear |
| Typography (Figure 3) | Apple M-Series / x86_64 | 4 cores | 6,000 | ~35 sec | $O(N)$ Linear |

---

## Double-Blind Anonymity Notice

In strict adherence to the ML4PS 2026 double-blind review policy, all identifying author information, institutional affiliations, and local repository metadata have been sanitized. Anonymized code sharing is hosted via Anonymous GitHub.

## License
This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
