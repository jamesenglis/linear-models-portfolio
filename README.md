# Linear Models Portfolio

A hands-on portfolio exploring linear models in scikit-learn, with a focus on understanding the mathematics behind the code.

# Linear Models Portfolio

A 13-day, hands-on exploration of **every subsection of scikit-learn's User Guide §1.1 (Linear Models)**.

Each script follows the same structure:
1. **Toy example** — the scikit-learn docs snippet, typed by hand for muscle memory.
2. **Real data** — the same model on a real dataset, with practical observations.
3. **Math check** — a manual calculation (normal equations, soft-thresholding, posterior mean, etc.) that reproduces scikit-learn's output to verify I understand what the library is doing.
4. **Reflection prompts** — questions I answered inline.

The goal was not to memorize API calls. The goal was to **learn the math behind the code** by typing it, changing one number at a time, and verifying predictions against the output.

---

## Coverage

All 16 subsections of scikit-learn User Guide §1.1:

| Subsection | Topic | Script |
|---|---|---|
| 1.1.1 | Ordinary Least Squares | `day01_ols.py` |
| 1.1.2 | Ridge regression and classification | `day02_ridge.py` |
| — | RidgeCV (LOO cross-validation) | `day03_ridgecv.py` |
| 1.1.3 | Lasso | `day04_lasso.py` |
| 1.1.3.x | LassoCV + AIC/BIC | `day05_lassocv_ic.py` |
| 1.1.4–1.1.6 | ElasticNet + MultiTask variants | `day06_elasticnet.py` |
| 1.1.7–1.1.9 | LARS, LARS Lasso, OMP | `day07_lars_omp.py` |
| 1.1.10 | Bayesian Ridge, ARD | `day08_bayesian_ard.py` |
| 1.1.11 | Logistic Regression + solvers | `day09_logistic.py` |
| 1.1.12 | Generalized Linear Models (Tweedie) | `day10_glm_pipeline.py` |
| 1.1.16 | Polynomial regression + Pipeline | `day10_glm_pipeline.py` |
| 1.1.13 | SGD, Perceptron, Passive-Aggressive | `day11_sgd.py` |
| 1.1.14 | Robust regression (RANSAC, Huber, Theil-Sen) | `day12_robust.py` |
| 1.1.15 | Quantile regression | `day13_quantile.py` |

---

## The math, one line per model

| Model | Objective / key formula |
|---|---|
| OLS | `min ‖Xw − y‖²` → `w = (XᵀX)⁻¹Xᵀy` |
| Ridge | `min ‖Xw − y‖² + α‖w‖²` → `w = (XᵀX + αI)⁻¹Xᵀy` |
| RidgeCV | Pick `α` minimizing Leave-One-Out error via hat matrix `h_ii` |
| Lasso | `min ‖Xw − y‖²/(2n) + α‖w‖₁` → soft-thresholding `S(z,α)` |
| LassoCV / LassoLarsIC | CV vs `AIC = −2 log L̂ + 2d` vs `BIC = −2 log L̂ + log(n)·d` |
| ElasticNet | `L2/(2n) + αρ‖w‖₁ + α(1−ρ)/2‖w‖²` |
| LARS | Equiangular direction between active features, full path in one fit |
| OMP | Greedy: pick max-correlated feature, refit least squares on active set |
| BayesianRidge | `m = (α XᵀX + λI)⁻¹ α Xᵀy` (shared `λ`) |
| ARD | Same but per-feature `λ_i` → automatic feature selection |
| Logistic | `p = 1/(1+e^(−Xw−w0))`, cross-entropy + `r(w)/(S·C)` |
| GLM / Tweedie | `ŷ = h(Xw)`, minimize unit deviance from exponential family |
| SGD | `w ← w − η·∇L_i(w)` per sample |
| Perceptron | SGD with `loss='perceptron'`, update only on mistakes |
| Passive-Aggressive | No update if margin ≥ 1, else minimal correction step |
| Huber | Quadratic loss for `\|r\| < ε`, linear beyond |
| RANSAC | Random subsample, count inliers, refit on best inlier set |
| Theil-Sen | Median of all pairwise OLS slopes, ~29.3% breakdown point |
| Quantile | Minimize pinball loss `q·t⁺ + (1−q)·t⁻`, `t = y − ŷ` |
| Polynomial | `z = [x₁, x₂, x₁², x₁x₂, x₂²]`, still linear in `w` |

---

## Setup

Requires Python 3.12 (via Homebrew) on macOS.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt