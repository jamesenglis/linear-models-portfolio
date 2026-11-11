# Day 08: Bayesian Ridge Regression + Automatic Relevance Determination (ARD)
#
# Idea: Instead of picking alpha by hand (Ridge) or via CV (RidgeCV), let the
# model INFER the regularization from the data using Bayes' rule.
#
# BayesianRidge:
#   - Prior on weights:      p(w | lambda) = N(w | 0, lambda^-1 * I)
#     -> one shared precision lambda for ALL features (isotropic)
#   - Prior on noise:        p(alpha_noise) = Gamma(...)  (uninformative)
#   - Uses empirical Bayes (evidence maximization) to learn lambda and alpha
#     from the data. Iteratively re-estimates them until convergence.
#
# ARD (Automatic Relevance Determination):
#   - Prior on weights:      p(w | lambda_vec) = N(w | 0, diag(lambda)^-1)
#     -> ONE precision PER FEATURE (anisotropic)
#   - Each lambda_i is learned. Irrelevant features get lambda_i -> infinity,
#     which drives w_i -> 0. So ARD is sparse, like Lasso.
#
# Math (posterior mean, given hyperparameters):
#   m = (alpha * X^T X + diag(lambda))^-1 * alpha * X^T y
#   where alpha = noise precision, lambda = weight precisions.
#   BayesianRidge: lambda is a scalar; ARD: lambda is a vector.
#
# Key difference: BayesianRidge = smooth shrinkage (like Ridge).
#                ARD             = per-feature shrinkage (like Lasso).

from sklearn import linear_model
from sklearn.datasets import load_diabetes, make_regression
import numpy as np

# --- Part 1: Toy Example (from sklearn docs) ---
# Purpose: Muscle memory. Both APIs are IDENTICAL to Ridge (same fit/predict).
print("--- Toy Example (from sklearn docs) ---")
X_toy = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
Y_toy = [0., 1., 2., 3.]

reg = linear_model.BayesianRidge()
reg.fit(X_toy, Y_toy)
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")
print(f"Prediction for [1, 0]: {reg.predict([[1, 0.]])}")

# PREDICT BEFORE RUNNING: Will the coefficients be EXACTLY 0.5, 0.5 (OLS-like)
# or slightly off because Bayesian shrinkage is applied?
# Your guess:

# --- Part 2: BayesianRidge on Real Data ---
# Purpose: See what learned hyperparameters look like.
print("\n--- BayesianRidge on Real Data ---")
X, y = load_diabetes(return_X_y=True)

br = linear_model.BayesianRidge()
br.fit(X, y)

print(f"Coefficients: {br.coef_}")
print(f"Intercept: {br.intercept_}")
print(f"Learned alpha (noise precision):  {br.alpha_:.4f}")
print(f"Learned lambda (weight precision): {br.lambda_:.4f}")
print(f"Number of iterations to converge:  {br.n_iter_}")
print(f"R^2 on training: {br.score(X, y):.4f}")

# Note: alpha_ = 1 / noise_variance, so HIGH alpha_ = LOW noise.
# lambda_ = 1 / weight_variance, so HIGH lambda_ = STRONG shrinkage.
print(f"\nImplied noise std: {1 / np.sqrt(br.alpha_):.4f}")
print(f"Implied weight std: {1 / np.sqrt(br.lambda_):.4f}")

# PREDICT BEFORE RUNNING: Will lambda_ be high (strong shrink) or low
# (weak shrink)? Hint: diabetes has 10 features, 442 samples. Well-conditioned.
# Your guess:

# --- Part 3: ARD on Real Data ---
# Purpose: See per-feature precisions. Irrelevant features get huge precision.
print("\n--- ARD on Real Data ---")
ard = linear_model.ARDRegression()
ard.fit(X, y)

print(f"Coefficients: {ard.coef_}")
print(f"Number of nonzero coefs: {np.sum(ard.coef_ != 0)} / {X.shape[1]}")
print(f"Learned alpha (noise precision):  {ard.alpha_:.4f}")
print(f"Lambda vector (per-feature):")
print(f"  {ard.lambda_}")
print(f"R^2 on training: {ard.score(X, y):.4f}")

# Features with LARGE lambda_ have STRONG shrinkage -> close to zero.
# Features with SMALL lambda_ have weak shrinkage -> important features.
print(f"\nMost important features (smallest lambda_): "
      f"{np.argsort(ard.lambda_)[:3]}")
print(f"Least important features (largest lambda_): "
      f"{np.argsort(ard.lambda_)[-3:]}")

# PREDICT BEFORE RUNNING: Will ARD set any coefficients EXACTLY to zero
# (like Lasso) or just very small?
# Your guess:

# --- Part 4: Manual Math Check (Posterior Mean) ---
# Purpose: Prove the formula. For given hyperparameters, the posterior mean is
#     m = (alpha * X^T X + diag(lambda))^-1 * alpha * X^T y
# where alpha = noise precision, lambda = weight precision (scalar for
# BayesianRidge, vector for ARD).
print("\n--- Manual Posterior Mean Check (using learned hyperparams) ---")

# Add intercept column
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# BayesianRidge: single scalar lambda, and we do NOT penalize the intercept
# (so the top-left entry of the penalty matrix is 0, not lambda).
lam_br = br.lambda_
alpha_br = br.alpha_
penalty_br = np.eye(X_b.shape[1]) * lam_br
penalty_br[0, 0] = 0   # intercept unpenalized
m_br = np.linalg.inv(alpha_br * X_b.T.dot(X_b) + penalty_br).dot(
    alpha_br * X_b.T.dot(y)
)
print(f"Manual BayesianRidge intercept:  {m_br[0]:.6f}")
print(f"Sklearn BayesianRidge intercept: {br.intercept_:.6f}")
print(f"Manual BayesianRidge coefs:  {m_br[1:4]}")
print(f"Sklearn BayesianRidge coefs: {br.coef_[:4]}")

# ARD: per-feature lambda vector, also intercept unpenalized.
lam_ard = ard.lambda_
alpha_ard = ard.alpha_
penalty_ard = np.diag(np.concatenate([[0.0], lam_ard]))  # intercept not penalized
m_ard = np.linalg.inv(alpha_ard * X_b.T.dot(X_b) + penalty_ard).dot(
    alpha_ard * X_b.T.dot(y)
)
print(f"\nManual ARD intercept:  {m_ard[0]:.6f}")
print(f"Sklearn ARD intercept: {ard.intercept_:.6f}")
print(f"Manual ARD coefs:  {m_ard[1:4]}")
print(f"Sklearn ARD coefs: {ard.coef_[:4]}")
# Should match closely (small differences from iterative re-estimation).
# The math check validates the POSTERIOR MEAN given the LEARNED hyperparams.

# --- Part 5: ARD as Feature Selector on Irrelevant Features ---
# Purpose: Add pure-noise features. ARD should shrink them aggressively.
print("\n--- ARD: Noise Features Get Shrunk ---")
rng = np.random.RandomState(42)
n_informative = 5
n_noise = 5
X_info, y_info = make_regression(
    n_samples=200, n_features=n_informative, n_informative=n_informative,
    noise=0.5, random_state=42
)
X_noise = rng.randn(200, n_noise)  # pure noise, no signal
X_mixed = np.c_[X_info, X_noise]

ard_mixed = linear_model.ARDRegression().fit(X_mixed, y_info)
br_mixed = linear_model.BayesianRidge().fit(X_mixed, y_info)

print(f"Coef shape: {ard_mixed.coef_.shape}")
print(f"Informative features (0..4):")
print(f"  ARD: {ard_mixed.coef_[:n_informative]}")
print(f"  BR:  {br_mixed.coef_[:n_informative]}")
print(f"Noise features (5..9):")
print(f"  ARD: {ard_mixed.coef_[n_informative:]}")
print(f"  BR:  {br_mixed.coef_[n_informative:]}")

print(f"\nARD noise-feature magnitude: "
      f"{np.linalg.norm(ard_mixed.coef_[n_informative:]):.4f}")
print(f"BR  noise-feature magnitude: "
      f"{np.linalg.norm(br_mixed.coef_[n_informative:]):.4f}")
print(f"ARD informative magnitude:   "
      f"{np.linalg.norm(ard_mixed.coef_[:n_informative]):.4f}")

# PREDICT BEFORE RUNNING: Which will shrink the noise features more --
# ARD (per-feature lambda) or BayesianRidge (shared lambda)?
# Your guess:

# --- Part 6: BayesianRidge vs Ridge -- when to use which ---
# Purpose: Practical decision-making.
print("\n--- BayesianRidge vs Ridge: Practical Choice ---")
print("Ridge:")
print("  + Fast, deterministic, one hyperparameter (alpha).")
print("  - You must pick alpha (by hand or by CV).")
print("BayesianRidge:")
print("  + Learns alpha AND noise variance automatically from data.")
print("  + Gives you uncertainty estimates (predict with return_std=True).")
print("  - Iterative; slightly slower. Assumes Gaussian noise.")
print("\nARD:")
print("  + Per-feature shrinkage -> automatic feature selection.")
print("  + Better than Lasso when features are correlated (no arbitrary pick).")
print("  - Slower. Can misbehave when n_features >= n_samples.")

# Show BayesianRidge's uncertainty output (unique advantage)
y_pred, y_std = br.predict(X[:5], return_std=True)
print(f"\nBayesianRidge predictions with uncertainty (first 5 rows):")
for i in range(5):
    print(f"  true={y[i]:>7.2f}, pred={y_pred[i]:>7.2f}, std={y_std[i]:>6.3f}")

# --- Part 7: Reflection ---
# Type your answers below.
# Q1: In one sentence, what is the difference between BayesianRidge and ARD?
# Your answer:
#
# Q2: Why does ARD's per-feature lambda achieve feature selection
#     (noise feature coefs -> ~0) while BayesianRidge's shared lambda does not?
# Your answer:
#
# Q3: What is ONE thing BayesianRidge gives you that Ridge does not?
# Your answer: