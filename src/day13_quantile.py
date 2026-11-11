# Day 13 (FINAL of section 1.1): Quantile Regression
# Covers scikit-learn User Guide §1.1.15
#
# Problem: OLS predicts the MEAN of y given X. If the distribution of y
# around the mean is skewed or has non-constant variance (heteroscedastic),
# the mean is not enough -- you may want a PREDICTION INTERVAL.
#
# Quantile regression predicts any quantile q in (0, 1):
#     q = 0.5  -> median (robust to outliers, like robust regression)
#     q = 0.1  -> 10th percentile (lower bound)
#     q = 0.9  -> 90th percentile (upper bound)
# Predict (q=0.1, q=0.9) together to get a 80% prediction interval.
#
# Math (PINBALL LOSS):
#     PB_q(t) = q * max(t, 0) + (1 - q) * max(-t, 0)
#   where t = y_i - y_hat_i is the residual.
#
#   Interpretation:
#     If t > 0 (under-prediction), penalty is q * t.
#     If t < 0 (over-prediction), penalty is (1 - q) * |t|.
#   So q=0.9 heavily penalizes UNDER-prediction (wanting the estimate high).
#   q=0.1 heavily penalizes OVER-prediction (wanting the estimate low).
#
# Objective:
#     min_w (1/n) * sum_i PB_q(y_i - X_i w) + alpha * ||w||_1
#   (alpha is the L1 regularization term, similar to Lasso.)
#
# Implementation uses scipy.optimize.linprog (linear programming).
# NOTE: Uses L1 penalty (not L2). This is quantile Lasso.

from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss
import numpy as np

# =========================================================================
# PART A: Setup -- heteroscedastic data
# =========================================================================
print("=" * 70)
print("PART A: Data with increasing variance")
print("=" * 70)
rng = np.random.RandomState(42)
# Generate data where noise SCALES with x1 -- heteroscedastic
n = 300
X = rng.randn(n, 2) * 5
# y depends linearly on X, but noise grows with |x1|
true_coef = np.array([2.0, 1.0])
noise_scale = 1.0 + np.abs(X[:, 0])  # bigger |x1| -> bigger noise
y = X.dot(true_coef) + rng.randn(n) * noise_scale

print(f"X shape: {X.shape}")
print(f"True coef: {true_coef}")
print(f"Noise std range: {noise_scale.min():.2f} to {noise_scale.max():.2f}")
print("(Note: noise variance GROWS with |x1|. OLS only sees the mean.)")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================================
# PART B: Mean vs Median vs Low vs High quantiles
# =========================================================================
print("\n" + "=" * 70)
print("PART B: Different quantiles learn different things")
print("=" * 70)
# Fit 4 models with different quantiles
models = {}
for q in [0.1, 0.5, 0.9]:
    qr = linear_model.QuantileRegressor(quantile=q, alpha=0.01, solver="highs")
    qr.fit(X_tr, y_tr)
    models[q] = qr
    print(f"\nq={q} (QuantileRegressor):")
    print(f"  coef={np.round(qr.coef_, 3)}, intercept={qr.intercept_:.3f}")
    print(f"  test pinball loss: "
          f"{mean_pinball_loss(y_te, qr.predict(X_te), alpha=q):.4f}")

# OLS for comparison
ols = linear_model.LinearRegression().fit(X_tr, y_tr)
print(f"\nOLS:  coef={np.round(ols.coef_, 3)}, intercept={ols.intercept_:.3f}")

print("\n--- What to look for ---")
print("q=0.1  coef < q=0.5 coef < q=0.9 coef  (usually) for the first feature.")
print(f"  q=0.1 coef[0] = {models[0.1].coef_[0]:.3f}")
print(f"  q=0.5 coef[0] = {models[0.5].coef_[0]:.3f}")
print(f"  q=0.9 coef[0] = {models[0.9].coef_[0]:.3f}")

# PREDICT BEFORE RUNNING: Will q=0.9 coef be BIGGER or SMALLER than q=0.1?
# Hint: q=0.9 tries to cover the high tail of y. If y grows with x1 and the
# noise grows with |x1|, the 90th percentile line should rise faster with x1.
# Your guess:

# =========================================================================
# PART C: Build a prediction interval
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Prediction interval from two quantiles")
print("=" * 70)
qr_low = linear_model.QuantileRegressor(quantile=0.05, alpha=0.01, solver="highs")
qr_high = linear_model.QuantileRegressor(quantile=0.95, alpha=0.01, solver="highs")
qr_low.fit(X_tr, y_tr)
qr_high.fit(X_tr, y_tr)

y_low = qr_low.predict(X_te)
y_high = qr_high.predict(X_te)
interval_width = y_high - y_low
print(f"Median width of 90% prediction interval: {np.median(interval_width):.2f}")
print(f"Min width: {interval_width.min():.2f}, "
      f"Max width: {interval_width.max():.2f}")
print("(Width should correlate with |x1| because of the heteroscedastic noise.)")

# Check empirical coverage: how many test y fall in [y_low, y_high]?
coverage = np.mean((y_te >= y_low) & (y_te <= y_high))
print(f"Empirical coverage: {coverage:.2%}  (target: ~90%)")

# PREDICT BEFORE RUNNING: Will empirical coverage be EXACTLY 90%?
# Hint: it's an approximation; finite sample + model misspecification.
# Your guess:

# =========================================================================
# PART D: Pinball loss by hand (prove the math)
# =========================================================================
print("\n" + "=" * 70)
print("PART D: Manual pinball loss calculation")
print("=" * 70)
# Manual PB_q for a few (y, y_hat) pairs
def pinball_loss(y_true, y_pred, q):
    t = y_true - y_pred
    return q * np.maximum(t, 0) + (1 - q) * np.maximum(-t, 0)

y_true_demo = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
y_pred_demo = np.array([7.0, 9.0, 10.0, 11.0, 13.0])
q = 0.9

print(f"q = {q}")
print(f"{'y_pred':>8} {'resid':>8} {'pinball':>10} {'note':<25}")
for yp in y_pred_demo:
    t = 10.0 - yp
    loss = pinball_loss(10.0, yp, q)
    note = "under (big penalty)" if t > 0 else ("over (small penalty)" if t < 0 else "perfect")
    print(f"{yp:>8.1f} {t:>8.1f} {loss:>10.3f} {note:<25}")

print("\nNotice asymmetry: q=0.9 penalizes UNDER-prediction 9x more than")
print("OVER-prediction. So the fitted line 'aims high' to balance this.")
print("That's why it lands on the 90th percentile, not the mean.")

# Compare with mean_pinball_loss
manual_total = np.mean([pinball_loss(10.0, yp, q) for yp in y_pred_demo])
sklearn_total = mean_pinball_loss(
    np.array([10.0] * 5), y_pred_demo, alpha=q
)
print(f"\nManual mean pinball: {manual_total:.6f}")
print(f"sklearn mean pinball: {sklearn_total:.6f}")

# =========================================================================
# PART E: Quantile vs OLS on a skewed distribution
# =========================================================================
print("\n" + "=" * 70)
print("PART E: When quantile regression beats OLS")
print("=" * 70)
# Generate y = exp(Xw) + noise -- right-skewed (lognormal-ish)
X_sk, _, _ = make_regression(
    n_samples=500, n_features=1, noise=0.0, random_state=42
)
y_skew = np.exp(0.5 * X_sk[:, 0]) + rng.randn(500) * 0.3

ols_skew = linear_model.LinearRegression().fit(X_sk, y_skew)
qr_median = linear_model.QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs")
qr_median.fit(X_sk, y_skew)

print(f"OLS mean prediction at x=0:     {ols_skew.predict([[0]])[0]:.4f}")
print(f"Median prediction at x=0:       {qr_median.predict([[0]])[0]:.4f}")
print(f"True value at x=0 (exp(0)+0):   {1.0:.4f}")
print("(OLS tries to fit the mean; the median is closer to 1.0 here,")
print(" because the lognormal tail pulls the mean above the median.)")

# PREDICT BEFORE RUNNING: Which model will better predict y_skew near x=0?
# Your guess:

# =========================================================================
# PART F: alpha (regularization) effect
# =========================================================================
print("\n" + "=" * 70)
print("PART F: alpha (L1 regularization) effect")
print("=" * 70)
print(f"{'alpha':>10} {'coef[0]':>10} {'coef[1]':>10} {'train_pinball':>15}")
for alpha in [0.0, 0.001, 0.01, 0.1, 1.0]:
    qr = linear_model.QuantileRegressor(
        quantile=0.5, alpha=alpha, solver="highs"
    )
    qr.fit(X_tr, y_tr)
    train_loss = mean_pinball_loss(y_tr, qr.predict(X_tr), alpha=0.5)
    print(f"{alpha:>10} {qr.coef_[0]:>10.3f} {qr.coef_[1]:>10.3f} "
          f"{train_loss:>15.4f}")

# PREDICT BEFORE RUNNING: As alpha goes UP, will coef[1] shrink toward 0?
# Your guess:

# =========================================================================
# PART G: Reflection -- FINAL reflection of section 1.1
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Reflection -- completing section 1.1")
print("=" * 70)

# Q1: In one sentence, what is quantile regression?
# Your answer:
#
# Q2: What is the "pinball loss" and why is it asymmetric?
# Your answer:
#
# Q3: You want to predict a 80% prediction interval. Which two quantiles
#     do you fit, and what do you use their outputs for?
# Your answer:
#
# Q4: How is quantile regression with q=0.5 different from OLS?
#     (Hint: OLS minimizes squared error -> predicts mean. Quantile q=0.5
#     minimizes absolute error -> predicts median. These coincide ONLY for
#     symmetric distributions.)
# Your answer:

# =========================================================================
# PART H: Portfolio completeness check
# =========================================================================
print("\n" + "=" * 70)
print("PART H: Section 1.1 -- COMPLETE")
print("=" * 70)
print("""
All 16 subsections of scikit-learn User Guide section 1.1 are now covered:

  1.1.1  Ordinary Least Squares         -> day01_ols.py
  1.1.2  Ridge regression               -> day02_ridge.py
  1.1.3  Lasso (+ CV, AIC/BIC)          -> day04_lasso.py, day05_lassocv_ic.py
  1.1.4  Multi-task Lasso               -> day06_elasticnet.py
  1.1.5  Elastic-Net                    -> day06_elasticnet.py
  1.1.6  Multi-task Elastic-Net         -> day06_elasticnet.py
  1.1.7  Least Angle Regression         -> day07_lars_omp.py
  1.1.8  LARS Lasso                     -> day07_lars_omp.py
  1.1.9  Orthogonal Matching Pursuit    -> day07_lars_omp.py
  1.1.10 Bayesian Regression (ARD)      -> day08_bayesian_ard.py
  1.1.11 Logistic regression            -> day09_logistic.py
  1.1.12 Generalized Linear Models      -> day10_glm_pipeline.py
  1.1.13 SGD (+ Perceptron, PA)         -> day11_sgd.py
  1.1.14 Robustness regression          -> day12_robust.py
  1.1.15 Quantile regression            -> day13_quantile.py
  1.1.16 Polynomial regression          -> day10_glm_pipeline.py

Plus Day 3 (RidgeCV) which goes deeper on cross-validated alpha selection.

Total: 13 files (day01 through day13).
""")