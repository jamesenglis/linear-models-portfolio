# Day 12: Robust Regression -- RANSAC, Huber, Theil-Sen
# Covers scikit-learn User Guide §1.1.14
#
# Problem: OLS minimizes SUM of squared residuals. A single large outlier
# has a HUGE squared residual, so OLS bends the whole fit toward it.
# Robust regression uses losses or algorithms that tolerate outliers.
#
# Breakdown point: the fraction of data that can be arbitrary outliers
# before the fit is ruined.
#   OLS:       0%   (one outlier can ruin it)
#   Huber:     ~0% but heavily downweighted (bounded influence, not BD point)
#   Theil-Sen: ~29.3% (univariate)
#   RANSAC:    user-controlled via residual_threshold and max_trials
#
# --- RANSAC (RANdom SAmple Consensus) ---
#   Non-deterministic. In each trial:
#     1. Sample min_samples points at random.
#     2. Fit the model on those points.
#     3. Count how many of the FULL dataset are "inliers" (residual <
#        residual_threshold).
#     4. Keep the model with the most inliers (ties broken by score).
#   Then refit on the final inlier set.
#
# --- Theil-Sen ---
#   Non-parametric. Fits OLS on EVERY pair of points and takes the MEDIAN
#   of the slopes/intercepts. Median is robust to up to 29.3% outliers.
#   Cost: O(n^2) pairs -> too slow for big n. sklearn subsamples.
#
# --- Huber ---
#   Uses a loss that is quadratic for small residuals and LINEAR for large
#   ones. So outliers are downweighted but not fully ignored.
#   Loss: H_eps(z) = z^2          if |z| < eps
#                   2*eps*|z| - eps^2  otherwise
#   eps (epsilon) = threshold. sklearn recommends 1.35 for 95% efficiency.

from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

# =========================================================================
# PART A: Setup -- generate data with outliers
# =========================================================================
print("=" * 70)
print("PART A: Data with outliers")
print("=" * 70)

rng = np.random.RandomState(42)
# Clean linear data: y = 3*x1 + 2*x2 + noise
X, y_true, coef_true = make_regression(
    n_samples=200, n_features=2, n_informative=2, noise=5.0,
    coef=True, random_state=42
)
# Add 15% outliers -- large random values in y
n_outliers = 30
outlier_idx = rng.choice(len(y_true), size=n_outliers, replace=False)
y = y_true.copy()
y[outlier_idx] += rng.randn(n_outliers) * 100  # huge offsets
print(f"Clean R^2 target: {coef_true}")
print(f"Number of outliers injected: {n_outliers} / {len(y)}")

# Split so we can compare train/test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================================
# PART B: Compare OLS vs Robust methods
# =========================================================================
print("\n" + "=" * 70)
print("PART B: OLS vs Huber vs Theil-Sen vs RANSAC")
print("=" * 70)

# --- B1: OLS (baseline) ---
ols = linear_model.LinearRegression().fit(X_tr, y_tr)
print(f"\nOLS:       coef={np.round(ols.coef_, 3)}, "
      f"intercept={ols.intercept_:.3f}")
print(f"           train R^2={ols.score(X_tr, y_tr):.4f}, "
      f"test R^2={ols.score(X_te, y_te):.4f}")

# --- B2: Huber ---
huber = linear_model.HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000)
huber.fit(X_tr, y_tr)
print(f"\nHuber:     coef={np.round(huber.coef_, 3)}, "
      f"intercept={huber.intercept_:.3f}")
print(f"           train R^2={huber.score(X_tr, y_tr):.4f}, "
      f"test R^2={huber.score(X_te, y_te):.4f}")
print(f"           scale_={huber.scale_:.3f}, n_iter={huber.n_iter_}")
# scale_ is the estimated scale of the residuals, used internally.

# --- B3: Theil-Sen ---
ts = linear_model.TheilSenRegressor(
    random_state=42, max_subpopulation=1e4, n_subsamples=50
)
ts.fit(X_tr, y_tr)
print(f"\nTheil-Sen: coef={np.round(ts.coef_, 3)}, "
      f"intercept={ts.intercept_:.3f}")
print(f"           train R^2={ts.score(X_tr, y_tr):.4f}, "
      f"test R^2={ts.score(X_te, y_te):.4f}")

# --- B4: RANSAC ---
ransac = linear_model.RANSACRegressor(
    estimator=linear_model.LinearRegression(),
    residual_threshold=20.0,   # adjust based on y scale
    max_trials=200,
    min_samples=0.5,
    random_state=42,
)
ransac.fit(X_tr, y_tr)
inlier_mask = ransac.inlier_mask_
print(f"\nRANSAC:    coef={np.round(ransac.estimator_.coef_, 3)}, "
      f"intercept={ransac.estimator_.intercept_:.3f}")
print(f"           train R^2={ransac.score(X_tr, y_tr):.4f}, "
      f"test R^2={ransac.score(X_te, y_te):.4f}")
print(f"           inliers={np.sum(inlier_mask)}/{len(inlier_mask)}")

print(f"\nTrue coefs: {np.round(coef_true, 3)}")
print("(Robust methods should be CLOSER to true coefs than OLS.)")

# PREDICT BEFORE RUNNING: Which method will give coefficients CLOSEST to the
# true coefs? Hint: RANSAC finds the largest inlier set; Theil-Sen uses
# medians; Huber downweights.
# Your guess:

# =========================================================================
# PART C: Manual Huber loss (prove the math)
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Manual Huber loss calculation")
print("=" * 70)
# Show that Huber loss equals squared error for small residuals and
# linear loss for large ones.
eps = 1.35
residuals = np.array([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0])

def huber_loss(z, eps):
    abs_z = np.abs(z)
    return np.where(abs_z <= eps, z**2, 2 * eps * abs_z - eps**2)

losses = huber_loss(residuals, eps)
print(f"{'residual':>10} {'huber_loss':>12} {'squared':>10} {'note':<20}")
for r, l in zip(residuals, losses):
    note = "QUADRATIC" if abs(r) <= eps else "LINEAR"
    print(f"{r:>10.2f} {l:>12.4f} {r**2:>10.4f} {note:<20}")

# PREDICT BEFORE RUNNING: For residual=10, is Huber loss CLOSER to the
# squared loss (100) or much smaller? (Hint: eps=1.35.)
# Your guess:

# =========================================================================
# PART D: RANSAC manual walkthrough (one trial)
# =========================================================================
print("\n" + "=" * 70)
print("PART D: RANSAC -- one manual trial")
print("=" * 70)
# Do ONE RANSAC iteration by hand: pick random inliers, fit, count inliers.
rng2 = np.random.RandomState(0)
sample_idx = rng2.choice(len(X_tr), size=10, replace=False)  # random points
X_rand = X_tr[sample_idx]
y_rand = y_tr[sample_idx]
model_rand = linear_model.LinearRegression().fit(X_rand, y_rand)
# Count inliers on FULL training set
residuals_full = np.abs(model_rand.predict(X_tr) - y_tr)
threshold = 20.0
n_inliers = np.sum(residuals_full < threshold)
print(f"Random 10-point fit: coef={np.round(model_rand.coef_, 3)}")
print(f"Inliers under threshold={threshold}: {n_inliers}/{len(y_tr)}")

# Do 100 random trials and keep the best
best_model, best_inliers = None, -1
for trial in range(100):
    idx = rng2.choice(len(X_tr), size=10, replace=False)
    m = linear_model.LinearRegression().fit(X_tr[idx], y_tr[idx])
    res = np.abs(m.predict(X_tr) - y_tr)
    n = np.sum(res < threshold)
    if n > best_inliers:
        best_inliers = n
        best_model = m
print(f"\nBest of 100 trials: {best_inliers} inliers, "
      f"coef={np.round(best_model.coef_, 3)}")

# Refit on ALL inliers
inlier_idx = np.abs(best_model.predict(X_tr) - y_tr) < threshold
refit = linear_model.LinearRegression().fit(X_tr[inlier_idx], y_tr[inlier_idx])
print(f"Refit on inliers: coef={np.round(refit.coef_, 3)}")
print(f"Compare with sklearn RANSAC: coef={np.round(ransac.estimator_.coef_, 3)}")
# These should be close (small variations from random subsampling).

# =========================================================================
# PART E: Theil-Sen as median of pairwise slopes (1D demo)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: Theil-Sen as median of pairwise slopes (1D)")
print("=" * 70)
# Take just ONE feature to make it clean
x1d = X[:, 0]
y1d = y.copy()
n = len(x1d)
# Compute all pairwise slopes
slopes = []
for i in range(n):
    for j in range(i + 1, n):
        if x1d[j] != x1d[i]:
            slopes.append((y1d[j] - y1d[i]) / (x1d[j] - x1d[i]))
slopes = np.array(slopes)
median_slope = np.median(slopes)
median_intercept = np.median(y1d - median_slope * x1d)
print(f"All pairwise slopes: n_pairs={len(slopes)}")
print(f"Median slope:      {median_slope:.4f}")
print(f"Median intercept:  {median_intercept:.4f}")

# Compare with sklearn's TheilSen (1D)
ts1d = linear_model.TheilSenRegressor(random_state=42)
ts1d.fit(x1d[:, np.newaxis], y1d)
print(f"Sklearn Theil-Sen: coef={ts1d.coef_[0]:.4f}, "
      f"intercept={ts1d.intercept_:.4f}")
# Should match closely. sklearn subsamples for speed by default; for small n
# and max_subpopulation >= C(n,2), it matches the exact median.

# PREDICT BEFORE RUNNING: Will the median slope be closer to the TRUE slope
# (which equals coef_true[0]) or closer to the OLS slope? (OLS is ruined
# by outliers; median ignores them.)
# Your guess:

# =========================================================================
# PART F: When to use which
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Decision guide")
print("=" * 70)
print("""
OLS (LinearRegression):
  - No outliers, Gaussian noise.
  - Fast. Closed form.

Huber (HuberRegressor):
  - Mild outliers.
  - Scaling-invariant (once epsilon is set, rescaling X and y gives the
    same robustness).
  - Good when n is small.

Theil-Sen:
  - Strong outliers; want a high breakdown point (~29.3%).
  - Small-to-medium n (< 10^4 pairs).
  - Non-parametric (no Gaussian assumption).

RANSAC:
  - Large outliers with a clear inlier majority.
  - Non-deterministic (different results each run unless random_state set).
  - You need to set residual_threshold carefully -- it depends on y's scale.

IMPORTANT: Robust regression is hard in high dimensions. These work best
when n_features is small.
""")

# =========================================================================
# PART G: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Reflection")
print("=" * 70)

# Q1: What is a "breakdown point"? Which method has the highest one here?
# Your answer:
#
# Q2: Why is Theil-Sen O(n^2)? What does sklearn do about it?
# Your answer:
#
# Q3: Why is Huber called "robust" but not "high-breakdown"?
# Your answer:
#
# Q4: What is the key hyperparameter of RANSAC that you must choose by hand?
# Your answer: