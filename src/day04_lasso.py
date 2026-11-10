# Day 04: Lasso Regression
# Objective: min (1 / (2*n_samples)) * ||Xw - y||^2 + alpha * ||w||_1
# Math: The L1 penalty (sum of absolute values) has a special property:
#       it drives some coefficients EXACTLY to zero. This is called
#       "sparsity" and it means Lasso does automatic feature selection.
# Why it exists: Ridge shrinks coefficients toward zero but never hits zero.
#       Lasso can hit exactly zero, so it tells you which features matter.

from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
import numpy as np

# --- Part 1: Toy Example (from sklearn docs) ---
# Purpose: Muscle memory. Note the API is IDENTICAL to Ridge.
print("--- Toy Example (from sklearn docs) ---")
reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])

print(f"Coefficients (w): {reg.coef_}")
print(f"Intercept (w0): {reg.intercept_}")
print(f"Prediction for [1, 1]: {reg.predict([[1, 1]])}")

# PREDICT BEFORE RUNNING: Will one of the two coefficients be exactly 0.0?
# Your guess:

# --- Part 2: Sparsity on Real Data (Lasso vs Ridge) ---
# Purpose: See Lasso zero out features that Ridge only shrinks.
print("\n--- Real Data: Sparsity Comparison ---")
X, y = load_diabetes(return_X_y=True)

for alpha in [0.01, 0.1, 1.0, 10.0]:
    lasso = linear_model.Lasso(alpha=alpha, max_iter=10000).fit(X, y)
    ridge = Ridge(alpha=alpha).fit(X, y)
    lasso_zeros = np.sum(lasso.coef_ == 0)
    ridge_zeros = np.sum(ridge.coef_ == 0)
    print(f"alpha={alpha:>5}: Lasso zeros={lasso_zeros:>2}, Ridge zeros={ridge_zeros:>2}, "
          f"Lasso ||w||_1={np.linalg.norm(lasso.coef_, 1):.3f}")

# PREDICT BEFORE RUNNING: As alpha grows, will Lasso zeros increase or decrease?
# Your guess:

# --- Part 3: Manual Math Check (Soft-Thresholding) ---
# Purpose: Prove the math works. For a SINGLE feature with others fixed,
# Lasso has a closed-form solution called "soft-thresholding":
#     w_j = S(x_j^T (y - X_(-j) w_(-j)), alpha) / ||x_j||^2
# where S(z, a) = sign(z) * max(0, |z| - a).
print("\n--- Manual Soft-Thresholding Check ---")
# Use just ONE feature to make it clean
X_1 = X[:, [0]]  # first column only
y_1 = y.copy()
alpha_test = 10.0

# Closed-form single-feature Lasso: center the data, then soft-threshold
x_centered = X_1[:, 0] - X_1[:, 0].mean()
y_centered = y_1 - y_1.mean()
correlation = np.dot(x_centered, y_centered)
denominator = np.dot(x_centered, x_centered)

# Soft-threshold function
def soft_threshold(z, a):
    return np.sign(z) * max(0.0, abs(z) - a)

w_manual = soft_threshold(correlation, alpha_test) / denominator
intercept_manual = y_1.mean() - w_manual * X_1[:, 0].mean()

# Compare with sklearn
lasso_single = linear_model.Lasso(alpha=alpha_test, fit_intercept=True)
lasso_single.fit(X_1, y_1)

print(f"Manual coefficient:  {w_manual:.6f}")
print(f"Sklearn coefficient: {lasso_single.coef_[0]:.6f}")
print(f"Manual intercept:    {intercept_manual:.4f}")
print(f"Sklearn intercept:   {lasso_single.intercept_:.4f}")
# These should match (small differences from sklearn's regularization
# of the intercept are possible; within ~0.001 is fine).

# --- Part 4: Why the L1 Penalty Creates Zeros ---
# Purpose: Understand the geometry. With L1, the solution "corner" is often
# exactly on an axis (which means one coefficient is zero).
print("\n--- Geometry of L1 vs L2 ---")
print("Imagine a 2D plot of (w1, w2).")
print("  L2 penalty (Ridge): ||w||^2 = w1^2 + w2^2  -> a CIRCLE.")
print("  L1 penalty (Lasso): ||w||_1 = |w1| + |w2| -> a DIAMOND (rotated square).")
print("The solution sits where the penalty shape first touches the loss contour.")
print("A diamond has CORNERS on the axes. A circle does not.")
print("So the solution for Lasso often lands on a corner -> one w is exactly 0.")
print("That is the whole reason Lasso is sparse.")

# Demonstration: how often does Lasso hit exactly zero vs Ridge, on random data?
rng = np.random.RandomState(42)
X_rand = rng.randn(50, 5)   # 5 independent features, 50 samples
y_rand = rng.randn(50)      # pure noise (no real signal)

lasso_rand = linear_model.Lasso(alpha=0.5, max_iter=10000).fit(X_rand, y_rand)
ridge_rand = Ridge(alpha=0.5).fit(X_rand, y_rand)
print(f"\nLasso coefficients: {lasso_rand.coef_}")
print(f"Ridge coefficients: {ridge_rand.coef_}")
# Count exactly-zero entries:
print(f"Lasso exact zeros: {np.sum(lasso_rand.coef_ == 0)} / 5")
print(f"Ridge exact zeros: {np.sum(ridge_rand.coef_ == 0)} / 5")

# PREDICT BEFORE RUNNING: Which will have more exact zeros on pure noise?
# Your guess:

# --- Part 5: Lasso on Correlated Features (Lasso vs Ridge vs ElasticNet preview) ---
# Purpose: See Lasso's weakness. When features are correlated, Lasso picks ONE
# at random and zeros the others. Ridge keeps both (split evenly).
print("\n--- Correlated Features: Lasso vs Ridge ---")
rng = np.random.RandomState(0)
x1 = rng.randn(100)
x2 = x1 + rng.randn(100) * 0.01   # x2 is basically x1
X_corr = np.c_[x1, x2]
y_corr = 3 * x1 + 2 * x2 + rng.randn(100) * 0.1

lasso_corr = linear_model.Lasso(alpha=0.1, max_iter=10000).fit(X_corr, y_corr)
ridge_corr = Ridge(alpha=1.0).fit(X_corr, y_corr)
print(f"Lasso coefficients: {lasso_corr.coef_}")   # One may be 0, the other big
print(f"Ridge coefficients: {ridge_corr.coef_}")  # Both nonzero, similar size
# This is exactly why ElasticNet exists (Day 6).

# --- Part 6: Reflection ---
# Type your answers below.
# Q1: In one sentence, what does Lasso do that Ridge does not?
# Your answer:
#
# Q2: What is the soft-thresholding function S(z, alpha) doing geometrically?
# Your answer:
#
# Q3: You have 500 features and 100 samples. Would you use Lasso or Ridge?
#     Why?
# Your answer: