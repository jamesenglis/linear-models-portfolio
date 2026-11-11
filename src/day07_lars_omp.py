# Day 07: LARS (Least Angle Regression) + OMP (Orthogonal Matching Pursuit)
#
# Both are FORWARD SELECTION algorithms: they build a model by adding one
# feature at a time. Useful when n_features >> n_samples.
#
# LARS:
#   - At each step, finds the feature most correlated with the residual.
#   - Then moves the coefficient in a direction EQUIANGULAR between all
#     currently-active features (so all their correlations with the
#     residual drop together).
#   - Produces the FULL piecewise-linear solution path "for free".
#     That path is what LassoLars uses to get an exact Lasso solution.
#
# OMP (Orthogonal Matching Pursuit):
#   - Greedy like LARS, but after adding a feature, it REFITS all active
#     coefficients via ordinary least squares (orthogonal projection).
#   - You choose either n_nonzero_coefs (fixed budget) or tol (target error).
#
# Math (LARS equiangular direction):
#   Let c = X^T (y - Xw) be the correlation vector. Active set A =
#   features with |c_j| equal to the current maximum. Direction u makes
#   X u equiangular with the active features' columns.
#
# Math (OMP):
#   argmin_w ||y - Xw||^2  subject to  ||w||_0 <= k
#   Each step: pick feature j most correlated with current residual,
#   add j to active set, solve least squares on the active set only.

from sklearn import linear_model
from sklearn.datasets import load_diabetes, make_regression
import numpy as np

# --- Part 1: Toy Example (from sklearn docs) ---
# Purpose: Muscle memory. Note LARS produces a PATH, not a single coef.
print("--- Toy Example (from sklearn docs) ---")
reg = linear_model.Lars(n_nonzero_coefs=1)
reg.fit([[-1, 1], [0, 0], [1, 1]], [1, 0, -1])
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

# PREDICT BEFORE RUNNING: With n_nonzero_coefs=1, how many coefficients
# will be nonzero?
# Your guess:

# --- Part 2: LARS Path vs Lasso Path ---
# Purpose: See that LARS gives you the whole coefficient path in one call.
print("\n--- LARS Path on Real Data ---")
X, y = load_diabetes(return_X_y=True)

# Lars with the full path (default: no constraint on n_nonzero_coefs)
lars_full = linear_model.Lars(n_nonzero_coefs=np.inf)
lars_full.fit(X, y)
print(f"Full path coef shape: {lars_full.coef_path_.shape}")
# Shape is (n_features, n_alphas_in_path). Each column is a solution.

# Lars limited to k nonzero coefficients
for k in [1, 3, 5, 10]:
    lars = linear_model.Lars(n_nonzero_coefs=k)
    lars.fit(X, y)
    nz = np.sum(lars.coef_ != 0)
    print(f"n_nonzero_coefs={k:>2}: actual nonzero = {nz:>2}, "
          f"R^2 = {lars.score(X, y):.4f}")

# PREDICT BEFORE RUNNING: Does LARS stop at EXACTLY k nonzero coefficients,
# or could it have FEWER than k?
# Your guess:

# --- Part 3: OMP (budget-based and tolerance-based) ---
# Purpose: See the two ways to stop OMP.
print("\n--- OMP: budget vs tolerance ---")
# Budget: stop when we have exactly k nonzero coefficients
for k in [1, 3, 5, 10]:
    omp = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=k)
    omp.fit(X, y)
    nz = np.sum(omp.coef_ != 0)
    print(f"n_nonzero_coefs={k:>2}: actual nonzero = {nz:>2}, "
          f"R^2 = {omp.score(X, y):.4f}")

# Tolerance: stop when residual norm ||y - Xw||^2 is below tol
for tol in [5000, 3000, 1000, 500]:
    omp = linear_model.OrthogonalMatchingPursuit(tol=tol)
    omp.fit(X, y)
    nz = np.sum(omp.coef_ != 0)
    rss = np.sum((y - omp.predict(X))**2)
    print(f"tol={tol:>6}: nonzero = {nz:>2}, actual RSS = {rss:.1f}")

# PREDICT BEFORE RUNNING: As tol gets SMALLER, will OMP use MORE or FEWER
# features? (Hint: smaller tolerance = need more features to fit better.)
# Your guess:

# --- Part 4: Manual OMP Step-by-Step (prove the math) ---
# Purpose: Do ONE iteration of OMP by hand and match sklearn.
# Step: pick feature with max |correlation| with residual, add to active set,
# then solve least squares on the active set only.
print("\n--- Manual OMP: one step at a time ---")

# Start with zero coefficients
w = np.zeros(X.shape[1])
residual = y - X.dot(w)  # = y when w = 0

# Step 1: find feature with max |correlation|
correlations = X.T.dot(residual)
j = np.argmax(np.abs(correlations))
print(f"Step 1: selected feature {j} (correlation={correlations[j]:.3f})")

# Step 2: add j to active set, solve least squares on that set only
active = [j]
X_active = X[:, active]
w_active, *_ = np.linalg.lstsq(X_active, y, rcond=None)
w_manual = np.zeros(X.shape[1])
w_manual[active] = w_active
print(f"Manual coef (1 step): {w_manual}")

# Compare with sklearn OMP with n_nonzero_coefs=1
omp_1 = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=1)
omp_1.fit(X, y)
print(f"Sklearn OMP coef (1 step): {omp_1.coef_}")
# Note: sklearn may have a different sign/sign-interpretation on the first
# coefficient. If the SELECTED feature index matches, the algorithm matches.

# Now do 2 more iterations manually
for step in range(2, 5):
    residual = y - X.dot(w_manual)
    correlations = X.T.dot(residual)
    # zero out already-active features
    correlations[active] = 0
    j_new = np.argmax(np.abs(correlations))
    active.append(j_new)
    X_active = X[:, active]
    w_active, *_ = np.linalg.lstsq(X_active, y, rcond=None)
    w_manual = np.zeros(X.shape[1])
    w_manual[active] = w_active
    print(f"After {step} steps: active features = {active}")

omp_3 = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=3)
omp_3.fit(X, y)
print(f"Sklearn OMP (3): {omp_3.coef_}")
print(f"Manual OMP (3):  {w_manual}")

# --- Part 5: LARS vs OMP Head-to-Head ---
# Purpose: Same data, same k, see how coefficients differ.
print("\n--- LARS vs OMP (n_nonzero_coefs=3) ---")
lars_3 = linear_model.Lars(n_nonzero_coefs=3).fit(X, y)
omp_3 = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=3).fit(X, y)

print(f"LARS coef: {lars_3.coef_}")
print(f"OMP  coef: {omp_3.coef_}")
print(f"LARS picked features: {np.where(lars_3.coef_ != 0)[0]}")
print(f"OMP  picked features: {np.where(omp_3.coef_ != 0)[0]}")
print(f"LARS R^2: {lars_3.score(X, y):.4f}")
print(f"OMP  R^2: {omp_3.score(X, y):.4f}")

# PREDICT BEFORE RUNNING: Will LARS and OMP pick the SAME features?
# Your guess:

# --- Part 6: LARS for Lasso (the connection) ---
# Purpose: LARS can produce a Lasso path. LassoLars is the exact solver.
print("\n--- LassoLars: Lasso solved via LARS ---")
from sklearn.linear_model import LassoLars, Lasso
ll = LassoLars(alpha=0.1).fit(X, y)
l_std = Lasso(alpha=0.1, max_iter=10000).fit(X, y)
print(f"LassoLars coef: {ll.coef_}")
print(f"Lasso    coef: {l_std.coef_}")
# They should match closely (both solve the exact Lasso objective).

# PREDICT BEFORE RUNNING: Will LassoLars and Lasso give EXACTLY the same
# coefficients, or just close?
# Your guess:

# --- Part 7: Reflection ---
# Type your answers below.
# Q1: In one sentence, what is the difference between LARS and OMP?
# Your answer:
#
# Q2: Why is LARS useful for n_features >> n_samples?
# Your answer:
#
# Q3: If you want exactly 5 nonzero coefficients, which would you use --
#     Lasso, LassoLars, or OMP? Why?
# Your answer: