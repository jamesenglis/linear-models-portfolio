# Day 06: ElasticNet + MultiTask Variants
# ElasticNet = Ridge + Lasso combined.
# Objective: min (1/(2n)) * ||Xw - y||^2
#            + alpha * rho * ||w||_1
#            + alpha * (1-rho)/2 * ||w||^2
#   - alpha controls TOTAL regularization strength.
#   - rho (l1_ratio in sklearn) splits between L1 and L2:
#       rho=1.0 -> pure Lasso
#       rho=0.0 -> pure Ridge
#       0<rho<1 -> ElasticNet
# Why it exists: Lasso picks ONE feature at random from a correlated group.
#   Ridge keeps BOTH but never zeroes. ElasticNet keeps BOTH and can zero some.
#   Best of both worlds for correlated features.
#
# MultiTask* variants: for MULTIPLE targets at once (y is 2D).
# Constraint: the SAME features are selected for ALL targets.
# Mixed norm ||W||_{2,1} = sum over features of the L2 norm across tasks.

from sklearn import linear_model
from sklearn.datasets import load_diabetes, make_regression
import numpy as np

# --- Part 1: Toy Example (from sklearn docs) ---
# Purpose: Muscle memory. Note alpha AND l1_ratio are both hyperparameters.
print("--- Toy Example (from sklearn docs) ---")
reg = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.5)
reg.fit([[0, 0], [1, 1]], [0, 1])

print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")
print(f"Prediction for [1, 1]: {reg.predict([[1, 1]])}")

# PREDICT BEFORE RUNNING: With l1_ratio=0.5, will either coefficient
# be exactly zero on this 2-sample toy?
# Your guess:

# --- Part 2: The alpha / l1_ratio Grid on Real Data ---
# Purpose: See how the two hyperparameters interact.
print("\n--- Real Data: alpha x l1_ratio Grid ---")
X, y = load_diabetes(return_X_y=True)

print(f"{'alpha':>6} {'l1_ratio':>9} {'||w||_1':>9} {'nonzero':>8} {'R^2':>7}")
for alpha in [0.01, 0.1, 1.0]:
    for l1_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Note: l1_ratio=0 is exactly Ridge, l1_ratio=1 is exactly Lasso.
        en = linear_model.ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=10000
        ).fit(X, y)
        nz = np.sum(en.coef_ != 0)
        print(f"{alpha:>6} {l1_ratio:>9} {np.linalg.norm(en.coef_, 1):>9.3f} "
              f"{nz:>8} {en.score(X, y):>7.4f}")

# PREDICT BEFORE RUNNING: At alpha=1.0, l1_ratio=0.0 (pure Ridge),
# will any coefficients be exactly zero?
# Your guess:
# At alpha=1.0, l1_ratio=1.0 (pure Lasso), will any coefficients be zero?
# Your guess:

# --- Part 3: Why ElasticNet? (Correlated Features) ---
# Purpose: Show that ElasticNet keeps BOTH correlated features,
# unlike Lasso which picks one at random.
print("\n--- Correlated Features: Lasso vs Ridge vs ElasticNet ---")
rng = np.random.RandomState(0)
x1 = rng.randn(100)
x2 = x1 + rng.randn(100) * 0.01   # x2 is essentially x1
X_corr = np.c_[x1, x2]
y_corr = 3 * x1 + 2 * x2 + rng.randn(100) * 0.1

lasso = linear_model.Lasso(alpha=0.1, max_iter=10000).fit(X_corr, y_corr)
ridge = linear_model.Ridge(alpha=1.0).fit(X_corr, y_corr)
enet = linear_model.ElasticNet(
    alpha=0.1, l1_ratio=0.5, max_iter=10000
).fit(X_corr, y_corr)

print(f"Lasso coefs:      {lasso.coef_}")   # One ~0, one ~5
print(f"Ridge coefs:      {ridge.coef_}")   # Both ~2.5
print(f"ElasticNet coefs: {enet.coef_}")    # Both nonzero, similar

# PREDICT BEFORE RUNNING: Will ElasticNet look more like Lasso or Ridge here?
# Your guess:

# --- Part 4: Manual Math Check (ElasticNet as modified Lasso) ---
# Purpose: ElasticNet with a single feature has a clean closed-form solution.
# It is soft-thresholding applied to a SCALED correlation:
#     w_j = S(corr_j, alpha * rho) / (||x_j||^2 + alpha * (1 - rho))
# where S(z, t) = sign(z) * max(0, |z| - t).
# Notice: Lasso is the special case rho=1 (denominator loses the L2 term).
print("\n--- Manual ElasticNet Soft-Threshold Check ---")
X_1 = X[:, [0]]
y_1 = y.copy()
alpha_test = 10.0
rho_test = 0.5

# Center data
xc = X_1[:, 0] - X_1[:, 0].mean()
yc = y_1 - y_1.mean()

def soft_threshold(z, t):
    return np.sign(z) * max(0.0, abs(z) - t)

corr = np.dot(xc, yc)
denom = np.dot(xc, xc) + alpha_test * (1 - rho_test)
w_manual = soft_threshold(corr, alpha_test * rho_test) / denom
intercept_manual = y_1.mean() - w_manual * X_1[:, 0].mean()

# Compare with sklearn
enet_single = linear_model.ElasticNet(
    alpha=alpha_test, l1_ratio=rho_test, fit_intercept=True, max_iter=10000
)
enet_single.fit(X_1, y_1)

print(f"Manual coefficient:  {w_manual:.6f}")
print(f"Sklearn coefficient: {enet_single.coef_[0]:.6f}")
print(f"Manual intercept:    {intercept_manual:.4f}")
print(f"Sklearn intercept:   {enet_single.intercept_:.4f}")
# Should match closely (small differences from intercept handling).

# --- Part 5: MultiTaskElasticNet on Multi-Target Data ---
# Purpose: When y is 2D (multiple targets), the constraint is that ALL targets
# share the same selected features. Mixed norm ||W||_{2,1} = sum_f ||W_f||_2.
print("\n--- MultiTaskElasticNet (multiple targets) ---")
# Generate a multi-target regression problem
X_mt, Y_mt = make_regression(
    n_samples=100, n_features=10, n_targets=3, n_informative=4,
    noise=0.1, random_state=42
)
print(f"X_mt shape: {X_mt.shape}")   # (100, 10)
print(f"Y_mt shape: {Y_mt.shape}")   # (100, 3)  <- three targets at once

mt = linear_model.MultiTaskElasticNet(
    alpha=0.5, l1_ratio=0.5, max_iter=10000
)
mt.fit(X_mt, Y_mt)

# Coef shape is (n_features, n_targets) for multitask -- note the TRANSPOSE
print(f"Coef shape: {mt.coef_.shape}")   # (10, 3)

# Which features were selected? A feature is "selected" if ANY target uses it.
selected = np.any(mt.coef_ != 0, axis=1)
print(f"Features selected: {np.sum(selected)} / {X_mt.shape[1]}")
print(f"Selected feature indices: {np.where(selected)[0]}")

# Compare with plain Lasso (fit each target separately)
lasso_target_0 = linear_model.Lasso(alpha=0.5, max_iter=10000)
lasso_target_0.fit(X_mt, Y_mt[:, 0])
lasso_target_1 = linear_model.Lasso(alpha=0.5, max_iter=10000)
lasso_target_1.fit(X_mt, Y_mt[:, 1])
indep_selected = np.logical_or(
    lasso_target_0.coef_ != 0,
    lasso_target_1.coef_ != 0,
)
print(f"Independent Lasso (targets 0+1): {np.sum(indep_selected)} features selected")
# MultiTask picks the SAME set for all targets -- that's the "constraint".

# PREDICT BEFORE RUNNING: Will MultiTask select MORE or FEWER features than
# fitting each target independently?
# Your guess:

# --- Part 6: Reflection ---
# Type your answers below.
# Q1: In one sentence, what is l1_ratio and what are its two extremes?
# Your answer:
#
# Q2: Why does ElasticNet beat Lasso on correlated features?
# Your answer:
#
# Q3: What is the "constraint" in MultiTaskElasticNet, in plain English?
# Your answer: