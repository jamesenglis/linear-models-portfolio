# Day 15: Kernel Ridge Regression (KRR)
# Covers scikit-learn User Guide §1.3
#
# Idea: Ridge regression learns a LINEAR function w^T x + b.
# If the relationship is nonlinear, we can still use Ridge -- but in a
# HIGHER-DIMENSIONAL feature space phi(x). The kernel trick says we never
# need to compute phi(x) explicitly; we only need the kernel:
#
#     K(x_i, x_j) = phi(x_i)^T phi(x_j)
#
# KernelRidge solves:
#     min_alpha  ||y - K alpha||^2 + alpha_reg * alpha^T K alpha
# Closed form:
#     alpha = (K + alpha_reg * I)^-1 y
# Prediction:
#     y_hat(x) = sum_i alpha_i * K(x_i, x)
#
# Key contrast with SVR (Support Vector Regression, §1.4):
#   - KRR uses SQUARED error loss. Closed-form fit.
#   - SVR uses EPSILON-INSENSITIVE loss. Iterative fit. Sparse solution.
#   - KRR: faster to FIT on medium data (< ~1000 samples); dense model.
#   - SVR: faster to PREDICT (sparse); scales better for large n.
#
# Kernel choices (same as SVM):
#   'linear'  -> K(x, z) = x^T z                  (reduces to Ridge)
#   'rbf'     -> K(x, z) = exp(-gamma * ||x-z||^2) (most common nonlinear)
#   'poly'    -> K(x, z) = (gamma * x^T z + r)^d
#   'sigmoid' -> K(x, z) = tanh(gamma * x^T z + r)

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import numpy as np
import time

# =========================================================================
# PART A: Toy Example (linear vs RBF kernel)
# =========================================================================
print("=" * 70)
print("PART A: Toy Example -- kernel choice matters")
print("=" * 70)
# A simple 1D nonlinear function: y = sin(x) + noise
rng = np.random.RandomState(42)
X_toy = np.sort(rng.uniform(-3, 3, size=20))[:, np.newaxis]
y_toy = np.sin(X_toy[:, 0]) + rng.randn(20) * 0.1

# Linear kernel (basically Ridge)
krr_lin = KernelRidge(kernel="linear", alpha=0.1)
krr_lin.fit(X_toy, y_toy)

# RBF kernel (nonlinear)
krr_rbf = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.5)
krr_rbf.fit(X_toy, y_toy)

print(f"Linear kernel train MSE: {mean_squared_error(y_toy, krr_lin.predict(X_toy)):.4f}")
print(f"RBF kernel train MSE:    {mean_squared_error(y_toy, krr_rbf.predict(X_toy)):.4f}")
print("(RBF should fit sin(x) much better than linear.)")

# PREDICT BEFORE RUNNING: Will the RBF kernel get LOWER training MSE?
# (Hint: sin is nonlinear; linear kernel can only fit a straight line.)
# Your guess:

# =========================================================================
# PART B: KRR vs Ridge vs SVR on nonlinear data
# =========================================================================
print("\n" + "=" * 70)
print("PART B: KRR vs Ridge vs SVR")
print("=" * 70)
# Generate a nonlinear regression problem
X, y = make_regression(
    n_samples=300, n_features=5, n_informative=3, noise=10.0, random_state=42
)
# Add a nonlinear interaction term to y so linear models struggle
y = y + 2.0 * np.sin(X[:, 0]) * X[:, 1]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features (SVR and KRR are both scale-sensitive)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Ridge (linear baseline)
ridge = Ridge(alpha=1.0).fit(X_tr_s, y_tr)
print(f"Ridge (linear)       test MSE: {mean_squared_error(y_te, ridge.predict(X_te_s)):.4f}")

# KernelRidge with RBF
t0 = time.time()
krr = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1)
krr.fit(X_tr_s, y_tr)
krr_fit_time = time.time() - t0
print(f"KernelRidge (RBF)    test MSE: {mean_squared_error(y_te, krr.predict(X_te_s)):.4f}, "
      f"fit time: {krr_fit_time*1000:.1f} ms")

# SVR with RBF
t0 = time.time()
svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_tr_s, y_tr)
svr_fit_time = time.time() - t0
print(f"SVR (RBF)            test MSE: {mean_squared_error(y_te, svr.predict(X_te_s)):.4f}, "
      f"fit time: {svr_fit_time*1000:.1f} ms")
print(f"SVR support vectors: {np.sum(svr.support_)} / {len(y_tr)}")

# PREDICT BEFORE RUNNING: Which will have LOWER test MSE -- Ridge or KRR(RBF)?
# (The data has a nonlinear term, so a linear model should underfit.)
# Your guess:

# =========================================================================
# PART C: Manual KRR closed-form check
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Manual KRR (closed-form verification)")
print("=" * 70)
# Take a small subset so the math is manageable
n_small = 30
X_small = X_tr_s[:n_small]
y_small = y_tr[:n_small]

# Step 1: Build the RBF kernel matrix K
gamma = 0.1
def rbf_kernel(A, B, gamma):
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    dists = (np.sum(A**2, axis=1)[:, None]
             + np.sum(B**2, axis=1)[None, :]
             - 2 * A.dot(B.T))
    return np.exp(-gamma * dists)

K = rbf_kernel(X_small, X_small, gamma)
print(f"Kernel matrix K shape: {K.shape}")
print(f"K diagonal (should all be 1.0): {np.round(np.diag(K)[:5], 4)}")

# Step 2: Solve (K + alpha * I) alpha_vec = y
alpha_reg = 0.1
alpha_vec = np.linalg.solve(K + alpha_reg * np.eye(n_small), y_small)

# Step 3: Predict on training points
y_manual_pred = K.dot(alpha_vec)

# Compare with sklearn
krr_small = KernelRidge(kernel="rbf", alpha=alpha_reg, gamma=gamma)
krr_small.fit(X_small, y_small)
y_sklearn_pred = krr_small.predict(X_small)

print(f"Manual first 3 predictions:  {np.round(y_manual_pred[:3], 6)}")
print(f"Sklearn first 3 predictions: {np.round(y_sklearn_pred[:3], 6)}")
print(f"Max abs diff: {np.max(np.abs(y_manual_pred - y_sklearn_pred)):.2e}")
# Should be near-zero (floating point differences only).

# PREDICT BEFORE RUNNING: Will manual and sklearn predictions match EXACTLY
# (to floating point), or be slightly different?
# Your guess:

# =========================================================================
# PART D: gamma (RBF bandwidth) effect
# =========================================================================
print("\n" + "=" * 70)
print("PART D: gamma (RBF bandwidth) effect")
print("=" * 70)
print(f"{'gamma':>10} {'train_MSE':>12} {'test_MSE':>12} {'note':<30}")
for gamma in [0.001, 0.01, 0.1, 1.0, 10.0]:
    krr = KernelRidge(kernel="rbf", alpha=0.1, gamma=gamma)
    krr.fit(X_tr_s, y_tr)
    train_mse = mean_squared_error(y_tr, krr.predict(X_tr_s))
    test_mse = mean_squared_error(y_te, krr.predict(X_te_s))
    note = ""
    if gamma < 0.01:
        note = "too smooth (underfit)"
    elif gamma > 1.0:
        note = "too spiky (overfit)"
    print(f"{gamma:>10} {train_mse:>12.4f} {test_mse:>12.4f} {note:<30}")

# PREDICT BEFORE RUNNING: As gamma goes UP, does the model get MORE or LESS
# flexible? (Hint: gamma controls the width of the RBF -- larger gamma =
# narrower bump = more local influence.)
# Your guess:

# =========================================================================
# PART E: alpha (regularization) effect
# =========================================================================
print("\n" + "=" * 70)
print("PART E: alpha (L2 regularization) effect")
print("=" * 70)
print(f"{'alpha':>10} {'train_MSE':>12} {'test_MSE':>12}")
for alpha in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
    krr = KernelRidge(kernel="rbf", alpha=alpha, gamma=0.1)
    krr.fit(X_tr_s, y_tr)
    train_mse = mean_squared_error(y_tr, krr.predict(X_tr_s))
    test_mse = mean_squared_error(y_te, krr.predict(X_te_s))
    print(f"{alpha:>10} {train_mse:>12.4f} {test_mse:>12.4f}")

# PREDICT BEFORE RUNNING: As alpha goes UP, will ||coef|| shrink or grow?
# (Same intuition as Ridge: larger alpha = more shrink.)
# Your guess:

# =========================================================================
# PART F: KRR vs SVR -- sparsity and speed
# =========================================================================
print("\n" + "=" * 70)
print("PART F: KRR vs SVR -- fit time, predict time, sparsity")
print("=" * 70)
# Fit both on the full training set
krr_full = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1)
svr_full = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)

t0 = time.time(); krr_full.fit(X_tr_s, y_tr); krr_fit = time.time() - t0
t0 = time.time(); svr_full.fit(X_tr_s, y_tr); svr_fit = time.time() - t0

t0 = time.time(); krr_full.predict(X_te_s); krr_pred = time.time() - t0
t0 = time.time(); svr_full.predict(X_te_s); svr_pred = time.time() - t0

print(f"KRR fit:     {krr_fit*1000:>8.1f} ms  |  predict: {krr_pred*1000:>6.1f} ms")
print(f"SVR fit:     {svr_fit*1000:>8.1f} ms  |  predict: {svr_pred*1000:>6.1f} ms")
print(f"SVR support vectors: {np.sum(svr_full.support_)} / {len(y_tr)} "
      f"({np.mean(svr_full.support_)*100:.1f}%)")
print("(SVR is sparse: uses only support vectors. KRR uses ALL training points.)")

# PREDICT BEFORE RUNNING: Which will be FASTER to predict -- KRR or SVR?
# (Hint: SVR only needs to evaluate the kernel on support vectors.)
# Your guess:

# =========================================================================
# PART G: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Reflection -- section 1.3")
print("=" * 70)

# Q1: In one sentence, what is Kernel Ridge Regression?
# Your answer:
#
# Q2: What is the "kernel trick" and why is it useful?
# Your answer:
#
# Q3: When would you choose KRR over SVR?
#     (Hint: think about fit vs predict time, and sparsity.)
# Your answer:
#
# Q4: What does gamma control in the RBF kernel?
# Your answer:
#
# Q5: If you set kernel='linear' in KernelRidge, what model do you get?
# Your answer: