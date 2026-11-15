# Day 19: Gaussian Processes (GP)
# Covers scikit-learn User Guide §1.7
#
# A GP is a BAYESIAN NON-PARAMETRIC model: it places a prior over FUNCTIONS
# (not parameters). The posterior gives you not just a prediction, but a
# full Gaussian distribution over the value at each test point.
#
# Outputs:
#   y_pred, y_std = gp.predict(X_test, return_std=True)
# That y_std is the unique advantage of GPs over Ridge/SVR.
#
# GP regression objective:
#   Prior:       f(x) ~ GP(0, K(x, x'))
#   Likelihood:  y = f(x) + noise, noise ~ N(0, alpha)
#   Posterior:   f_* | X, y, X_* ~ N(mu_*, Sigma_*)
#   mu_*    = K(X_*, X) [K(X, X) + alpha*I]^-1 y
#   Sigma_* = K(X_*, X_*) - K(X_*, X) [K(X, X) + alpha*I]^-1 K(X, X_*)
#
# Kernels (the covariance function defines the prior):
#   RBF / Matern / RationalQuadratic / ExpSineSquared / DotProduct
#   Additive / multiplicative combinations via `+` and `*`
#
# Key properties:
#   - Uncertainty SHRINKS near training points, GROWS away from them.
#   - Very expensive: O(n^3) training (matrix inversion), O(n^2) prediction.
#   - High-dimensional problems are slow; best for n <= ~10,000.
#   - Kernel hyperparameters can be tuned by maximizing the LOG-MARGINAL
#     LIKELIHOOD (no validation set needed).

from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel as C, WhiteKernel, RationalQuadratic
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import time
import warnings

# =========================================================================
# PART A: Toy Example -- prediction with std
# =========================================================================
print("=" * 70)
print("PART A: Toy Example")
print("=" * 70)

X_toy = np.array([[1, 2], [3, 4]])
y_toy = np.array([1, 2])
gp = GaussianProcessRegressor()
gp.fit(X_toy, y_toy)
y_pred, sigma = gp.predict([[5, 6]], return_std=True)
print(f"Mean prediction: {y_pred[0]:.4f}")
print(f"Std  prediction: {sigma[0]:.4f}")
print(f"5% CI: [{y_pred[0] - 1.96 * sigma[0]:.2f}, "
      f"{y_pred[0] + 1.96 * sigma[0]:.2f}]")

# PREDICT BEFORE RUNNING: On just 2 training points, will the std at [5, 6]
# be SMALL or LARGE? (Hint: [5, 6] is far from both training points, so
# the prior dominates -> large std.)
# Your guess:

# =========================================================================
# PART B: Uncertainty grows away from data
# =========================================================================
print("\n" + "=" * 70)
print("PART B: Uncertainty vs distance from training data")
print("=" * 70)
# 1D example with a nonlinear function
rng = np.random.RandomState(42)
X_1d = np.sort(rng.uniform(-3, 3, size=10))[:, np.newaxis]
y_1d = np.sin(X_1d[:, 0]) + rng.randn(10) * 0.05

gp = GaussianProcessRegressor(
    kernel=C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01),
    n_restarts_optimizer=5, random_state=42,
)
gp.fit(X_1d, y_1d)
print(f"Learned kernel: {gp.kernel_}")

# Test at points inside and far outside the training range
X_test = np.array([[-3.0], [-2.0], [0.0], [2.0], [3.0], [6.0], [10.0]])
y_pred, y_std = gp.predict(X_test, return_std=True)
print(f"\n{'x':>6} {'mean':>10} {'std':>10} {'note':<30}")
for x, m, s in zip(X_test[:, 0], y_pred, y_std):
    if x < -3 or x > 3:
        note = "extrapolation (std should be large)"
    elif abs(x) < 1:
        note = "near training (std should be small)"
    else:
        note = ""
    print(f"{x:>6.1f} {m:>10.4f} {s:>10.4f} {note:<30}")

# PREDICT BEFORE RUNNING: Will std at x=10 be SMALLER or LARGER than std
# at x=2? (x=2 is inside the training range; x=10 is far outside.)
# Your guess:

# =========================================================================
# PART C: Kernel shapes -- RBF, Matern, RationalQuadratic
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Different kernels on the same 1D data")
print("=" * 70)
kernels = {
    "RBF":                 C(1.0) * RBF(length_scale=1.0),
    "Matern(nu=1.5)":      C(1.0) * Matern(length_scale=1.0, nu=1.5),
    "Matern(nu=2.5)":      C(1.0) * Matern(length_scale=1.0, nu=2.5),
    "RationalQuadratic":   C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0),
    "DotProduct":          C(1.0) * DotProduct(sigma_0=1.0),
}
print(f"{'kernel':<20} {'log-marg-lik':>15} {'train_MSE':>10}")
for name, kernel in kernels.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # some kernels warn on optimization
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp.fit(X_1d, y_1d)
    train_mse = mean_squared_error(y_1d, gp.predict(X_1d))
    print(f"{name:<20} {gp.log_marginal_likelihood_value_:>15.4f} "
          f"{train_mse:>10.6f}")
print("(Higher log-marginal-likelihood = better fit to the prior.)")

# PREDICT BEFORE RUNNING: Will the TRUE kernel (RBF used to generate the
# data) have the HIGHEST log-marginal-likelihood? (Hint: usually yes, but
# Matern(2.5) often gets close because it also fits smooth functions well.)
# Your guess:

# =========================================================================
# PART D: GP regression on real data (diabetes, 1 feature)
# =========================================================================
print("\n" + "=" * 70)
print("PART D: GP regression (diabetes, 1D)")
print("=" * 70)
from sklearn.datasets import load_diabetes
Xr, yr = load_diabetes(return_X_y=True)
Xr_1d = Xr[:, [2]]  # use one feature for a clean 1D example
X_tr, X_te, yr_tr, yr_te = train_test_split(
    Xr_1d, yr, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

gp = GaussianProcessRegressor(
    kernel=C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
    n_restarts_optimizer=3, random_state=42, normalize_y=True,
)
t0 = time.time()
gp.fit(X_tr_s, yr_tr)
print(f"Fit time: {time.time() - t0:.2f} s")
print(f"Learned kernel: {gp.kernel_}")
y_pred, y_std = gp.predict(X_te_s, return_std=True)
print(f"Test R^2:  {gp.score(X_te_s, yr_te):.4f}")
print(f"Test MSE:  {mean_squared_error(yr_te, y_pred):.4f}")
print(f"Mean std:  {y_std.mean():.4f}")

# PREDICT BEFORE RUNNING: Will the 95% interval [y_pred - 1.96*std,
# y_pred + 1.96*std] cover approximately 95% of test points?
# Your guess:

coverage = np.mean(
    (yr_te >= y_pred - 1.96 * y_std) & (yr_te <= y_pred + 1.96 * y_std)
)
print(f"Empirical 95% coverage: {coverage:.2%}  (target: ~95%)")

# =========================================================================
# PART E: GP Classification on Iris (2 classes)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: GP Classification (Iris, 2 classes)")
print("=" * 70)
Xc, yc = load_iris(return_X_y=True)
mask = yc < 2  # binary: setosa vs versicolor
Xc_bin = Xc[mask]
yc_bin = yc[mask]
# Use 2 features for speed and visualization-friendly setup
Xc_bin = Xc_bin[:, :2]

X_tr, X_te, yc_tr, yc_te = train_test_split(
    Xc_bin, yc_bin, test_size=0.3, random_state=42, stratify=yc_bin
)
scaler_c = StandardScaler()
X_tr_s = scaler_c.fit_transform(X_tr)
X_te_s = scaler_c.transform(X_te)

gpc = GaussianProcessClassifier(
    kernel=C(1.0) * RBF(length_scale=1.0),
    random_state=42, n_restarts_optimizer=2,
)
t0 = time.time()
gpc.fit(X_tr_s, yc_tr)
print(f"Fit time: {time.time() - t0:.2f} s")
print(f"Learned kernel: {gpc.kernel_}")
print(f"Train acc: {gpc.score(X_tr_s, yc_tr):.4f}")
print(f"Test acc:  {gpc.score(X_te_s, yc_te):.4f}")

# predict_proba gives calibrated probabilities
proba = gpc.predict_proba(X_te_s[:5])
print(f"\nFirst 5 test predictions with probabilities:")
for i in range(5):
    print(f"  pred={gpc.predict([X_te_s[i]])[0]}, "
          f"P(class 1)={proba[i, 1]:.3f}, true={yc_te[i]}")

# PREDICT BEFORE RUNNING: Will GP classification accuracy be HIGHER, LOWER,
# or SIMILAR to SVC (Day 16) on this 2-feature Iris slice?
# (Hint: both are strong kernel-based models; differences are small.)
# Your guess:

# =========================================================================
# PART F: Manual GP mean computation (proves the math)
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Manual GP mean computation")
print("=" * 70)
# Use the simple 1D toy data from Part B
# Step 1: Build K(X, X) for training data
kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
# Compute kernel matrices manually via the kernel object
K_XX = kernel(X_1d)  # (n, n)
K_XsX = kernel(X_test, X_1d)  # (n_test, n)
K_XsXs = kernel(X_test)  # (n_test, n_test)

# mu_* = K(X_*, X) [K(X, X) + alpha*I]^-1 y
# Note: in sklearn, alpha is set on the GP object (default 1e-10); the
# WhiteKernel adds noise on top.
alpha = 1e-10
n = X_1d.shape[0]
K_inv = np.linalg.inv(K_XX + alpha * np.eye(n))
mu_manual = K_XsX.dot(K_inv).dot(y_1d)
# Sigma_* = K(X_*, X_*) - K(X_*, X) K^-1 K(X, X_*)
Sigma_manual = K_XsXs - K_XsX.dot(K_inv).dot(K_XsX.T)
std_manual = np.sqrt(np.diag(Sigma_manual))

# Compare with sklearn GP fitted on the same data
gp_manual = GaussianProcessRegressor(kernel=kernel, optimizer=None)
gp_manual.fit(X_1d, y_1d)
mu_sklearn, std_sklearn = gp_manual.predict(X_test, return_std=True)

print(f"{'x':>6} {'manual_mean':>12} {'sklearn_mean':>14} "
      f"{'manual_std':>12} {'sklearn_std':>14}")
for i, x in enumerate(X_test[:, 0]):
    print(f"{x:>6.1f} {mu_manual[i]:>12.4f} {mu_sklearn[i]:>14.4f} "
          f"{std_manual[i]:>12.4f} {std_sklearn[i]:>14.4f}")

# PREDICT BEFORE RUNNING: Will manual and sklearn predictions match
# EXACTLY? (Hint: we set optimizer=None so the kernel is FIXED, so the
# matrix math should agree to floating point.)
# Your guess:

# =========================================================================
# PART G: Cost scaling (n^3)
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Fit time scales as O(n^3)")
print("=" * 70)
print(f"{'n_train':>10} {'fit_time_s':>12}")
for n_train in [100, 300, 1000, 2000]:
    rng = np.random.RandomState(42)
    X_n = rng.randn(n_train, 3)
    y_n = np.sin(X_n[:, 0]) + rng.randn(n_train) * 0.1
    gp = GaussianProcessRegressor(
        kernel=C(1.0) * RBF(length_scale=1.0), random_state=42
    )
    t0 = time.time()
    gp.fit(X_n, y_n)
    print(f"{n_train:>10} {time.time() - t0:>12.2f}")

# PREDICT BEFORE RUNNING: Going from 1000 to 2000 samples, will fit time
# grow roughly 2x, 4x, or 8x? (Hint: O(n^3) => 2^3 = 8x.)
# Your guess:

# =========================================================================
# PART H: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART H: Reflection -- section 1.7")
print("=" * 70)

# Q1: In one sentence, what is a Gaussian Process?
# Your answer:
#
# Q2: What is the unique advantage of GPs over Ridge / SVR?
# Your answer:
#
# Q3: Why is GP O(n^3)? What does that mean in practice?
# Your answer:
#
# Q4: What happens to GP uncertainty far away from training data?
# Your answer:
#
# Q5: In GP classification, why is the model fit iterative instead of
#     closed-form? (Hint: classification uses a non-Gaussian likelihood,
#     so the posterior is not analytic. Laplace approximation is used.)
# Your answer: