# Day 20: Cross-Decomposition (PLS family + CCA)
# Covers scikit-learn User Guide §1.8
#
# Cross-decomposition finds LATENT VARIABLES that explain the covariance
# between two matrices X and Y. Unlike PCR (unsupervised PCA), PLS uses Y
# to guide the dimensionality reduction [citation:1][citation:2].
#
# Key insight: PLS projects BOTH X and Y into a lower-dimensional subspace
# such that the COVARIANCE between transformed(X) and transformed(Y) is
# maximal [citation:1]. CCA instead maximizes the CORRELATION [citation:5].
#
# Main classes:
#   PLSRegression    : PLS for regression (Y continuous). PLS1 (univariate y)
#                      or PLS2 (multivariate Y) [citation:6].
#   PLSCanonical     : Symmetric PLS; treats X and Y equally, maximizes
#                      covariance of scores [citation:1].
#   CCA              : Canonical Correlation Analysis; maximizes correlation
#                      between X and Y projections [citation:5].
#   PLSSVD           : Simple SVD-based PLS (no deflation) [citation:2].
#
# PLSCanonical algorithm (one component at a time) [citation:1][citation:2]:
#   1. Compute first singular vectors u, v of cross-covariance C = X^T Y.
#   2. Scores: xi = X u, omega = Y v.
#   3. Loadings: gamma = X^T xi / (xi^T xi), delta = Y^T omega / (omega^T omega).
#   4. Deflate: X <- X - xi gamma^T, Y <- Y - omega delta^T.
#   Repeat for K components.
#
# When to use PLS (from docs) [citation:1]:
#   - n_features > n_samples (high-dimensional predictors).
#   - Multicollinearity among X features.
#   - You want a SUPERVISED dimensionality reduction (unlike PCA).

from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# =========================================================================
# PART A: Toy Example -- PLSRegression on simple data
# =========================================================================
print("=" * 70)
print("PART A: Toy Example -- PLSRegression")
print("=" * 70)

# Simple case: y = 2*x1 + 3*x2, no noise
X_toy = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=float)
y_toy = 2 * X_toy[:, 0] + 3 * X_toy[:, 1]

pls = PLSRegression(n_components=2)
pls.fit(X_toy, y_toy)
print(f"Predictions: {np.round(pls.predict(X_toy).ravel(), 4)}")
print(f"True y:      {y_toy}")
print(f"R^2: {pls.score(X_toy, y_toy):.4f}")

# PREDICT BEFORE RUNNING: On this perfectly linear toy with 2 components,
# will PLS get R^2 = 1.0?
# Your guess:

# =========================================================================
# PART B: PLS vs PCR vs OLS on high-dimensional data
# =========================================================================
print("\n" + "=" * 70)
print("PART B: PLS vs PCR vs OLS (high-dimensional)")
print("=" * 70)

# Generate data with n_features > n_samples (the classic PLS use case)
rng = np.random.RandomState(42)
n_samples = 50
n_features = 100
X_hd, y_hd = make_regression(
    n_samples=n_samples, n_features=n_features, n_informative=10,
    noise=10.0, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_hd, y_hd, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# OLS on high-dim: will fail or overfit badly
from sklearn.linear_model import LinearRegression, Ridge
ols = LinearRegression().fit(X_tr_s, y_tr)
print(f"OLS:           train R^2={ols.score(X_tr_s, y_tr):.4f}, "
      f"test R^2={ols.score(X_te_s, y_te):.4f}")

# Ridge (regularized, baseline)
ridge = Ridge(alpha=10.0).fit(X_tr_s, y_tr)
print(f"Ridge:         train R^2={ridge.score(X_tr_s, y_tr):.4f}, "
      f"test R^2={ridge.score(X_te_s, y_te):.4f}")

# PLS with 5 components
for n_comp in [1, 3, 5, 10]:
    pls = PLSRegression(n_components=n_comp).fit(X_tr_s, y_tr)
    test_r2 = pls.score(X_te_s, y_te)
    print(f"PLS (n={n_comp:>2}):      train R^2={pls.score(X_tr_s, y_tr):.4f}, "
          f"test R^2={test_r2:.4f}")

# PREDICT BEFORE RUNNING: Will PLS with 5 components beat Ridge on test R^2?
# (Hint: PLS is supervised, Ridge is not. But Ridge is a strong baseline.)
# Your guess:

# =========================================================================
# PART C: PLS1 (univariate y) vs PLS2 (multivariate Y)
# =========================================================================
print("\n" + "=" * 70)
print("PART C: PLS1 (univariate) vs PLS2 (multivariate)")
print("=" * 70)

# PLS2: multivariate Y (2 targets)
rng = np.random.RandomState(0)
n, p, q = 200, 10, 2
X_multi = rng.randn(n, p)
B_true = np.array([[1, 2] + [0] * (p - 2)] * q).T  # both targets depend on x1, x2
Y_multi = X_multi.dot(B_true) + rng.randn(n, q) * 0.1

pls2 = PLSRegression(n_components=3)
pls2.fit(X_multi, Y_multi)
print(f"PLS2 coef shape: {pls2.coef_.shape}")  # (p, q)
print(f"PLS2 coefs (rows=features, cols=targets):\n{np.round(pls2.coef_, 2)}")
print(f"True B:\n{B_true}")
print("(PLS2 estimates both targets jointly.)")

# PLS1: univariate y (just first target)
y_single = Y_multi[:, 0]
pls1 = PLSRegression(n_components=3)
pls1.fit(X_multi, y_single)
print(f"\nPLS1 coef shape: {pls1.coef_.shape}")  # (p, 1)
print(f"PLS1 coefs: {np.round(pls1.coef_.ravel(), 2)}")
print(f"PLS2 first column: {np.round(pls2.coef_[:, 0], 2)}")
print("(PLS1 and PLS2's first column should be similar.)")

# PREDICT BEFORE RUNNING: Will PLS1 and PLS2's first coefficient column be
# IDENTICAL, similar, or completely different?
# Your guess:

# =========================================================================
# PART D: PLSCanonical -- symmetric treatment
# =========================================================================
print("\n" + "=" * 70)
print("PART D: PLSCanonical (X and Y treated equally)")
print("=" * 70)

# Create X and Y that share latent structure
rng = np.random.RandomState(42)
n = 300
latent = rng.randn(n, 2)  # 2 latent factors
X_shared = latent.dot(rng.randn(2, 4)) + rng.randn(n, 4) * 0.1  # 4 features
Y_shared = latent.dot(rng.randn(2, 3)) + rng.randn(n, 3) * 0.1  # 3 features

plsca = PLSCanonical(n_components=2)
plsca.fit(X_shared, Y_shared)
X_scores, Y_scores = plsca.transform(X_shared, Y_shared)
print(f"X_scores shape: {X_scores.shape}")  # (n, 2)
print(f"Y_scores shape: {Y_scores.shape}")  # (n, 2)

# Correlation between X and Y scores should be high (they share latents)
corr_comp0 = np.corrcoef(X_scores[:, 0], Y_scores[:, 0])[0, 1]
corr_comp1 = np.corrcoef(X_scores[:, 1], Y_scores[:, 1])[0, 1]
print(f"Correlation of X/Y scores (component 0): {corr_comp0:.4f}")
print(f"Correlation of X/Y scores (component 1): {corr_comp1:.4f}")

# PREDICT BEFORE RUNNING: Will the correlation between component-0 scores
# be HIGH (>0.8) or LOW? (Hint: PLSCanonical maximizes covariance, which
# tends to align the scores.)
# Your guess:

# =========================================================================
# PART E: CCA -- maximize CORRELATION (not covariance)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: CCA vs PLSCanonical")
print("=" * 70)
# CCA maximizes the CORRELATION between projections. PLS maximizes
# COVARIANCE. The difference: covariance = correlation * std(X) * std(Y).
# So CCA cares about correlation only; PLS also cares about variance.

cca = CCA(n_components=2)
cca.fit(X_shared, Y_shared)
X_cca, Y_cca = cca.transform(X_shared, Y_shared)
print(f"CCA X_scores shape: {X_cca.shape}")
print(f"CCA Y_scores shape: {Y_cca.shape}")
print(f"CCA correlation component 0: "
      f"{np.corrcoef(X_cca[:, 0], Y_cca[:, 0])[0, 1]:.4f}")
print(f"CCA correlation component 1: "
      f"{np.corrcoef(X_cca[:, 1], Y_cca[:, 1])[0, 1]:.4f}")
print(f"\nPLSCanonical correlation component 0: {corr_comp0:.4f}")
print(f"CCA correlation component 0:           "
      f"{np.corrcoef(X_cca[:, 0], Y_cca[:, 0])[0, 1]:.4f}")

# PREDICT BEFORE RUNNING: Will CCA's correlation be HIGHER, LOWER, or
# EQUAL to PLSCanonical's? (Hint: CCA directly maximizes correlation.)
# Your guess:

# =========================================================================
# PART F: Manual PLS component extraction (prove the math)
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Manual PLS component extraction")
print("=" * 70)
# Do ONE step of PLSCanonical by hand on small data.
# Step 1: SVD of cross-covariance C = X^T Y.
X_small = X_shared[:50]
Y_small = Y_shared[:50]
X_c = X_small - X_small.mean(axis=0)
Y_c = Y_small - Y_small.mean(axis=0)

C = X_c.T.dot(Y_c)  # cross-covariance matrix
U, S, Vt = np.linalg.svd(C, full_matrices=False)
u1 = U[:, 0]  # first left singular vector (X weights)
v1 = Vt[0, :]  # first right singular vector (Y weights)
print(f"u1 (X weights) shape: {u1.shape}, norm: {np.linalg.norm(u1):.4f}")
print(f"v1 (Y weights) shape: {v1.shape}, norm: {np.linalg.norm(v1):.4f}")

# Step 2: Scores
xi = X_c.dot(u1)
omega = Y_c.dot(v1)
print(f"xi (X scores) shape: {xi.shape}")
print(f"omega (Y scores) shape: {omega.shape}")
print(f"Correlation(xi, omega): {np.corrcoef(xi, omega)[0, 1]:.4f}")

# Compare with sklearn's PLSCanonical
plsca_small = PLSCanonical(n_components=1)
plsca_small.fit(X_small, Y_small)
sklearn_scores = plsca_small.transform(X_small, Y_small)[0][:, 0]
print(f"\nSklearn first X score (first 3): {sklearn_scores[:3]}")
print(f"Manual first X score (first 3):  {xi[:3]}")
# Signs may differ (SVD sign ambiguity), but magnitudes should match.

# PREDICT BEFORE RUNNING: Will the manual and sklearn scores match EXACTLY
# or only up to a sign flip?
# Your guess:

# =========================================================================
# PART G: PLSRegression on diabetes
# =========================================================================
print("\n" + "=" * 70)
print("PART G: PLSRegression on real data (diabetes)")
print("=" * 70)
from sklearn.datasets import load_diabetes
X_d, y_d = load_diabetes(return_X_y=True)
X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)
scaler_d = StandardScaler()
X_d_tr_s = scaler_d.fit_transform(X_d_tr)
X_d_te_s = scaler_d.transform(X_d_te)

print(f"{'n_components':>12} {'train_R2':>10} {'test_R2':>10}")
for n_comp in [1, 2, 3, 5, 10]:
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_d_tr_s, y_d_tr)
    print(f"{n_comp:>12} {pls.score(X_d_tr_s, y_d_tr):>10.4f} "
          f"{pls.score(X_d_te_s, y_d_te):>10.4f}")

# Ridge baseline (from Day 2)
from sklearn.linear_model import Ridge
ridge_d = Ridge(alpha=1.0).fit(X_d_tr_s, y_d_tr)
print(f"\nRidge test R^2: {ridge_d.score(X_d_te_s, y_d_te):.4f}")

# PREDICT BEFORE RUNNING: Will PLS with 3 components beat Ridge on diabetes?
# (Hint: diabetes is mostly linear; PLS's latent variable approach may or
# may not help. Ridge is a strong baseline.)
# Your guess:

# =========================================================================
# PART H: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART H: Reflection -- section 1.8")
print("=" * 70)

# Q1: In one sentence, what does PLS do that PCA does not?
# Your answer:
#
# Q2: What is the key difference between PLS and CCA?
#     (Hint: covariance vs correlation.)
# Your answer:
#
# Q3: When is PLS particularly useful?
#     (Hint: think about n_features vs n_samples and multicollinearity.)
# Your answer:
#
# Q4: What does `n_components` control in PLSRegression?
# Your answer:
#
# Q5: Why might PLS be preferred over Ridge for high-dimensional data?
# Your answer: