# Day 10 (FINAL): Generalized Linear Models (GLM / Tweedie)
#               + Pipeline + PolynomialFeatures
#
# PART A -- GLM:
#   Linear models so far assumed y is Gaussian: y = Xw + noise.
#   GLMs generalize this in two ways:
#     1. The mean of y is a LINK FUNCTION h of Xw:
#            y_hat = h(Xw)
#     2. The loss is a UNIT DEVIANCE d(y, y_hat) from an exponential family.
#   Objective: min (1/(2n)) * sum_i d(y_i, y_hat_i) + alpha/2 * ||w||^2
#
#   Common distributions (chosen via `power` in TweedieRegressor):
#     power=0 -> Normal        (link: identity)     -> like Ridge
#     power=1 -> Poisson       (link: log)          -> counts
#     power=2 -> Gamma         (link: log)          -> positive skewed
#     power=3 -> Inverse Gauss (link: log)          -> heavier tails
#
# PART B -- PIPELINE + POLYNOMIAL:
#   Linear models are still "linear" after adding polynomial features.
#   z = [x1, x2, x1^2, x1*x2, x2^2] is just a NEW feature space.
#   Fit a LinearRegression on z -> still a linear model (in w), but it
#   fits a curved surface in the original x space.
#   PolynomialFeatures(degree=d) generates all monomials up to degree d.
#   interaction_only=True -> only products (no powers).
#
# PART C -- WHY PIPELINE:
#   Pipeline chains transformers + estimator. It makes the whole thing
#   behave like one estimator: fit(X, y) fits scaler THEN model; predict(X)
#   applies scaler THEN model. No leakage from test to train.

from sklearn import linear_model
from sklearn.datasets import load_diabetes, make_regression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

# =========================================================================
# PART A: Generalized Linear Models (TweedieRegressor)
# =========================================================================
print("=" * 70)
print("PART A: Generalized Linear Models")
print("=" * 70)

# --- A1: Toy Example (from sklearn docs) ---
print("\n--- A1: Toy Example (from sklearn docs) ---")
reg = linear_model.TweedieRegressor(power=1, alpha=0.5, link="log")
reg.fit([[0, 0], [0, 1], [2, 2]], [0, 1, 2])
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")
print(f"Predictions for [[1, 1]]: {reg.predict([[1, 1]])}")

# PREDICT BEFORE RUNNING: Since link='log', the prediction is
# exp(Xw + w0). Would a SMALLER Xw give a HIGHER or LOWER prediction?
# Your guess:

# --- A2: Poisson vs Gaussian on Count Data ---
# Purpose: When y is counts (non-negative integers), use Poisson.
print("\n--- A2: Poisson vs Gaussian on Counts ---")
rng = np.random.RandomState(42)
X_p = rng.randn(500, 3)
# Poisson mean is exp(Xw). We generate counts accordingly.
true_w = np.array([0.5, -0.3, 0.8])
mu = np.exp(X_p.dot(true_w))
y_p = rng.poisson(mu).astype(float)
print(f"y_p stats: mean={y_p.mean():.2f}, min={y_p.min():.0f}, "
      f"max={y_p.max():.0f}")
print(f"y_p is non-negative integers? {np.all(y_p >= 0) and np.all(y_p == np.round(y_p))}")

poisson = linear_model.PoissonRegressor(alpha=0.01, max_iter=1000)
poisson.fit(X_p, y_p)
print(f"True w:    {true_w}")
print(f"Poisson w: {poisson.coef_}")   # Should be close to true w

# Compare with plain linear regression (which can go negative)
ols = linear_model.LinearRegression().fit(X_p, y_p)
print(f"OLS w:     {ols.coef_}")
print(f"Poisson R^2: {poisson.score(X_p, y_p):.4f}")
print(f"OLS R^2:     {ols.score(X_p, y_p):.4f}")

# PREDICT BEFORE RUNNING: Will OLS sometimes predict NEGATIVE counts?
# (OLS has no constraint on the output sign.)
# Your guess:

# Check the min predicted value for OLS vs Poisson
ols_pred = ols.predict(X_p)
poisson_pred = poisson.predict(X_p)
print(f"OLS min prediction:     {ols_pred.min():.3f}  (can be negative!)")
print(f"Poisson min prediction: {poisson_pred.min():.3f}  (always > 0)")

# --- A3: TweedieRegressor as a family (power parameter) ---
print("\n--- A3: TweedieRegressor family (power 0..3) ---")
for power in [0, 1, 2, 3]:
    reg = linear_model.TweedieRegressor(power=power, alpha=0.1, link="log")
    # Use positive y for power >= 1
    y_pos = y_p + 1.0  # shift to strictly positive
    try:
        reg.fit(X_p, y_pos)
        print(f"power={power}: coef={np.round(reg.coef_, 3)}, "
              f"intercept={reg.intercept_:.3f}")
    except Exception as e:
        print(f"power={power}: FAILED ({e})")

# PREDICT BEFORE RUNNING: Which power should give coefficients CLOSEST to
# the true w = [0.5, -0.3, 0.8]?
# Hint: we generated y with a Poisson (power=1).
# Your guess:

# --- A4: GammaRegressor (for skewed positive targets) ---
print("\n--- A4: GammaRegressor (skewed positive targets) ---")
# Generate skewed positive data: Gamma with mean = exp(Xw)
shape_k = 5.0
y_gamma = rng.gamma(shape=shape_k, scale=mu / shape_k) + 0.01
gamma = linear_model.GammaRegressor(alpha=0.01, max_iter=1000)
gamma.fit(X_p, y_gamma)
print(f"True w:    {true_w}")
print(f"Gamma w:   {gamma.coef_}")
print(f"Gamma R^2: {gamma.score(X_p, y_gamma):.4f}")

# =========================================================================
# PART B: Pipeline + PolynomialFeatures
# =========================================================================
print("\n" + "=" * 70)
print("PART B: Pipeline + PolynomialFeatures")
print("=" * 70)

# --- B1: PolynomialFeatures toy example (from sklearn docs) ---
print("\n--- B1: PolynomialFeatures toy ---")
X_poly = np.arange(6).reshape(3, 2)
print(f"Original X:\n{X_poly}")
poly = PolynomialFeatures(degree=2)
X_transformed = poly.fit_transform(X_poly)
print(f"Transformed (degree=2):\n{X_transformed}")
print(f"Feature names: {poly.get_feature_names_out(['x1', 'x2'])}")
# Columns are [1, x1, x2, x1^2, x1*x2, x2^2].
# The first column of 1s is the intercept -- pairs with fit_intercept=False.

# PREDICT BEFORE RUNNING: With degree=2 and 2 input features, how many
# output columns will there be? (Hint: 1 + 2 + 3 = ...)
# Your guess:

# --- B2: Polynomial regression via Pipeline ---
# Purpose: Fit a cubic to nonlinear data using LinearRegression.
print("\n--- B2: Polynomial regression via Pipeline ---")
# True relationship: y = 3 - 2x + x^2 - x^3
x = np.arange(20).astype(float)
y_poly = 3 - 2 * x + x**2 - x**3
# Add some noise so it's not trivially exact
y_poly += np.random.RandomState(0).randn(20) * 0.5

model = Pipeline([
    ("poly", PolynomialFeatures(degree=3)),
    ("linear", linear_model.LinearRegression(fit_intercept=False)),
])
model.fit(x[:, np.newaxis], y_poly)

print(f"Learned coefs: {model.named_steps['linear'].coef_}")
print(f"True coefs:    [3., -2., 1., -1.]")
print(f"R^2: {model.score(x[:, np.newaxis], y_poly):.4f}")
# Without fit_intercept=False, the model would add its OWN intercept on
# top of the polynomial's constant column -> collinearity. Important detail.

# PREDICT BEFORE RUNNING: Will the learned coefficients match the true
# coefficients EXACTLY, APPROXIMATELY, or NOT AT ALL (given the noise)?
# Your guess:

# --- B3: Pipeline in practice (standardize + Ridge) ---
# Purpose: Pipelines chain preprocessing and model -- proper workflow.
print("\n--- B3: Pipeline of StandardScaler + Ridge ---")
X_d, y_d = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("ridge", linear_model.Ridge(alpha=1.0)),
])
pipe.fit(X_tr, y_tr)
print(f"Train R^2: {pipe.score(X_tr, y_tr):.4f}")
print(f"Test R^2:  {pipe.score(X_te, y_te):.4f}")
# Notice: The scaler was fit ONLY on the training set. No leakage.

# --- B4: Polynomial + Ridge on nonlinear diabetes? ---
print("\n--- B4: PolynomialFeatures(degree=2) + Ridge ---")
pipe_poly = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", linear_model.Ridge(alpha=10.0)),
])
pipe_poly.fit(X_tr, y_tr)
print(f"Train R^2: {pipe_poly.score(X_tr, y_tr):.4f}")
print(f"Test R^2:  {pipe_poly.score(X_te, y_te):.4f}")
print(f"Number of features after poly: "
      f"{pipe_poly.named_steps['poly'].n_output_features_}")

# PREDICT BEFORE RUNNING: Will polynomial degree=2 OVERFIT or UNDERFIT
# on diabetes? (10 features, 353 training samples, degree=2 -> 65 features)
# Your guess:

# =========================================================================
# PART C: Reflection + Portfolio Wrap-Up
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Reflection")
print("=" * 70)

# --- C1: When to use what ---
print("""
--- Decision Guide (memorize this) ---

Regression (continuous y):
  - Baseline                 -> LinearRegression
  - Correlated features       -> Ridge (or RidgeCV)
  - Sparse solution wanted    -> Lasso (or LassoCV)
  - Correlated + sparse       -> ElasticNet
  - Learned regularization    -> BayesianRidge (dense) / ARD (sparse)
  - High-dim, forward select  -> LARS / OMP
  - Counts (non-negative)     -> PoissonRegressor
  - Skewed positive           -> GammaRegressor
  - Any Tweedie family        -> TweedieRegressor(power=...)
  - Nonlinear curve           -> PolynomialFeatures + (Ridge|Linear)

Classification (categorical y):
  - Default                   -> LogisticRegression(solver='lbfgs')
  - Sparse (L1)               -> LogisticRegression(solver='saga', penalty='l1')
  - Multiclass (true multinomial) -> solver='lbfgs' or 'newton-cg' or 'saga'
  - Very large n, many feats  -> solver='sag' or 'saga'
  - Tiny n, L1 or OvR         -> solver='liblinear'
""")

# --- C2: Reflection questions ---
# Q1: In one sentence, what is a "link function" in a GLM?
# Your answer:
#
# Q2: Why does PoissonRegressor never predict negative y? (What in the
#     math enforces this?)
# Your answer:
#
# Q3: Why is Pipeline important for cross-validation? (What would go wrong
#     if you fit StandardScaler on the FULL dataset before CV?)
# Your answer:
#
# Q4: PolynomialFeatures(degree=3) on 2 input features -> how many columns?
# Your answer:

# --- C3: Portfolio completeness check ---
print("\n--- Portfolio Files Should Now Be: ---")
print("src/day01_ols.py")
print("src/day02_ridge.py")
print("src/day03_ridgecv.py")
print("src/day04_lasso.py")
print("src/day05_lassocv_ic.py")
print("src/day06_elasticnet.py")
print("src/day07_lars_omp.py")
print("src/day08_bayesian_ard.py")
print("src/day09_logistic.py")
print("src/day10_glm_pipeline.py")
print("\nIf any are missing, you skipped a day. Go back and do it.")