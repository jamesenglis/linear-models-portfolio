# Day 05: LassoCV + Information Criteria (AIC / BIC)
# Two ways to pick the Lasso alpha automatically:
#   1. LassoCV  -> k-fold or LOO cross-validation (cost: k+1 fits)
#   2. LassoLarsIC -> AIC or BIC score in ONE pass (cost: 1 fit along the path)
# Math (AIC): AIC = -2 * log(L_hat) + 2 * d
# Math (BIC): BIC = -2 * log(L_hat) + log(n) * d
#   where d = number of nonzero coefficients (effective degrees of freedom)
#         L_hat = maximum likelihood of the fitted model
# BIC penalizes complexity harder than AIC (log(n) > 2 when n > 7).
# Both are CHEAP alternatives to CV, but assume large n and a correct model.

from sklearn import linear_model
from sklearn.datasets import load_diabetes
import numpy as np

# --- Part 1: Toy Example (from sklearn docs) ---
# Purpose: Muscle memory. LassoCV takes a LIST of alphas, just like RidgeCV.
print("--- Toy Example (from sklearn docs) ---")
reg = linear_model.LassoCV(cv=5)
reg.fit([[0, 0], [0, 0.1], [1, 1]], [0, -0.1, 1])

print(f"Best alpha chosen: {reg.alpha_}")
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

# PREDICT BEFORE RUNNING: Will LassoCV set any coefficient to exactly 0.0
# on this tiny 3-sample dataset?
# Your guess:

# --- Part 2: LassoCV on Real Data ---
# Purpose: See LassoCV pick alpha via cross-validation.
print("\n--- LassoCV on Real Data (5-fold CV) ---")
X, y = load_diabetes(return_X_y=True)

lassocv = linear_model.LassoCV(cv=5, random_state=42, max_iter=10000)
lassocv.fit(X, y)

print(f"Best alpha (CV): {lassocv.alpha_:.6f}")
print(f"Number of nonzero coefs: {np.sum(lassocv.coef_ != 0)} / {X.shape[1]}")
print(f"Coefficients: {lassocv.coef_}")
print(f"R^2 on training: {lassocv.score(X, y):.4f}")

# PREDICT BEFORE RUNNING: Will LassoCV keep all 10 features or zero some out?
# Your guess:

# --- Part 3: LassoLarsIC (AIC and BIC) ---
# Purpose: Compare CV-selected alpha vs AIC-selected vs BIC-selected.
print("\n--- LassoLarsIC: AIC vs BIC ---")

# AIC criterion
lassolars_aic = linear_model.LassoLarsIC(criterion="aic")
lassolars_aic.fit(X, y)
print(f"AIC best alpha:  {lassolars_aic.alpha_:.6f}")
print(f"AIC nonzero coefs: {np.sum(lassolars_aic.coef_ != 0)} / {X.shape[1]}")

# BIC criterion
lassolars_bic = linear_model.LassoLarsIC(criterion="bic")
lassolars_bic.fit(X, y)
print(f"BIC best alpha:  {lassolars_bic.alpha_:.6f}")
print(f"BIC nonzero coefs: {np.sum(lassolars_bic.coef_ != 0)} / {X.shape[1]}")

# PREDICT BEFORE RUNNING: Which will pick a LARGER alpha -> AIC or BIC?
# Hint: BIC penalizes complexity harder with log(n). More penalty = larger alpha.
# Your guess:

# --- Part 4: Manual AIC / BIC Calculation ---
# Purpose: Prove the math. Recompute AIC and BIC by hand for a few alphas
# and find the minimum. Compare with LassoLarsIC.
print("\n--- Manual AIC / BIC Calculation ---")
alphas = np.logspace(-4, 1, 40)
n = X.shape[0]

# Standardize features the way LassoLarsIC does internally
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

aic_manual = []
bic_manual = []
for a in alphas:
    lasso = linear_model.Lasso(alpha=a, fit_intercept=True, max_iter=10000)
    lasso.fit(X_std, y)
    residuals = y - lasso.predict(X_std)
    rss = np.sum(residuals**2)
    sigma2 = rss / n  # biased ML estimate, what LassoLarsIC uses
    # Max log-likelihood for Gaussian model (constants dropped):
    log_lik = -n / 2 * np.log(sigma2) - rss / (2 * sigma2)
    d = np.sum(lasso.coef_ != 0)  # effective degrees of freedom
    aic_manual.append(-2 * log_lik + 2 * d)
    bic_manual.append(-2 * log_lik + np.log(n) * d)

aic_manual = np.array(aic_manual)
bic_manual = np.array(bic_manual)

best_alpha_aic_manual = alphas[np.argmin(aic_manual)]
best_alpha_bic_manual = alphas[np.argmin(bic_manual)]

print(f"Manual AIC best alpha: {best_alpha_aic_manual:.6f}")
print(f"sklearn AIC best alpha: {lassolars_aic.alpha_:.6f}")
print(f"Manual BIC best alpha: {best_alpha_bic_manual:.6f}")
print(f"sklearn BIC best alpha: {lassolars_bic.alpha_:.6f}")
# These will be CLOSE but not exactly equal. Why? LassoLarsIC uses the LARS
# path (exact, piecewise-linear alphas), our manual grid is discrete.
# Same ballpark = success. Same order of magnitude is the goal here.

# PREDICT BEFORE RUNNING: Will BIC pick a larger alpha than AIC? (Repeat of
# Part 3's prediction, now that you've thought about it.)
# Your guess:

# --- Part 5: When Would You Actually Use Each? ---
# Purpose: Practical decision-making.
print("\n--- CV vs AIC vs BIC: Choosing a Method ---")
print("LassoCV:")
print("  + Robust, works with any n, any sample size, any model.")
print("  - Costs k+1 fits (k=5 means 6 total). Slow on huge data.")
print("LassoLarsIC (AIC):")
print("  + One pass along the LARS path. Very fast.")
print("  + Good when you care about prediction accuracy.")
print("  - Assumes correct model, large n. Can be badly wrong on small n.")
print("LassoLarsIC (BIC):")
print("  + Same speed as AIC. Stronger penalty -> sparser model.")
print("  + Good when you want feature selection (fewer features kept).")
print("  - Same assumptions as AIC.")

# Show whether BIC actually keeps fewer features (usually yes)
print(f"\nNumber of nonzero coefs:")
print(f"  CV:  {np.sum(lassocv.coef_ != 0)}")
print(f"  AIC: {np.sum(lassolars_aic.coef_ != 0)}")
print(f"  BIC: {np.sum(lassolars_bic.coef_ != 0)}")

# --- Part 6: Reflection ---
# Type your answers below.
# Q1: In one sentence, what is the difference between AIC and BIC?
# Your answer:
#
# Q2: Why is LassoLarsIC "one pass" but LassoCV is "k+1 passes"?
# Your answer:
#
# Q3: You have 10 million rows and 100 features. Which would you use and why?
# Your answer: