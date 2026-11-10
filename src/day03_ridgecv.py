# Day 03: RidgeCV - Cross-Validation to Choose alpha
# Objective: Instead of guessing alpha, let CV find the best one.
# Math: For each alpha in a grid, compute Leave-One-Out (LOO) error.
#       Pick the alpha with the lowest error.
# Why LOO is cheap for Ridge: There is a closed-form trick (hat matrix)
#       that makes LOO cost the SAME as a single fit. No need to refit k times.

from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
import numpy as np

# --- Part 1: Toy Example (from sklearn docs) ---
# Purpose: Muscle memory. Note: alphas is a LIST of candidates, not a single value.
print("--- Toy Example (from sklearn docs) ---")
reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0.1], [1, 1]], [0, -0.1, 1])

print(f"Best alpha chosen: {reg.alpha_}")
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

# PREDICT BEFORE RUNNING: Why is alpha_ a single number, not a list?
# Your guess:

# --- Part 2: RidgeCV on Real Data (Leave-One-Out) ---
# Purpose: See LOO pick an alpha automatically.
print("\n--- RidgeCV on Real Data (LOO default) ---")
X, y = load_diabetes(return_X_y=True)

alphas = np.logspace(-3, 3, 50)  # 50 candidates from 0.001 to 1000
regcv = linear_model.RidgeCV(alphas=alphas)
regcv.fit(X, y)

print(f"Best alpha (LOO): {regcv.alpha_:.4f}")
print(f"||w|| at best alpha: {np.linalg.norm(regcv.coef_):.4f}")
print(f"R^2 on training: {regcv.score(X, y):.4f}")

# PREDICT BEFORE RUNNING: Will alpha_ be small (near 0.001) or large (near 1000)?
# Hint: diabetes has 10 moderately-correlated features.
# Your guess:

# --- Part 3: Manual Verification (GridSearchCV vs RidgeCV) ---
# Purpose: Prove RidgeCV actually does what it says.
# We use train_test_split + GridSearchCV with k-fold as a "control experiment".
print("\n--- Manual Verification: GridSearchCV (5-fold) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {"alpha": np.logspace(-3, 3, 50)}
grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring="neg_mean_squared_error")
grid.fit(X_train, y_train)

print(f"Best alpha (5-fold CV): {grid.best_params_['alpha']:.4f}")
print(f"Test R^2: {grid.score(X_test, y_test):.4f}")

print(f"\nRidgeCV (LOO) picked: {regcv.alpha_:.4f}")
print(f"GridSearchCV (5-fold) picked: {grid.best_params_['alpha']:.4f}")
# They will be CLOSE but not identical. Why? LOO and 5-fold are different
# estimators of generalization error. Both are valid.

# PREDICT BEFORE RUNNING: Will they be exactly equal? Why or why not?
# Your guess:

# --- Part 4: Manual Math Check (LOO Formula) ---
# Purpose: Prove the LOO trick is real. Show that RidgeCV's choice matches
# the alpha that minimizes the closed-form LOO error.
print("\n--- Manual LOO Error Calculation ---")
# For Ridge, the LOO residual for sample i is:
#     r_i / (1 - h_ii)
# where r_i is the OLS-style residual and h_ii is the i-th diagonal of the
# hat matrix H = X (X^T X + alpha I)^-1 X^T.
X_b = np.c_[np.ones((X.shape[0], 1)), X]

loo_errors = []
for a in alphas:
    I = np.eye(X_b.shape[1])
    I[0, 0] = 0  # do not penalize intercept
    A = X_b.T.dot(X_b) + a * I
    A_inv = np.linalg.inv(A)
    w = A_inv.dot(X_b.T).dot(y)
    residuals = y - X_b.dot(w)
    h = np.einsum("ij,jk,ik->i", X_b, A_inv, X_b)  # diagonal of hat matrix
    loo_residuals = residuals / (1 - h)
    loo_errors.append(np.mean(loo_residuals**2))

best_alpha_manual = alphas[np.argmin(loo_errors)]
print(f"Manual LOO best alpha: {best_alpha_manual:.4f}")
print(f"RidgeCV best alpha:    {regcv.alpha_:.4f}")
# These should match (within floating point tolerance).

# --- Part 5: Reflection ---
# Type your answers below.
# Q1: In one sentence, what does RidgeCV do?
# Your answer:
#
# Q2: Why is LOO cheaper than k-fold for Ridge specifically?
# Your answer:
#
# Q3: If you had 1 million rows, which would you use and why?
#     RidgeCV (LOO) or GridSearchCV (k-fold)?
# Your answer:GridSearchCV (k-fold)
