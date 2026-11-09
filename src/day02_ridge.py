# Day 02: Ridge Regression
# Objective: min ||Xw - y||^2 + alpha * ||w||^2
# Math: The solution is w = (X^T X + alpha * I)^-1 X^T y
# Why it exists: OLS blows up when features are correlated (X^T X becomes
# near-singular). Ridge adds alpha to the diagonal, making it invertible.

from sklearn import linear_model
from sklearn.datasets import load_diabetes
import numpy as np

# --- Part 1: Toy Example (The Documentation Snippet) ---
# Purpose: Build muscle memory. Notice this is IDENTICAL to Day 1 API.
print("--- Toy Example (from sklearn docs) ---")
reg = linear_model.Ridge(alpha=0.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, 0.1, 1])

print(f"Coefficients (w): {reg.coef_}")
print(f"Intercept (w0): {reg.intercept_}")

# PREDICT BEFORE RUNNING: What happens if alpha=0?
# Your guess:
# Your guess:

# --- Part 2: Compare alpha values on Real Data ---
# Purpose: See how alpha "shrinks" coefficients toward zero.
print("\n--- Real Data: Effect of alpha ---")
X, y = load_diabetes(return_X_y=True)

for alpha in [0, 0.1, 1.0, 10.0, 100.0]:
    reg = linear_model.Ridge(alpha=alpha)
    reg.fit(X, y)
    # np.linalg.norm gives the "length" of the coefficient vector.
    coef_length = np.linalg.norm(reg.coef_)
    print(f"alpha={alpha:>6}: ||w|| = {coef_length:.4f}, R^2 = {reg.score(X, y):.4f}")

# PREDICT BEFORE RUNNING: As alpha goes UP, what happens to ||w||?
# Your guess:
# What happens to R^2 on the TRAINING set?
# Your guess:

# --- Part 3: Math Check (Manual Ridge Normal Equation) ---
# Purpose: Prove the math works. Same as OLS, but add alpha to the diagonal.
print("\n--- Manual Math Check (alpha = 1.0) ---")
alpha = 1.0

X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept column
I = np.eye(X_b.shape[1])
# IMPORTANT: Do not regularize the intercept. Standard practice sets I[0,0]=0.
I[0, 0] = 0

# Formula: theta = (X^T X + alpha * I)^-1 X^T y
theta_best = np.linalg.inv(X_b.T.dot(X_b) + alpha * I).dot(X_b.T).dot(y)

# Compare with sklearn
reg = linear_model.Ridge(alpha=alpha)
reg.fit(X, y)

print(f"Manual intercept:  {theta_best[0]:.4f}")
print(f"Sklearn intercept: {reg.intercept_:.4f}")
print(f"Manual coefs:  {theta_best[1:4]}")
print(f"Sklearn coefs: {reg.coef_[:3]}")
# These should match to ~4 decimal places.

# --- Part 4: Ridge vs OLS on Correlated Features ---
# Purpose: Demonstrate WHY ridge exists.
print("\n--- Why Ridge Exists (Correlated Features) ---")
# Create two features that are almost identical (highly correlated)
rng = np.random.RandomState(42)
x1 = rng.randn(100)
x2 = x1 + rng.randn(100) * 0.01  # x2 is basically x1 with tiny noise
X_corr = np.c_[x1, x2]
y_corr = 3 * x1 + 2 * x2 + rng.randn(100) * 0.1

ols = linear_model.LinearRegression().fit(X_corr, y_corr)
ridge = linear_model.Ridge(alpha=1.0).fit(X_corr, y_corr)

print(f"OLS coefficients:   {ols.coef_}")     # May be wild (e.g. [100, -95])
print(f"Ridge coefficients: {ridge.coef_}")   # Stable (e.g. [2.5, 2.5])
# The data cannot distinguish x1 from x2, so OLS picks extreme opposite values.
# Ridge splits the weight evenly. This is the whole point of Ridge.

# --- Part 5: Reflection ---
# Type your answers below.
# Q1: In your own words, what does alpha do?
# Your answer:
#
# Q2: Why do we set I[0,0] = 0 in the math check (don't penalize intercept)?
# Your answer:
#
# Q3: If alpha is huge, what does Ridge predict for every input?
# Your answer: