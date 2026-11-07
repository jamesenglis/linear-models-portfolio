# Day 01: Ordinary Least Squares (OLS)
# Objective: min ||Xw - y||^2
# Math: The solution is w = (X^T X)^-1 X^T y

from sklearn import linear_model
from sklearn.datasets import load_diabetes
import numpy as np

# --- Part 1: Toy Example (The Documentation Snippet) ---
# Purpose: Type this to build muscle memory for the fit/predict API.
print("--- Toy Example ---")
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

# Check the attributes
print(f"Coefficients (w): {reg.coef_}")
print(f"Intercept (w0): {reg.intercept_}")

# Prediction
print(f"Prediction for [1, 1]: {reg.predict([[1, 1]])}")
print(f"Prediction for [3, 3]: {reg.predict([[3, 3]])}")

# --- Part 2: Real Data (Diabetes Dataset) ---
# Purpose: See how it behaves on real numbers.
print("\n--- Real Data ---")
X, y = load_diabetes(return_X_y=True)
print(f"Data shape: {X.shape}")

# Fit the model
reg_real = linear_model.LinearRegression()
reg_real.fit(X, y)

# Examine the coefficients
# Note: There are 10 features, so we get 10 coefficients.
print(f"Real Coefficients: {reg_real.coef_}")
print(f"Real Intercept: {reg_real.intercept_}")

# Score (R^2)
# 1.0 is perfect, 0.0 is random.
score = reg_real.score(X, y)
print(f"R^2 Score: {score:.4f}")

# --- Part 3: Math Check (Manual Normal Equation) ---
# Purpose: Prove the math works under the hood.
# Note: We add a column of ones to X to account for the intercept.
print("\n--- Manual Math Check ---")
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add x0 = 1
# Formula: theta = (X^T X)^-1 X^T y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(f"First 3 manual coefficients: {theta_best[1:4]}")
print(f"First 3 sklearn coefficients: {reg_real.coef_[:3]}")
# They should be extremely close (ignoring floating point differences).

# --- Part 4: Reflection ---
# Type out your answer to this question:
# Why does sklearn's intercept_ roughly match theta_best[0]?
# Your answer: because of how the bias (intercept) term is mathematically integrated into the feature matrix
