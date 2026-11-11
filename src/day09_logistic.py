# Day 09: Logistic Regression + Solver Comparison
#
# Logistic regression is a LINEAR CLASSIFIER, despite the name "regression".
# It models P(y=1 | x) with the logistic (sigmoid) function:
#
#   p_hat(x) = expit(x w + w0) = 1 / (1 + exp(-x w - w0))
#
# Binary objective (cross-entropy / log loss + regularization):
#   min_w  (1/S) * sum_i s_i * [-y_i log p_hat_i - (1-y_i) log(1-p_hat_i)]
#          + r(w) / (S * C)
#   where C is the INVERSE regularization strength (C large = weak penalty),
#         S is the total sample weight,
#         r(w) is the penalty: L1, L2, ElasticNet, or none.
#
# Multiclass (multinomial) uses softmax instead of sigmoid.
#
# Solvers (all minimize the SAME objective; they differ in HOW):
#   lbfgs           : quasi-Newton, default, robust, no L1
#   liblinear       : coordinate descent, L1/L2, NO true multinomial (OvR)
#   newton-cg       : Newton with conjugate gradient, L2 only
#   newton-cholesky : exact Newton, L2 only, great when n_samples >> n_features
#   sag             : stochastic average gradient, fast on large data, L2 only
#   saga            : SAG variant that supports L1 + ElasticNet
#
# Key intuition: use lbfgs by default. Use saga for L1/ElasticNet. Use
# liblinear for tiny datasets or when you need OvR multiclass. Use
# newton-cholesky when n_samples >> n_features.

from sklearn import linear_model
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# --- Part 1: Toy Example (from sklearn docs) ---
# Purpose: Muscle memory. The API is fit/predict just like regressors.
print("--- Toy Example ---")
X_toy = [[0, 0], [1, 1], [2, 2], [3, 3]]
y_toy = [0, 0, 1, 1]

clf = linear_model.LogisticRegression()
clf.fit(X_toy, y_toy)
print(f"Coefficients: {clf.coef_}")
print(f"Intercept: {clf.intercept_}")
print(f"Predictions for [[1, 1], [2, 2]]: {clf.predict([[1, 1], [2, 2]])}")
# predict_proba gives the raw probability (this is the logistic output).
print(f"Probabilities: {clf.predict_proba([[1, 1], [2, 2]])}")

# PREDICT BEFORE RUNNING: For input [1, 1], which class will it predict?
# [2, 2]?
# Your guess:

# --- Part 2: Binary Classification on Real Data ---
# Purpose: Standard workflow with a real dataset.
print("\n--- Binary Classification (breast cancer) ---")
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Class balance (train): {np.bincount(y_train)}")

# IMPORTANT: scale features for solvers like lbfgs / newton-cg / sag.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = linear_model.LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)

print(f"Train accuracy: {clf.score(X_train_s, y_train):.4f}")
print(f"Test accuracy:  {clf.score(X_test_s, y_test):.4f}")
print(f"Coefficient shape: {clf.coef_.shape}")  # (1, n_features) for binary

# PREDICT BEFORE RUNNING: Will test accuracy be HIGHER or LOWER than train?
# Your guess:

# --- Part 3: The Solver Comparison Table (the heart of today) ---
# Purpose: Fit the SAME objective with 5 different solvers and compare
# accuracy + fit time. This is the table from the docs, made concrete.
print("\n--- Solver Comparison (same L2 objective, different solver) ---")
import time

solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
print(f"{'solver':<18} {'train_acc':>10} {'test_acc':>10} {'fit_time_ms':>13}")

for solver in solvers:
    clf = linear_model.LogisticRegression(
        solver=solver, max_iter=2000, random_state=42
    )
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    t1 = time.time()
    train_acc = clf.score(X_train_s, y_train)
    test_acc = clf.score(X_test_s, y_test)
    print(f"{solver:<18} {train_acc:>10.4f} {test_acc:>10.4f} "
          f"{(t1 - t0) * 1000:>13.1f}")

# PREDICT BEFORE RUNNING: Will all solvers give SIMILAR test accuracy?
# (They optimize the same convex objective to within tolerance.)
# Your guess:

# --- Part 4: Penalty x Solver Compatibility Matrix ---
# Purpose: See which solvers accept which penalties. This is the practical
# part of the docs table -- what works with what.
print("\n--- Penalty x Solver Compatibility ---")
print(f"{'penalty':<12} {'lbfgs':>8} {'liblinear':>10} {'newton-cg':>10} "
      f"{'newton-cholesky':>16} {'sag':>6} {'saga':>6}")

for penalty in ["l2", "l1", "elasticnet", "none"]:
    row = f"{penalty:<12}"
    for solver in ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]:
        try:
            # elasticnet needs an l1_ratio
            kwargs = {"penalty": penalty, "solver": solver, "max_iter": 500}
            if penalty == "elasticnet":
                kwargs["l1_ratio"] = 0.5
            clf = linear_model.LogisticRegression(**kwargs)
            clf.fit(X_train_s, y_train)
            row += f" {'yes':>8}" if solver != "newton-cholesky" else f" {'yes':>16}"
        except Exception:
            row += f" {'--':>8}" if solver != "newton-cholesky" else f" {'--':>16}"
    print(row)
# This mirrors the docs table. Yes/-- tells you which combos are legal.

# --- Part 5: Effect of C (inverse regularization) ---
# Purpose: See C shrink/extend the coefficients.
print("\n--- Effect of C (inverse regularization, L2) ---")
for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    clf = linear_model.LogisticRegression(C=C, max_iter=2000)
    clf.fit(X_train_s, y_train)
    coef_norm = np.linalg.norm(clf.coef_)
    train_acc = clf.score(X_train_s, y_train)
    test_acc = clf.score(X_test_s, y_test)
    print(f"C={C:>6}: ||w||={coef_norm:>8.4f}, train={train_acc:.4f}, test={test_acc:.4f}")

# PREDICT BEFORE RUNNING: As C goes UP, does ||w|| go UP or DOWN?
# Hint: C large = weak penalty = small regularization.
# Your guess:

# --- Part 6: Manual sigmoid + cross-entropy check ---
# Purpose: Prove the math. Compute the probability by hand using
#   p_hat = 1 / (1 + exp(-(X w + w0)))
# then cross-entropy, and match sklearn.predict_proba.
print("\n--- Manual Sigmoid + Cross-Entropy Check ---")
# Use the fitted model from Part 2
clf = linear_model.LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
# Take 3 test samples
idx = [0, 1, 2]
z = X_test_s[idx].dot(clf.coef_[0]) + clf.intercept_[0]
p_manual = 1.0 / (1.0 + np.exp(-z))

print(f"Manual probabilities:  {p_manual}")
print(f"Sklearn probabilities: {clf.predict_proba(X_test_s[idx])[:, 1]}")
# Should match to floating point tolerance.

# Manual log-loss for these 3 samples
y_true = y_test[idx]
eps = 1e-12  # avoid log(0)
p_clip = np.clip(p_manual, eps, 1 - eps)
ll_manual = -np.mean(
    y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip)
)
print(f"Manual log-loss (3 samples): {ll_manual:.6f}")
# sklearn's log_loss:
from sklearn.metrics import log_loss
ll_sklearn = log_loss(y_test[idx], p_manual)
print(f"sklearn log-loss (3 samples): {ll_sklearn:.6f}")

# --- Part 7: Multiclass (multinomial) on Digits ---
# Purpose: Show that multinomial logistic uses softmax instead of sigmoid,
# and that liblinear cannot do true multinomial (it uses One-vs-Rest).
print("\n--- Multiclass on Digits (0-9) ---")
X_d, y_d = load_digits(return_X_y=True)
X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42, stratify=y_d
)
sc_d = StandardScaler()
X_d_tr_s = sc_d.fit_transform(X_d_tr)
X_d_te_s = sc_d.transform(X_d_te)

for solver in ["lbfgs", "newton-cg", "sag", "saga"]:
    clf = linear_model.LogisticRegression(
        solver=solver, max_iter=2000, multi_class="multinomial"
    )
    clf.fit(X_d_tr_s, y_d_tr)
    test_acc = clf.score(X_d_te_s, y_d_te)
    print(f"{solver:<12} multinomial test acc: {test_acc:.4f}")

# liblinear only supports OvR (multiclass is handled as one-vs-rest).
clf_ovr = linear_model.LogisticRegression(solver="liblinear", max_iter=2000)
clf_ovr.fit(X_d_tr_s, y_d_tr)
print(f"liblinear    OvR test acc:        "
      f"{clf_ovr.score(X_d_te_s, y_d_te):.4f}")
print(f"Coefficient shape (multinomial lbfgs):")
clf_lbfgs = linear_model.LogisticRegression(solver="lbfgs", max_iter=2000)
clf_lbfgs.fit(X_d_tr_s, y_d_tr)
print(f"  {clf_lbfgs.coef_.shape}")   # (n_classes, n_features) = (10, 64)
# In multinomial, each class has its own coefficient vector.
# liblinear does OvR (one binary classifier per class); it does NOT
# share the softmax normalization across classes.

# PREDICT BEFORE RUNNING: Will multinomial scores be HIGHER, LOWER, or
# SIMILAR to OvR on digits? (Multinomial is usually a bit better.)
# Your guess:

# --- Part 8: Reflection ---
# Type your answers below.
# Q1: In one sentence, what is C in LogisticRegression?
# Your answer:
#
# Q2: Which solvers support L1? Which support ElasticNet?
# Your answer:
#
# Q3: Why can't liblinear learn a TRUE multinomial model?
#     (Answer: because it uses coordinate descent on a per-class binary
#      objective; there's no softmax coupling across classes in liblinear.)
# Your answer:
#
# Q4: You have n_samples=1,000,000 and n_features=100. Which solver? Why?
# Your answer: