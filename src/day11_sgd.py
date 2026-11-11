# Day 11: Stochastic Gradient Descent (SGD) + Perceptron + Passive-Aggressive
# Covers scikit-learn User Guide §1.1.13
#
# SGD idea: instead of computing the full gradient over ALL n samples
# (expensive), take the gradient from ONE sample (or a mini-batch) at a time.
# This makes SGD the go-to for very large n. `partial_fit` lets you stream
# data in chunks (out-of-core learning).
#
# Update rule (per sample):
#     w <- w - eta * grad_i(loss)
# where eta is the learning rate. scikit-learn uses a schedule:
#     eta_t = eta0 / (1 + t * power_t)   (for learning_rate='invscaling')
# or 'constant', 'optimal', 'adaptive' depending on the setting.
#
# Loss functions:
#     loss='log'                  -> logistic regression (SGDClassifier)
#     loss='hinge'                -> linear SVM
#     loss='squared_error'        -> linear regression (SGDRegressor)
#     loss='huber'                -> robust regression
#     loss='epsilon_insensitive'  -> linear SVR
#
# Perceptron: SGDClassifier with loss='perceptron', no regularization,
#             no learning rate decay, updates ONLY on mistakes.
#             -> slightly faster than hinge SGD, sparser coefficients.
#
# Passive-Aggressive: SGDClassifier(loss='hinge', learning_rate='pa1' or 'pa2',
#             penalty=None, eta0=1.0). It does no update if the sample is
#             correctly classified with margin >= 1; otherwise it takes the
#             smallest step to fix the violation (that's the "aggressive" part).

from sklearn import linear_model
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# =========================================================================
# PART A: SGDClassifier (classification)
# =========================================================================
print("=" * 70)
print("PART A: SGDClassifier")
print("=" * 70)

# --- A1: Toy Example ---
print("\n--- A1: Toy Example ---")
X_toy = [[0, 0], [1, 1], [2, 2], [3, 3]]
y_toy = [0, 0, 1, 1]
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_toy, y_toy)
print(f"Predictions: {clf.predict(X_toy)}")
print(f"Coefficients: {clf.coef_}")
print(f"Intercept: {clf.intercept_}")

# PREDICT BEFORE RUNNING: On this perfectly separable toy, will SGD get
# 100% accuracy?
# Your guess:

# --- A2: Different losses on real data ---
# Purpose: See that SGD supports multiple loss functions in ONE class.
print("\n--- A2: Different losses (breast cancer) ---")
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"{'loss':<20} {'train_acc':>10} {'test_acc':>10}")
for loss in ["log_loss", "hinge", "modified_huber", "perceptron"]:
    clf = linear_model.SGDClassifier(
        loss=loss, max_iter=2000, tol=1e-4, random_state=42
    )
    clf.fit(X_train_s, y_train)
    print(f"{loss:<20} {clf.score(X_train_s, y_train):>10.4f} "
          f"{clf.score(X_test_s, y_test):>10.4f}")

# PREDICT BEFORE RUNNING: Will all losses give SIMILAR test accuracy?
# (They all optimize convex surrogates of the same 0/1 loss.)
# Your guess:

# --- A3: partial_fit for streaming / out-of-core learning ---
# Purpose: SGD's superpower -- train incrementally without all data in memory.
print("\n--- A3: partial_fit (streaming) ---")
clf = linear_model.SGDClassifier(
    loss="log_loss", max_iter=1, tol=None, random_state=42
)
classes = np.array([0, 1])
# Simulate 10 chunks of the training set arriving one by one
n_chunks = 10
chunk_size = len(X_train_s) // n_chunks
for i in range(n_chunks):
    start = i * chunk_size
    end = start + chunk_size
    X_chunk = X_train_s[start:end]
    y_chunk = y_train[start:end]
    clf.partial_fit(X_chunk, y_chunk, classes=classes)

print(f"After streaming {n_chunks} chunks: "
      f"test accuracy = {clf.score(X_test_s, y_test):.4f}")

# PREDICT BEFORE RUNNING: Would `clf.fit()` work the same as `partial_fit()`
# when the data fits in memory?
# Your guess:

# =========================================================================
# PART B: SGDRegressor (regression)
# =========================================================================
print("\n" + "=" * 70)
print("PART B: SGDRegressor")
print("=" * 70)

# --- B1: Toy example ---
print("\n--- B1: Toy Example ---")
reg = linear_model.SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

# --- B2: Different losses on diabetes ---
print("\n--- B2: Different losses (diabetes) ---")
X_d, y_d = load_diabetes(return_X_y=True)
X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)
sc_d = StandardScaler()
X_d_tr_s = sc_d.fit_transform(X_d_tr)
X_d_te_s = sc_d.transform(X_d_te)

print(f"{'loss':<22} {'train_R2':>10} {'test_R2':>10}")
for loss in ["squared_error", "huber", "epsilon_insensitive"]:
    reg = linear_model.SGDRegressor(
        loss=loss, max_iter=2000, tol=1e-4, random_state=42
    )
    reg.fit(X_d_tr_s, y_d_tr)
    print(f"{loss:<22} {reg.score(X_d_tr_s, y_d_tr):>10.4f} "
          f"{reg.score(X_d_te_s, y_d_te):>10.4f}")

# PREDICT BEFORE RUNNING: Will 'squared_error' give the HIGHEST training R^2?
# (Hint: it directly minimizes R^2's underlying loss.)
# Your guess:

# =========================================================================
# PART C: Perceptron (the simplest SGD variant)
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Perceptron")
print("=" * 70)

# --- C1: Perceptron toy ---
print("\n--- C1: Perceptron Toy ---")
clf_per = linear_model.Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf_per.fit(X_toy, y_toy)
print(f"Predictions: {clf_per.predict(X_toy)}")
print(f"Coefficients: {clf_per.coef_}")
print(f"Intercept: {clf_per.intercept_}")

# --- C2: Perceptron vs SGDClassifier(hinge) on breast cancer ---
print("\n--- C2: Perceptron vs SGD(hinge) ---")
per = linear_model.Perceptron(max_iter=2000, tol=1e-4, random_state=42)
per.fit(X_train_s, y_train)
print(f"Perceptron:  train={per.score(X_train_s, y_train):.4f}, "
      f"test={per.score(X_test_s, y_test):.4f}, "
      f"||w||={np.linalg.norm(per.coef_):.4f}")

sgd_hinge = linear_model.SGDClassifier(
    loss="hinge", max_iter=2000, tol=1e-4, random_state=42
)
sgd_hinge.fit(X_train_s, y_train)
print(f"SGD(hinge):  train={sgd_hinge.score(X_train_s, y_train):.4f}, "
      f"test={sgd_hinge.score(X_test_s, y_test):.4f}, "
      f"||w||={np.linalg.norm(sgd_hinge.coef_):.4f}")

# PREDICT BEFORE RUNNING: Will the Perceptron's coefficient vector be
# SHORTER (smaller norm) or LONGER than SGD(hinge)?
# Hint: Perceptron only updates on MISTAKES, so fewer updates -> shorter w.
# Your guess:

# =========================================================================
# PART D: Passive-Aggressive
# =========================================================================
print("\n" + "=" * 70)
print("PART D: Passive-Aggressive")
print("=" * 70)

print("\n--- D1: PA-I and PA-II on breast cancer ---")
for lr in ["pa1", "pa2"]:
    clf_pa = linear_model.SGDClassifier(
        loss="hinge", penalty=None, learning_rate=lr, eta0=1.0,
        max_iter=2000, tol=1e-4, random_state=42
    )
    clf_pa.fit(X_train_s, y_train)
    print(f"{lr.upper()}: train={clf_pa.score(X_train_s, y_train):.4f}, "
          f"test={clf_pa.score(X_test_s, y_test):.4f}, "
          f"||w||={np.linalg.norm(clf_pa.coef_):.4f}")

print("\nKey difference:")
print("  Perceptron: update on EVERY mistake, fixed step size.")
print("  Passive-Aggressive: NO update if margin >= 1 (passive).")
print("                       If violated, take the MINIMAL step to fix it")
print("                       (aggressive). PA-I uses L1, PA-II uses L2.")

# PREDICT BEFORE RUNNING: Will PA give HIGHER test accuracy than plain
# Perceptron? (Both are mistake-driven; PA's minimal-step correction
# tends to help when the data has noise.)
# Your guess:

# =========================================================================
# PART E: Learning rate schedules
# =========================================================================
print("\n" + "=" * 70)
print("PART E: Learning rate schedules")
print("=" * 70)
print(f"{'schedule':<20} {'train_acc':>10} {'test_acc':>10}")
for lr in ["constant", "optimal", "invscaling", "adaptive"]:
    clf = linear_model.SGDClassifier(
        loss="log_loss", learning_rate=lr, eta0=0.01,
        max_iter=2000, tol=1e-4, random_state=42
    )
    clf.fit(X_train_s, y_train)
    print(f"{lr:<20} {clf.score(X_train_s, y_train):>10.4f} "
          f"{clf.score(X_test_s, y_test):>10.4f}")

# PREDICT BEFORE RUNNING: Which schedule will likely be MOST robust?
# Hint: 'adaptive' reduces eta when training loss plateaus.
# Your guess:

# =========================================================================
# PART F: Manual SGD step (prove the update rule)
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Manual SGD update (one step)")
print("=" * 70)
# Do ONE manual SGD step with hinge loss and verify it matches the
# direction sklearn would take.
# Hinge loss: L_i(w) = max(0, 1 - y_i * (x_i . w))
# Gradient:   dL/dw = -y_i * x_i   if y_i * (x_i . w) < 1  (violation)
#                      0          otherwise
# y in {-1, +1} for the math below.

w_manual = np.zeros(X_train_s.shape[1])
x_sample = X_train_s[0]
y_sample = 1.0 if y_train[0] == 1 else -1.0
margin = y_sample * x_sample.dot(w_manual)  # = 0 since w=0
print(f"Initial margin: {margin:.4f} (violation since < 1)")

# Gradient step (hinge loss, L2 penalty with alpha = 1e-4 like sklearn)
alpha_reg = 1e-4
eta = 0.01
if margin < 1:
    grad = -y_sample * x_sample + alpha_reg * w_manual
    w_manual = w_manual - eta * grad
print(f"After one manual step, ||w|| = {np.linalg.norm(w_manual):.6f}")

# sklearn does the same thing internally. We won't match exactly (sklearn
# uses averaged coefficients, alpha regularization tuning, and shuffling),
# but the DIRECTION of the update is correct.
print("Key insight: the update is only -y*x when margin < 1.")
print("When margin >= 1, hinge has ZERO gradient -> no update.")

# =========================================================================
# PART G: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Reflection")
print("=" * 70)

# Q1: Why is SGD preferred for very large datasets?
# Your answer:
#
# Q2: What does `partial_fit` enable that `fit` does not?
# Your answer:
#
# Q3: In one sentence, what makes Perceptron different from SGD(hinge)?
# Your answer:
#
# Q4: What is the "passive" and what is the "aggressive" in Passive-Aggressive?
# Your answer: