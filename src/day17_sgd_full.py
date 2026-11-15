# Day 17: Stochastic Gradient Descent (SGD) -- full section
# Covers scikit-learn User Guide §1.5
# (Day 11 was a preview; this is the complete treatment.)
#
# SGD minimises a sum of per-sample losses:
#     E(w) = (1/n) * sum_i L(y_i, f(x_i; w)) + alpha * R(w)
#
# At each step, it uses ONE sample (or a mini-batch) to estimate the
# gradient and takes a step:
#     w <- w - eta * grad_i E(w)
#
# Learning rate schedules (learning_rate= parameter):
#   'constant'    : eta = eta0
#   'optimal'     : eta = 1 / (alpha * (t + t0))   [theoretically optimal]
#   'invscaling'  : eta = eta0 / (t^power_t)
#   'adaptive'    : eta = eta0, shrinks when training loss plateaus
#
# Losses for SGDClassifier:
#   'hinge'            -> linear SVM
#   'log_loss'         -> logistic regression
#   'modified_huber'   -> smooth hinge, gives predict_proba
#   'squared_hinge'    -> linear SVM with squared hinge
#   'perceptron'       -> Perceptron
#   'squared_error'    -> ridge regression (classifier-style)
#   'huber'            -> robust regression
#   'epsilon_insensitive' -> linear SVR
#   'squared_epsilon_insensitive'
#
# Penalties: 'l2', 'l1', 'elasticnet' (needs l1_ratio).
#
# Early stopping: set early_stopping=True to hold out a fraction of data
# and stop when validation score stops improving.
#
# partial_fit: stream data in chunks. fit() does many epochs internally;
# partial_fit() does ONE pass over the chunk you hand it.
#
# NOTE: SGD is scale-sensitive. Always StandardScaler first.

from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import time

# =========================================================================
# PART A: Toy Example -- identical API to other classifiers
# =========================================================================
print("=" * 70)
print("PART A: Toy Example")
print("=" * 70)

X_toy = [[0, 0], [1, 1], [2, 2], [3, 3]]
y_toy = [0, 0, 1, 1]
clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_toy, y_toy)
print(f"Predictions: {clf.predict(X_toy)}")
print(f"Coefficients: {clf.coef_}")
print(f"Intercept: {clf.intercept_}")
print(f"n_iter_ (epochs run): {clf.n_iter_}")

# PREDICT BEFORE RUNNING: On this perfectly separable toy, will SGD get
# 100% accuracy? (Hint: hinge loss with default lr='optimal' converges
# fast on tiny separable data.)
# Your guess:

# =========================================================================
# PART B: Loss functions on breast cancer
# =========================================================================
print("\n" + "=" * 70)
print("PART B: Loss functions on real classification data")
print("=" * 70)
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

print(f"{'loss':<28} {'train_acc':>10} {'test_acc':>10}")
for loss in ["hinge", "log_loss", "modified_huber",
             "squared_hinge", "perceptron"]:
    clf = SGDClassifier(
        loss=loss, max_iter=2000, tol=1e-4, random_state=42
    )
    clf.fit(X_tr_s, y_tr)
    print(f"{loss:<28} {clf.score(X_tr_s, y_tr):>10.4f} "
          f"{clf.score(X_te_s, y_te):>10.4f}")

# PREDICT BEFORE RUNNING: Will 'hinge' (SVM) and 'log_loss' (logistic) give
# SIMILAR test accuracy? (Hint: they minimize different convex surrogates of
# the same 0/1 loss, so they often agree on clean data.)
# Your guess:

# =========================================================================
# PART C: Penalties on breast cancer
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Penalties (l2, l1, elasticnet)")
print("=" * 70)
print(f"{'penalty':<12} {'l1_ratio':>9} {'train_acc':>10} {'test_acc':>10} "
      f"{'nonzero':>8}")
for penalty, l1_ratio in [
    ("l2", None), ("l1", None), ("elasticnet", 0.15), ("elasticnet", 0.5),
]:
    kwargs = dict(
        penalty=penalty, loss="log_loss", max_iter=2000, tol=1e-4,
        random_state=42
    )
    if l1_ratio is not None:
        kwargs["l1_ratio"] = l1_ratio
    clf = SGDClassifier(**kwargs)
    clf.fit(X_tr_s, y_tr)
    nz = int(np.sum(np.abs(clf.coef_) > 1e-6))
    print(f"{penalty:<12} {str(l1_ratio):>9} {clf.score(X_tr_s, y_tr):>10.4f} "
          f"{clf.score(X_te_s, y_te):>10.4f} {nz:>8}")

# PREDICT BEFORE RUNNING: Which penalty will produce the FEWEST nonzero
# coefficients? (Hint: l1 = sparsity, l2 = shrink, elasticnet = in between.)
# Your guess:

# =========================================================================
# PART D: Learning rate schedules
# =========================================================================
print("\n" + "=" * 70)
print("PART D: Learning rate schedules")
print("=" * 70)
print(f"{'schedule':<14} {'train_acc':>10} {'test_acc':>10} {'n_iter':>8}")
for lr in ["constant", "optimal", "invscaling", "adaptive"]:
    clf = SGDClassifier(
        loss="log_loss", learning_rate=lr, eta0=0.01,
        max_iter=2000, tol=1e-4, random_state=42
    )
    clf.fit(X_tr_s, y_tr)
    print(f"{lr:<14} {clf.score(X_tr_s, y_tr):>10.4f} "
          f"{clf.score(X_te_s, y_te):>10.4f} {clf.n_iter_:>8}")

# PREDICT BEFORE RUNNING: Will 'adaptive' require FEWER or MORE epochs to
# converge? (Hint: it shrinks eta when loss plateaus, so it stops early.)
# Your guess:

# =========================================================================
# PART E: Early stopping (hold-out validation, no GridSearchCV)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: Early stopping")
print("=" * 70)
# With early_stopping=True, SGD holds out a fraction of training data
# (validation_fraction=0.1 by default) and stops when validation
# accuracy hasn't improved for n_iter_no_change epochs.
for es in [False, True]:
    clf = SGDClassifier(
        loss="log_loss", max_iter=3000, tol=1e-3,
        early_stopping=es, validation_fraction=0.1,
        n_iter_no_change=10, random_state=42
    )
    t0 = time.time()
    clf.fit(X_tr_s, y_tr)
    dt = time.time() - t0
    print(f"early_stopping={es!s:<5}: n_iter={clf.n_iter_:>3}, "
          f"test_acc={clf.score(X_te_s, y_te):.4f}, fit={dt*1000:.1f} ms")

# PREDICT BEFORE RUNNING: Will early_stopping=True REDUCE or INCREASE the
# number of epochs?
# Your guess:

# =========================================================================
# PART F: partial_fit -- out-of-core learning
# =========================================================================
print("\n" + "=" * 70)
print("PART F: partial_fit (streaming)")
print("=" * 70)
# Simulate chunks of data arriving over time. Model never sees all data
# at once; it updates incrementally. Useful when data > RAM.
clf = SGDClassifier(
    loss="log_loss", max_iter=1, tol=None, random_state=42
)
classes = np.array([0, 1])
n_chunks = 10
chunk_size = len(X_tr_s) // n_chunks
for i in range(n_chunks):
    start = i * chunk_size
    end = start + chunk_size
    clf.partial_fit(X_tr_s[start:end], y_tr[start:end], classes=classes)
print(f"After streaming {n_chunks} chunks: "
      f"test_acc = {clf.score(X_te_s, y_te):.4f}")

# PREDICT BEFORE RUNNING: Would you expect partial_fit over chunks to give
# the SAME result as fit() on the whole dataset?
# (Hint: partial_fit does one pass per chunk; fit does multiple epochs
# over the whole thing. Not identical.)
# Your guess:

# =========================================================================
# PART G: SGDRegressor losses
# =========================================================================
print("\n" + "=" * 70)
print("PART G: SGDRegressor losses")
print("=" * 70)
Xr, yr = load_diabetes(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)
scaler_r = StandardScaler()
Xr_tr_s = scaler_r.fit_transform(Xr_tr)
Xr_te_s = scaler_r.transform(Xr_te)

print(f"{'loss':<28} {'train_R2':>10} {'test_R2':>10}")
for loss in ["squared_error", "huber", "epsilon_insensitive"]:
    reg = SGDRegressor(
        loss=loss, max_iter=3000, tol=1e-4, random_state=42
    )
    reg.fit(Xr_tr_s, yr_tr)
    print(f"{loss:<28} {reg.score(Xr_tr_s, yr_tr):>10.4f} "
          f"{reg.score(Xr_te_s, yr_te):>10.4f}")

# PREDICT BEFORE RUNNING: Will 'squared_error' have the HIGHEST training R^2?
# (It directly minimizes squared error, which R^2 measures.)
# Your guess:

# =========================================================================
# PART H: GridSearchCV on SGD hyperparameters
# =========================================================================
print("\n" + "=" * 70)
print("PART H: GridSearchCV on C and learning rate")
print("=" * 70)
param_grid = {
    "alpha": np.logspace(-5, -2, 4),
    "learning_rate": ["optimal", "adaptive"],
    "eta0": [0.001, 0.01, 0.1],
}
grid = GridSearchCV(
    SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-4, random_state=42),
    param_grid, cv=3, scoring="accuracy", n_jobs=-1,
)
grid.fit(X_tr_s, y_tr)
print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")
print(f"Test accuracy with best: {grid.score(X_te_s, y_te):.4f}")

# PREDICT BEFORE RUNNING: Will the best CV accuracy beat every single
# baseline in Part B? (Hint: sometimes grid search is not needed on simple
# problems; SGD defaults are already strong.)
# Your guess:

# =========================================================================
# PART I: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART I: Reflection -- section 1.5")
print("=" * 70)

# Q1: Why is SGD preferred for very large datasets?
# Your answer:
#
# Q2: What is the difference between `fit` and `partial_fit`?
# Your answer:
#
# Q3: What does `early_stopping=True` do internally?
# Your answer:
#
# Q4: In one sentence, what does the `alpha` parameter control?
# Your answer:
#
# Q5: Why is feature scaling REQUIRED for SGD?
# Your answer: