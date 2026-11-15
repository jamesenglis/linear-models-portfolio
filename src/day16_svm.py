# Day 16: Support Vector Machines (SVM)
# Covers scikit-learn User Guide §1.4
#
# SVM idea: find the hyperplane that separates classes with the LARGEST
# MARGIN (distance to the nearest points). The nearest points are the
# SUPPORT VECTORS -- the model depends only on them, not all training data.
#
# When classes overlap, we allow some margin violations (slack zeta_i),
# penalized by C (inverse regularization: low C = smooth boundary, high C =
# tries to classify everything correctly).
#
# Kernel trick: map x -> phi(x) implicitly via K(x, z) = phi(x)^T phi(z).
#   'linear' -> K = x^T z (reduces to linear SVM)
#   'rbf'    -> K = exp(-gamma * ||x - z||^2) (most common)
#   'poly'   -> K = (gamma * x^T z + r)^d
#   'sigmoid'-> K = tanh(gamma * x^T z + r)
#
# Three main classes:
#   SVC      : kernel SVM classifier (libsvm). OVO multiclass internally.
#   LinearSVC: linear-only, faster, OVR multiclass, no support_vectors_ attr.
#   SVR      : kernel SVM regressor (epsilon-insensitive loss).
#
# Key practical: ALWAYS SCALE data. SVMs are not scale invariant [citation:4].
# Key tuning: RBF has TWO hyperparameters (C and gamma) that interact --
# use GridSearchCV on a log grid for both [citation:1][citation:10].

from sklearn import svm
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import time

# =========================================================================
# PART A: Toy SVC + support vectors
# =========================================================================
print("=" * 70)
print("PART A: Toy SVC and support vector attributes")
print("=" * 70)

X_toy = [[0, 0], [1, 1]]
y_toy = [0, 1]
clf = svm.SVC()
clf.fit(X_toy, y_toy)
print(f"Prediction for [[2, 2]]: {clf.predict([[2, 2]])}")
print(f"support_vectors_:\n{clf.support_vectors_}")
print(f"support_ (indices): {clf.support_}")
print(f"n_support_ (per class): {clf.n_support_}")

# PREDICT BEFORE RUNNING: On only 2 training points, will BOTH be support
# vectors? (Hint: with no margin slack, both sit on the boundary.)
# Your guess:

# =========================================================================
# PART B: Kernel comparison on Iris (2D for visibility)
# =========================================================================
print("\n" + "=" * 70)
print("PART B: Kernel comparison (Iris, first 2 features)")
print("=" * 70)

X, y = load_iris(return_X_y=True)
X_2d = X[:, :2]  # first 2 features for clean 2D comparison
X_tr, X_te, y_tr, y_te = train_test_split(
    X_2d, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

models = {
    "SVC linear": svm.SVC(kernel="linear", C=1.0),
    "LinearSVC": svm.LinearSVC(C=1.0, max_iter=10000),
    "SVC rbf": svm.SVC(kernel="rbf", C=1.0, gamma=0.7),
    "SVC poly": svm.SVC(kernel="poly", degree=3, gamma="auto", C=1.0),
}

print(f"{'model':<15} {'train_acc':>10} {'test_acc':>10}")
for name, model in models.items():
    model.fit(X_tr_s, y_tr)
    print(f"{name:<15} {model.score(X_tr_s, y_tr):>10.4f} "
          f"{model.score(X_te_s, y_te):>10.4f}")

# PREDICT BEFORE RUNNING: Which kernel will do BEST on this 2D slice of Iris?
# (Hint: Iris is nearly linearly separable in the first 2 features.)
# Your guess:

# =========================================================================
# PART C: C and gamma interaction (RBF kernel)
# =========================================================================
print("\n" + "=" * 70)
print("PART C: C and gamma -- the two RBF hyperparameters")
print("=" * 70)
# Use full Iris (4D) for a more realistic setting
X_tr4, X_te4, y_tr4, y_te4 = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler4 = StandardScaler()
X_tr4_s = scaler4.fit_transform(X_tr4)
X_te4_s = scaler4.transform(X_te4)

print(f"{'C':>6} {'gamma':>8} {'train_acc':>10} {'test_acc':>10}")
for C in [0.1, 1.0, 10.0, 100.0]:
    for gamma in [0.01, 0.1, 1.0, 10.0]:
        clf = svm.SVC(kernel="rbf", C=C, gamma=gamma)
        clf.fit(X_tr4_s, y_tr4)
        print(f"{C:>6} {gamma:>8} {clf.score(X_tr4_s, y_tr4):>10.4f} "
              f"{clf.score(X_te4_s, y_te4):>10.4f}")

# PREDICT BEFORE RUNNING: As gamma goes UP, does the model get MORE or LESS
# complex (fits training data more tightly)?
# Your guess:
# As C goes UP, does the model overfit or underfit more?
# Your guess:

# =========================================================================
# PART D: GridSearchCV for C and gamma (the recommended workflow)
# =========================================================================
print("\n" + "=" * 70)
print("PART D: GridSearchCV on C and gamma (log grid)")
print("=" * 70)
param_grid = {
    "C": np.logspace(-1, 2, 4),      # 0.1, 1, 10, 100
    "gamma": np.logspace(-2, 1, 4),  # 0.01, 0.1, 1, 10
}
grid = GridSearchCV(
    svm.SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid.fit(X_tr4_s, y_tr4)
print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")
print(f"Test accuracy with best: {grid.score(X_te4_s, y_te4):.4f}")
print("(Docs recommend GridSearchCV with C and gamma spaced exponentially [citation:1].)")

# PREDICT BEFORE RUNNING: Will the best CV accuracy be HIGHER or LOWER than
# the best test accuracy from Part C's single grid?
# Your guess:

# =========================================================================
# PART E: LinearSVC vs SVC(kernel='linear')
# =========================================================================
print("\n" + "=" * 70)
print("PART E: LinearSVC vs SVC(kernel='linear')")
print("=" * 70)
# LinearSVC uses liblinear (fast, OVR multiclass, squared hinge loss).
# SVC(kernel='linear') uses libsvm (OVO, regular hinge loss).
# LinearSVC does NOT have support_vectors_ attribute.
print(f"LinearSVC has support_vectors_? {hasattr(svm.LinearSVC(), 'support_vectors_')}")
print(f"SVC has support_vectors_? {hasattr(svm.SVC(), 'support_vectors_')}")

# Timing comparison
X_big, y_big = make_classification(
    n_samples=5000, n_features=20, n_informative=10, random_state=42
)
X_big_tr, X_big_te, y_big_tr, y_big_te = train_test_split(
    X_big, y_big, test_size=0.2, random_state=42
)
scaler_big = StandardScaler()
X_big_tr_s = scaler_big.fit_transform(X_big_tr)
X_big_te_s = scaler_big.transform(X_big_te)

for name, model in [
    ("LinearSVC", svm.LinearSVC(C=1.0, max_iter=10000)),
    ("SVC linear", svm.SVC(kernel="linear", C=1.0)),
]:
    t0 = time.time()
    model.fit(X_big_tr_s, y_big_tr)
    fit_time = time.time() - t0
    acc = model.score(X_big_te_s, y_big_te)
    print(f"{name:<12}: fit={fit_time*1000:>8.1f} ms, test_acc={acc:.4f}")

# PREDICT BEFORE RUNNING: Which will be FASTER on 5000 samples?
# (Hint: LinearSVC uses liblinear which scales almost linearly [citation:9].)
# Your guess:

# =========================================================================
# PART F: SVR (Support Vector Regression)
# =========================================================================
print("\n" + "=" * 70)
print("PART F: SVR (epsilon-insensitive loss)")
print("=" * 70)
# SVR ignores errors within epsilon of the target. Only points outside
# the "epsilon tube" contribute to the loss (they become support vectors).

X_reg, y_reg = make_regression(
    n_samples=500, n_features=5, noise=10.0, random_state=42
)
X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
scaler_r = StandardScaler()
X_r_tr_s = scaler_r.fit_transform(X_r_tr)
X_r_te_s = scaler_r.transform(X_r_te)

# SVR with RBF
svr = svm.SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
t0 = time.time()
svr.fit(X_r_tr_s, y_r_tr)
svr_fit = time.time() - t0
svr_mse = mean_squared_error(y_r_te, svr.predict(X_r_te_s))
print(f"SVR(RBF):    fit={svr_fit*1000:>7.1f} ms, test MSE={svr_mse:.4f}, "
      f"support vectors={np.sum(svr.support_)}/{len(y_r_tr)}")

# LinearSVR
lin_svr = svm.LinearSVR(C=100, epsilon=0.1, max_iter=10000)
t0 = time.time()
lin_svr.fit(X_r_tr_s, y_r_tr)
lin_svr_fit = time.time() - t0
lin_svr_mse = mean_squared_error(y_r_te, lin_svr.predict(X_r_te_s))
print(f"LinearSVR:   fit={lin_svr_fit*1000:>7.1f} ms, test MSE={lin_svr_mse:.4f}")

# Compare with KernelRidge from Day 15
from sklearn.kernel_ridge import KernelRidge
krr = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1)
krr.fit(X_r_tr_s, y_r_tr)
krr_mse = mean_squared_error(y_r_te, krr.predict(X_r_te_s))
print(f"KernelRidge:                    test MSE={krr_mse:.4f} (dense, no sparsity)")

# PREDICT BEFORE RUNNING: Will SVR use ALL training points as support
# vectors, or only a SUBSET? (Hint: epsilon-insensitive loss ignores points
# inside the tube.)
# Your guess:

# =========================================================================
# PART G: Custom kernel (Python function)
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Custom kernel as a Python function")
print("=" * 70)
def my_linear_kernel(X, Y):
    """Custom linear kernel: K(x, z) = x^T z."""
    return np.dot(X, Y.T)

clf_custom = svm.SVC(kernel=my_linear_kernel)
clf_custom.fit(X_tr_s, y_tr)
print(f"Custom kernel test accuracy: {clf_custom.score(X_te_s, y_te):.4f}")
print(f"support_vectors_ is empty? {len(clf_custom.support_vectors_) == 0}")
# With a custom function kernel, support_vectors_ is empty; only indices
# are stored in support_. Use kernel='precomputed' for a Gram matrix.

# PREDICT BEFORE RUNNING: Will the custom linear kernel give the SAME
# accuracy as svm.SVC(kernel='linear')?
# Your guess:

# =========================================================================
# PART H: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART H: Reflection -- section 1.4")
print("=" * 70)

# Q1: In one sentence, what is a "support vector"?
# Your answer:
#
# Q2: What do C and gamma each control in the RBF kernel?
# Your answer:
#
# Q3: Why must you always scale data before fitting an SVM?
# Your answer:
#
# Q4: What is the main practical difference between SVC(kernel='linear')
#     and LinearSVC?
# Your answer:
#
# Q5: In SVR, what does epsilon control?
# Your answer: