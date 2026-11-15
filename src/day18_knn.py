# Day 18: Nearest Neighbors (KNN)
# Covers scikit-learn User Guide §1.6
#
# KNN is INSTANCE-BASED: no training (lazy learning). Prediction stores
# the training data and, for a new point, looks up the k nearest neighbors.
#
# Classification: majority vote among the k nearest labels.
# Regression:     average of the k nearest y values.
#
# The model has THREE moving parts:
#   1. n_neighbors (k) -- how many neighbors vote.
#   2. metric / p       -- distance measure (Euclidean p=2, Manhattan p=1).
#   3. algorithm        -- brute / kd_tree / ball_tree / auto.
#
# Algorithms:
#   brute    : exact, O(n * d) per query. Good for high d.
#   kd_tree  : exact, tree splits on one axis at a time. Good for low d
#              (rule of thumb: d < ~20). Fails when d is high (curse).
#   ball_tree: exact, balls instead of hyperrectangles. Slightly better
#              for moderate d.
#   auto     : pick the best of the above automatically.
#
# Practical tips (from docs):
#   - ALWAYS scale features. KNN is distance-based, so units matter.
#   - If n_neighbors=1, the classifier is a 1-NN (Voronoi), heavy overfit.
#   - For probability estimates, use predict_proba -- fraction of
#     neighbors in each class.
#   - In high d, KNN's accuracy collapses (curse of dimensionality).
#     Reduce dimensions with PCA first.
#
# Weighted variant: weights='distance' -> closer neighbors count more.

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import time

# =========================================================================
# PART A: Toy Example -- the manual distance computation
# =========================================================================
print("=" * 70)
print("PART A: Toy Example")
print("=" * 70)

X_toy = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
y_toy = [0, 0, 1, 1]
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_toy, y_toy)
print(f"Prediction for [0, 0, 0.1]: {clf.predict([[0, 0, 0.1]])}")
print(f"Probabilities: {clf.predict_proba([[0, 0, 0.1]])}")
print(f"Classes: {clf.classes_}")
print(f"kneighbors for [0, 0, 0.1]:")
dist, idx = clf.kneighbors([[0, 0, 0.1]], n_neighbors=3)
print(f"  distances: {dist[0]}")
print(f"  indices:   {idx[0]}")

# PREDICT BEFORE RUNNING: The new point [0, 0, 0.1] is closest to point 0
# ([0,0,0], label 0), then point 1 ([0,1,0], label 0) and point 3
# ([0,0,1], label 1). With k=3, will the vote be 2-1 for class 0, or
# split differently? (It will be 2-1 for class 0. Confirm with the output.)
# Your guess:

# =========================================================================
# PART B: KNN on Iris -- baseline, k sweep
# =========================================================================
print("\n" + "=" * 70)
print("PART B: k sweep on Iris")
print("=" * 70)
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

print(f"{'k':>3} {'train_acc':>10} {'test_acc':>10}")
for k in [1, 3, 5, 7, 9, 15, 25]:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_tr_s, y_tr)
    print(f"{k:>3} {clf.score(X_tr_s, y_tr):>10.4f} "
          f"{clf.score(X_te_s, y_te):>10.4f}")

# PREDICT BEFORE RUNNING: With k=1, will TRAIN accuracy be 100%?
# (Hint: every training point is its own nearest neighbor.)
# Your guess:
# As k grows beyond 15, will TEST accuracy continue to increase, plateau,
# or decrease? (Hint: underfitting.)
# Your guess:

# =========================================================================
# PART C: Weighted vs unweighted
# =========================================================================
print("\n" + "=" * 70)
print("PART C: weights='uniform' vs weights='distance'")
print("=" * 70)
for weights in ["uniform", "distance"]:
    clf = KNeighborsClassifier(n_neighbors=7, weights=weights)
    clf.fit(X_tr_s, y_tr)
    print(f"weights={weights:<10}: train_acc={clf.score(X_tr_s, y_tr):.4f}, "
          f"test_acc={clf.score(X_te_s, y_te):.4f}")

# PREDICT BEFORE RUNNING: Will 'distance' improve test accuracy on Iris?
# (Hint: it usually helps on small datasets where boundaries are tight.)
# Your guess:

# =========================================================================
# PART D: Distance metrics (Euclidean, Manhattan, Chebyshev)
# =========================================================================
print("\n" + "=" * 70)
print("PART D: Distance metrics")
print("=" * 70)
# p=2 Euclidean, p=1 Manhattan, p=inf Chebyshev.
# Minkowski(p) is the family; Euclidean/Manhattan are p=2/p=1 special cases.
print(f"{'metric':<14} {'train_acc':>10} {'test_acc':>10}")
for metric_name, kwargs in [
    ("euclidean (p=2)", {"p": 2}),
    ("manhattan (p=1)", {"p": 1}),
    ("chebyshev (p=inf)", {"p": np.inf}),
    ("cosine", {"metric": "cosine"}),
]:
    clf = KNeighborsClassifier(n_neighbors=7, **kwargs)
    clf.fit(X_tr_s, y_tr)
    print(f"{metric_name:<14} {clf.score(X_tr_s, y_tr):>10.4f} "
          f"{clf.score(X_te_s, y_te):>10.4f}")

# PREDICT BEFORE RUNNING: Which will do BEST on Iris?
# (Hint: Iris features are physical measurements -- Euclidean is a
# reasonable fit. Cosine similarity ignores magnitude, which is less
# appropriate here.)
# Your guess:

# =========================================================================
# PART E: Algorithm comparison (brute vs kd_tree vs ball_tree)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: Algorithm comparison")
print("=" * 70)
# Use a larger dataset to make algorithm differences visible.
rng = np.random.RandomState(42)
X_big = rng.randn(20000, 10)        # 20k samples, 10 features (moderate d)
y_big = (X_big[:, 0] > 0).astype(int)

# Query on a subset
X_query = X_big[:500]

for algo in ["brute", "kd_tree", "ball_tree", "auto"]:
    clf = NearestNeighbors(n_neighbors=5, algorithm=algo)
    clf.fit(X_big)
    t0 = time.time()
    dist, idx = clf.kneighbors(X_query)
    dt = time.time() - t0
    print(f"{algo:<10}: query_time={dt*1000:>7.1f} ms")
    print(f"  first query distances: {np.round(dist[0], 4)}")
# All three algorithms should produce IDENTICAL distances (they are exact
# algorithms, not approximations).

# PREDICT BEFORE RUNNING: Will all four algorithms return EXACTLY the same
# neighbor distances for the first query? (Hint: yes -- they are exact
# but differ only in the internal search strategy.)
# Your guess:

# =========================================================================
# PART F: High-dimensional curse -- KD-tree collapses
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Curse of dimensionality (algorithm timings)")
print("=" * 70)
print(f"{'d':>4} {'brute_ms':>10} {'kd_tree_ms':>12} {'ball_tree_ms':>14}")
for d in [2, 10, 50, 200]:
    rng = np.random.RandomState(0)
    X_hd = rng.randn(2000, d)
    X_q = X_hd[:200]
    times = {}
    for algo in ["brute", "kd_tree", "ball_tree"]:
        try:
            clf = NearestNeighbors(
                n_neighbors=5, algorithm=algo
            )
            clf.fit(X_hd)
            t0 = time.time()
            clf.kneighbors(X_q)
            times[algo] = (time.time() - t0) * 1000
        except Exception:
            times[algo] = float("nan")
    print(f"{d:>4} {times['brute']:>10.1f} {times['kd_tree']:>12.1f} "
          f"{times['ball_tree']:>14.1f}")
# In low d, kd_tree dominates. As d grows, brute force wins because kd_tree
# examines almost every ball. Docs recommend: use kd_tree only when d < ~20.

# PREDICT BEFORE RUNNING: At d=200, will brute be FASTER or SLOWER than
# kd_tree? (Hint: the tree degenerates and visits almost every point.)
# Your guess:

# =========================================================================
# PART G: Manual 1-NN computation (verifies the model)
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Manual 1-NN distance computation")
print("=" * 70)
# Take one test point, compute distances to all training points, pick the
# nearest, and confirm it matches sklearn's prediction.
x_query = X_te_s[0]
dists = np.sqrt(np.sum((X_tr_s - x_query) ** 2, axis=1))
nearest_idx = np.argmin(dists)
nearest_label = y_tr[nearest_idx]
print(f"Manual nearest training index: {nearest_idx}")
print(f"Manual nearest distance:       {dists[nearest_idx]:.4f}")
print(f"Manual label:                  {nearest_label}")

clf_1nn = KNeighborsClassifier(n_neighbors=1)
clf_1nn.fit(X_tr_s, y_tr)
sklearn_pred = clf_1nn.predict([x_query])[0]
sklearn_dist, sklearn_idx = clf_1nn.kneighbors([x_query], n_neighbors=1)
print(f"Sklearn prediction:            {sklearn_pred}")
print(f"Sklearn nearest index:         {sklearn_idx[0][0]}")
print(f"Sklearn nearest distance:      {sklearn_dist[0][0]:.4f}")

# PREDICT BEFORE RUNNING: Will manual and sklearn match EXACTLY?
# Your guess:

# =========================================================================
# PART H: KNN Regression on diabetes
# =========================================================================
print("\n" + "=" * 70)
print("PART H: KNN Regression")
print("=" * 70)
Xr, yr = load_diabetes(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)
scaler_r = StandardScaler()
Xr_tr_s = scaler_r.fit_transform(Xr_tr)
Xr_te_s = scaler_r.transform(Xr_te)

print(f"{'k':>3} {'train_R2':>10} {'test_R2':>10} {'test_MSE':>10}")
for k in [3, 5, 10, 20]:
    reg = KNeighborsRegressor(n_neighbors=k)
    reg.fit(Xr_tr_s, yr_tr)
    pred = reg.predict(Xr_te_s)
    mse = mean_squared_error(yr_te, pred)
    print(f"{k:>3} {reg.score(Xr_tr_s, yr_tr):>10.4f} "
          f"{reg.score(Xr_te_s, yr_te):>10.4f} {mse:>10.2f}")

# PREDICT BEFORE RUNNING: Will KNN regressor beat Ridge from Day 2?
# (Hint: diabetes has 10 features; KNN is generally weaker than a well-tuned
# linear model on tabular data with mostly linear relationships.)
# Your guess:

# =========================================================================
# PART I: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART I: Reflection -- section 1.6")
print("=" * 70)

# Q1: What does "instance-based" mean for KNN?
# Your answer:
#
# Q2: Why does k=1 give 100% training accuracy?
# Your answer:
#
# Q3: Why does kd_tree lose to brute force in high dimensions?
# Your answer:
#
# Q4: Why must you always scale features for KNN?
# Your answer:
#
# Q5: On diabetes, did KNN beat Ridge? Why do you think that is?
# Your answer: