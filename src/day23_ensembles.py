# Day 23: Ensembles
# Covers scikit-learn User Guide §1.11
#
# A single decision tree (Day 22) has LOW bias but HIGH variance: small
# changes in training data -> very different tree. Ensembles average many
# trees to reduce variance (bagging) or bias (boosting).
#
# --- BAGGING (parallel) ---
# RandomForestClassifier / ExtraTreesClassifier / BaggingClassifier
#   - Train many independent trees on BOOTSTRAP samples (with replacement).
#   - Each split considers only a RANDOM SUBSET of features
#     (max_features parameter). Reduces correlation between trees.
#   - Average predictions (or majority vote).
#   - ExtraTrees: splits are RANDOM thresholds, not optimized -> faster,
#     slightly more bias, less variance.
#
# --- BOOSTING (sequential) ---
# GradientBoostingClassifier / HistGradientBoostingClassifier / AdaBoost
#   - Fit trees SEQUENTIALLY. Each tree corrects the residuals (gradient)
#     of the previous ensemble.
#   - learning_rate controls how much each tree contributes.
#   - HistGradientBoosting is the modern version (LightGBM-style binning),
#     MUCH faster on large data, supports missing values natively.
#
# --- META-ESTIMATORS ---
# VotingClassifier     : combines multiple models' predictions (hard or soft).
# StackingClassifier   : trains a meta-model on the out-of-fold predictions
#                        of base models.
# BaggingClassifier    : bootstrap-aggregate any base model.
#
# Rule of thumb (docs):
#   - Small data: RandomForest / ExtraTrees.
#   - Medium data: HistGradientBoosting.
#   - Interpretability: shallow GB + learning_rate tuning.
#   - Always try RandomForest first as a baseline.
#
# Key hyperparameters (RandomForest):
#   n_estimators    : number of trees (more = better, with diminishing returns).
#   max_depth       : per-tree depth limit (None = full tree).
#   max_features    : features per split. Default 'sqrt' for classification.
#   min_samples_leaf: per-leaf minimum.
#   n_jobs=-1       : use all CPU cores (M1 has 8-10).
#   oob_score=True  : out-of-bag score, free validation set.

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import time

# =========================================================================
# PART A: Baseline single tree vs RandomForest on iris
# =========================================================================
print("=" * 70)
print("PART A: Single tree vs RandomForest")
print("=" * 70)

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Single tree (unpruned)
tree = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
print(f"Decision tree:    train={tree.score(X_tr, y_tr):.4f}, "
      f"test={tree.score(X_te, y_te):.4f}")

# RandomForest
rf = RandomForestClassifier(
    n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
).fit(X_tr, y_tr)
print(f"RandomForest(200):train={rf.score(X_tr, y_tr):.4f}, "
      f"test={rf.score(X_te, y_te):.4f}")

# OOB score: free validation (each sample evaluated on trees that did NOT
# see it during bagging).
rf_oob = RandomForestClassifier(
    n_estimators=200, oob_score=True, n_jobs=-1, random_state=42
).fit(X_tr, y_tr)
print(f"RandomForest OOB score: {rf_oob.oob_score_:.4f}")

# PREDICT BEFORE RUNNING: Will RandomForest beat the single tree on TEST
# accuracy? (Hint: usually yes, because averaging reduces variance.)
# Your guess:

# =========================================================================
# PART B: n_estimators sweep -- diminishing returns
# =========================================================================
print("\n" + "=" * 70)
print("PART B: n_estimators sweep")
print("=" * 70)
print(f"{'n_estimators':>14} {'train_acc':>10} {'test_acc':>10} "
      f"{'fit_time_ms':>12}")
for n_est in [10, 50, 100, 200, 500]:
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    dt = time.time() - t0
    print(f"{n_est:>14} {rf.score(X_tr, y_tr):>10.4f} "
          f"{rf.score(X_te, y_te):>10.4f} {dt*1000:>12.1f}")

# PREDICT BEFORE RUNNING: Will test accuracy keep rising past 200 trees?
# (Hint: diminishing returns, quickly.)
# Your guess:

# =========================================================================
# PART C: RandomForest vs ExtraTrees
# =========================================================================
print("\n" + "=" * 70)
print("PART C: RandomForest vs ExtraTrees")
print("=" * 70)
for name, cls in [
    ("RandomForest", RandomForestClassifier),
    ("ExtraTrees", ExtraTreesClassifier),
]:
    rf = cls(n_estimators=200, n_jobs=-1, random_state=42)
    t0 = time.time()
    rf.fit(X_tr, y_tr)
    dt = time.time() - t0
    print(f"{name:<14}: train={rf.score(X_tr, y_tr):.4f}, "
          f"test={rf.score(X_te, y_te):.4f}, fit={dt*1000:.1f} ms")

# PREDICT BEFORE RUNNING: Will ExtraTrees be FASTER or SLOWER than
# RandomForest to fit? (Hint: random thresholds mean fewer computations.)
# Your guess:

# =========================================================================
# PART D: Gradient Boosting family
# =========================================================================
print("\n" + "=" * 70)
print("PART D: Gradient Boosting vs HistGradientBoosting vs AdaBoost")
print("=" * 70)
# Use breast cancer (larger, more realistic)
Xb, yb = load_breast_cancer(return_X_y=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    Xb, yb, test_size=0.2, random_state=42, stratify=yb
)

print(f"{'model':<32} {'train_acc':>10} {'test_acc':>10} {'fit_ms':>8}")
for name, model in [
    ("GradientBoosting(n=100)", GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42)),
    ("HistGradientBoosting", HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, random_state=42)),
    ("AdaBoost(n=200)", AdaBoostClassifier(
        n_estimators=200, random_state=42)),
    ("RandomForest(n=200)", RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42)),
]:
    t0 = time.time()
    model.fit(Xb_tr, yb_tr)
    dt = time.time() - t0
    print(f"{name:<32} {model.score(Xb_tr, yb_tr):>10.4f} "
          f"{model.score(Xb_te, yb_te):>10.4f} {dt*1000:>8.1f}")

# PREDICT BEFORE RUNNING: Which will be FASTER to fit -- GradientBoosting or
# HistGradientBoosting? (Hint: Hist uses histogram binning, O(n) per
# iteration; classic GB is O(n * log(n)) per split.)
# Your guess:

# =========================================================================
# PART E: learning_rate and n_estimators interaction (GB)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: learning_rate x n_estimators (GradientBoosting)")
print("=" * 70)
print(f"{'lr':>6} {'n_est':>6} {'train_acc':>10} {'test_acc':>10}")
for lr in [0.01, 0.05, 0.1, 0.3]:
    for n_est in [100, 300]:
        gb = GradientBoostingClassifier(
            n_estimators=n_est, learning_rate=lr, max_depth=3,
            random_state=42
        )
        gb.fit(Xb_tr, yb_tr)
        print(f"{lr:>6} {n_est:>6} {gb.score(Xb_tr, yb_tr):>10.4f} "
              f"{gb.score(Xb_te, yb_te):>10.4f}")

# PREDICT BEFORE RUNNING: Which combo will win -- LOW lr + MANY trees, or
# HIGH lr + FEWER trees? (Docs: low lr + many trees = better generalization,
# but slower.)
# Your guess:

# =========================================================================
# PART F: VotingClassifier (soft vs hard voting)
# =========================================================================
print("\n" + "=" * 70)
print("PART F: VotingClassifier")
print("=" * 70)
# Hard voting: majority vote of predicted class labels.
# Soft voting:  average the probabilities, then take argmax.
# Soft usually wins because it uses more information.
scaler = StandardScaler()
Xb_tr_s = scaler.fit_transform(Xb_tr)
Xb_te_s = scaler.transform(Xb_te)

base_learners = [
    ("lr", LogisticRegression(max_iter=2000)),
    ("svc", SVC(kernel="rbf", C=1.0, gamma=0.1, probability=True, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
]

for name, voting in [("Hard voting", "hard"), ("Soft voting", "soft")]:
    vc = VotingClassifier(estimators=base_learners, voting=voting, n_jobs=-1)
    vc.fit(Xb_tr_s, yb_tr)
    print(f"{name:<14}: train={vc.score(Xb_tr_s, yb_tr):.4f}, "
          f"test={vc.score(Xb_te_s, yb_te):.4f}")

# Individual models for comparison
for name, model in base_learners:
    model.fit(Xb_tr_s, yb_tr)
    print(f"  {name:<12} alone: test={model.score(Xb_te_s, yb_te):.4f}")

# PREDICT BEFORE RUNNING: Will the VotingClassifier beat each individual
# model? (Hint: yes, usually, because it hedges across different biases.)
# Your guess:

# =========================================================================
# PART G: StackingClassifier
# =========================================================================
print("\n" + "=" * 70)
print("PART G: StackingClassifier")
print("=" * 70)
# Stacking trains a META-MODEL on the OUT-OF-FOLD predictions of base
# learners. It learns HOW to combine the base learners, not just average.
stack = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=2000)),
        ("svc", SVC(kernel="rbf", C=1.0, gamma=0.1, probability=True, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5, n_jobs=-1,
)
t0 = time.time()
stack.fit(Xb_tr_s, yb_tr)
dt = time.time() - t0
print(f"Stacking: train={stack.score(Xb_tr_s, yb_tr):.4f}, "
      f"test={stack.score(Xb_te_s, yb_te):.4f}, fit={dt*1000:.1f} ms")

# PREDICT BEFORE RUNNING: Will Stacking beat VotingClassifier? (Hint: often
# yes if base learners have DIFFERENT strengths, but slower to fit.)
# Your guess:

# =========================================================================
# PART H: Feature importance across ensemble models
# =========================================================================
print("\n" + "=" * 70)
print("PART H: Feature importance (RandomForest vs GradientBoosting)")
print("=" * 70)
feature_names = load_breast_cancer().feature_names

rf_imp = RandomForestClassifier(
    n_estimators=200, n_jobs=-1, random_state=42
).fit(Xb_tr, yb_tr)
gb_imp = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42
).fit(Xb_tr, yb_tr)

print(f"{'feature':<30} {'RF_imp':>10} {'GB_imp':>10}")
for i in np.argsort(rf_imp.feature_importances_)[::-1][:5]:
    print(f"{feature_names[i]:<30} {rf_imp.feature_importances_[i]:>10.4f} "
          f"{gb_imp.feature_importances_[i]:>10.4f}")
print(f"\nTop RF features: "
      f"{[feature_names[i] for i in np.argsort(rf_imp.feature_importances_)[::-1][:5]]}")

# PREDICT BEFORE RUNNING: Will RF and GB agree on the top-5 features?
# (Hint: they usually agree on 3 of 5 top features, because both use
# impurity-based importance.)
# Your guess:

# =========================================================================
# PART I: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART I: Reflection -- section 1.11")
print("=" * 70)

# Q1: In one sentence, what does bagging do? What does boosting do?
# Your answer:
#
# Q2: What is the difference between RandomForest and ExtraTrees?
# Your answer:
#
# Q3: Why is HistGradientBoosting preferred over GradientBoosting on large
#     datasets?
# Your answer:
#
# Q4: What does learning_rate control in boosting?
# Your answer:
#
# Q5: What is the difference between VotingClassifier and StackingClassifier?
# Your answer: