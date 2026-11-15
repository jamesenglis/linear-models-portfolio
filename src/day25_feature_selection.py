# Day 25: Feature Selection
# Covers scikit-learn User Guide §1.13
#
# Feature selection = pick a SUBSET of features. (Feature EXTRACTION
# transforms features instead -- e.g. PCA. Different problem.)
#
# Three families (from the docs):
#
# 1. FILTER methods (unsupervised or supervised, cheap):
#    - VarianceThreshold         : drop features with variance < threshold.
#    - SelectKBest + score_func  : rank features by a univariate score.
#      * f_classif (ANOVA F)     : classification
#      * chi2                    : non-negative features, classification
#      * f_regression            : regression
#      * mutual_info_classif     : captures NONLINEAR dependence
#      * mutual_info_regression  : same for regression
#
# 2. WRAPPER methods (expensive, use a model):
#    - RFE / RFECV              : recursively eliminate the least
#                                 important feature one at a time.
#    - SequentialFeatureSelector: forward or backward greedy selection
#                                 (adds/removes one feature per step).
#
# 3. EMBEDDED methods (model-driven, cheapest of the "smart" ones):
#    - SelectFromModel          : use `feature_importances_` or `coef_`
#                                 from a fitted model to drop weak features.
#    - L1-based models (Lasso, LogisticRegression with penalty='l1')
#      already perform feature selection -- use SelectFromModel on top.
#
# Rule of thumb (from docs):
#   - Try VarianceThreshold first (removes constants).
#   - Then a univariate filter (SelectKBest + mutual_info).
#   - Then a model-based method (SelectFromModel or RFE).
#   - Wrappers (SFS, RFE) if you have time and n_features is modest.
#
# Pipeline warning: ALWAYS put feature selection in a Pipeline when doing
# cross-validation. Otherwise you leak information from the test fold into
# the feature-selection step.

from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, SelectFromModel, RFE, RFECV,
    SequentialFeatureSelector, f_classif, chi2, mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, make_regression
import numpy as np
import time

# =========================================================================
# PART A: Baseline -- all features on breast cancer
# =========================================================================
print("=" * 70)
print("PART A: Baseline (all 30 features)")
print("=" * 70)

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

clf = LogisticRegression(max_iter=2000).fit(X_tr_s, y_tr)
print(f"All features: train_acc={clf.score(X_tr_s, y_tr):.4f}, "
      f"test_acc={clf.score(X_te_s, y_te):.4f}")

# PREDICT BEFORE RUNNING: With 30 features on breast cancer, do you think
# feature selection will HURT, HELP, or not change test accuracy?
# (Hint: most features are useful; removing some may reduce overfit.)
# Your guess:

# =========================================================================
# PART B: VarianceThreshold -- remove low-variance features
# =========================================================================
print("\n" + "=" * 70)
print("PART B: VarianceThreshold")
print("=" * 70)
# First scale, then VarianceThreshold. Without scaling, variance depends on
# units. With standardization, variance ~= 1 for non-constant features.
variances = X_tr_s.var(axis=0)
print(f"Feature variances after scaling:")
print(f"  min: {variances.min():.4f}, max: {variances.max():.4f}")
print(f"  # features with variance < 0.1: {(variances < 0.1).sum()}")

vt = VarianceThreshold(threshold=0.1)
X_tr_vt = vt.fit_transform(X_tr_s)
print(f"Kept {X_tr_vt.shape[1]} of {X_tr_s.shape[1]} features")

# PREDICT BEFORE RUNNING: After standard scaling, what is the variance of
# every NON-CONSTANT feature? (Hint: exactly 1.0, because StandardScaler
# divides by std.)
# Your guess:

# =========================================================================
# PART C: SelectKBest -- univariate filters
# =========================================================================
print("\n" + "=" * 70)
print("PART C: SelectKBest with different score functions")
print("=" * 70)
feature_names = load_breast_cancer().feature_names

for name, score_func in [
    ("f_classif (ANOVA F)", f_classif),
    ("mutual_info_classif", mutual_info_classif),
]:
    selector = SelectKBest(score_func=score_func, k=10)
    X_tr_sel = selector.fit_transform(X_tr_s, y_tr)
    X_te_sel = selector.transform(X_te_s)
    clf = LogisticRegression(max_iter=2000).fit(X_tr_sel, y_tr)
    chosen = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"\n{name}: kept 10 features")
    print(f"  train_acc={clf.score(X_tr_sel, y_tr):.4f}, "
          f"test_acc={clf.score(X_te_sel, y_te):.4f}")
    print(f"  selected: {chosen[:5]}...")

# PREDICT BEFORE RUNNING: Will f_classif and mutual_info choose the SAME
# top-10 features? (Hint: often overlap on 6-8 features, but differ on
# subtle nonlinear ones.)
# Your guess:

# =========================================================================
# PART D: SelectKBest -- k sweep
# =========================================================================
print("\n" + "=" * 70)
print("PART D: SelectKBest k sweep (f_classif)")
print("=" * 70)
print(f"{'k':>4} {'train_acc':>10} {'test_acc':>10}")
for k in [3, 5, 10, 15, 20, 25, 30]:
    selector = SelectKBest(score_func=f_classif, k=k)
    X_tr_sel = selector.fit_transform(X_tr_s, y_tr)
    X_te_sel = selector.transform(X_te_s)
    clf = LogisticRegression(max_iter=2000).fit(X_tr_sel, y_tr)
    print(f"{k:>4} {clf.score(X_tr_sel, y_tr):>10.4f} "
          f"{clf.score(X_te_sel, y_te):>10.4f}")

# PREDICT BEFORE RUNNING: Will test accuracy PEAK at some k < 30?
# (Hint: yes, often feature selection removes noise features.)
# Your guess:

# =========================================================================
# PART E: SelectFromModel -- use a model's feature importance
# =========================================================================
print("\n" + "=" * 70)
print("PART E: SelectFromModel")
print("=" * 70)
# Train a model, then use its coef_ or feature_importances_ to drop features.
# L1-regularized LogisticRegression naturally pushes weak coefs to zero.

# Using LogisticRegression with L1 penalty
l1 = LogisticRegression(
    penalty="l1", solver="saga", C=1.0, max_iter=5000
).fit(X_tr_s, y_tr)
n_nonzero = int(np.sum(np.abs(l1.coef_) > 1e-6))
print(f"L1 LogisticRegression: {n_nonzero} nonzero coefs / 30")

selector_l1 = SelectFromModel(l1, prefit=True)
X_tr_sel = selector_l1.transform(X_tr_s)
X_te_sel = selector_l1.transform(X_te_s)
print(f"SelectFromModel kept {X_tr_sel.shape[1]} features")

# RandomForest importance-based selection
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_tr_s, y_tr)
selector_rf = SelectFromModel(rf, prefit=True, threshold="mean")
X_tr_sel_rf = selector_rf.transform(X_tr_s)
X_te_sel_rf = selector_rf.transform(X_te_s)
print(f"SelectFromModel(RF, mean threshold) kept {X_tr_sel_rf.shape[1]} features")

# PREDICT BEFORE RUNNING: Will L1-based selection keep MORE or FEWER than
# 15 features with C=1.0? (Hint: depends on data; C=1.0 is a moderate
# penalty, so it may keep most features or drop several.)
# Your guess:

# =========================================================================
# PART F: RFE -- recursive feature elimination
# =========================================================================
print("\n" + "=" * 70)
print("PART F: RFE (recursive feature elimination)")
print("=" * 70)
# RFE fits the model, ranks features by importance, removes the least
# important, and repeats. Cost: ~ n_features fits.
for n_features_to_select in [5, 10, 15, 20]:
    rfe = RFE(
        estimator=LogisticRegression(max_iter=2000),
        n_features_to_select=n_features_to_select,
    )
    t0 = time.time()
    rfe.fit(X_tr_s, y_tr)
    dt = time.time() - t0
    X_tr_sel = rfe.transform(X_tr_s)
    X_te_sel = rfe.transform(X_te_s)
    clf = LogisticRegression(max_iter=2000).fit(X_tr_sel, y_tr)
    print(f"RFE to {n_features_to_select:>2}: test_acc={clf.score(X_te_sel, y_te):.4f}, "
          f"fit={dt*1000:.0f} ms")

# PREDICT BEFORE RUNNING: RFE fits LogisticRegression many times. Roughly
# how many fits for 30 features -> 5 features? (Hint: 30 - 5 + 1 = 26 fits,
# since it removes one per step.)
# Your guess:

# =========================================================================
# PART G: RFECV -- RFE with automatic k via cross-validation
# =========================================================================
print("\n" + "=" * 70)
print("PART G: RFECV (auto-select number of features)")
print("=" * 70)
# RFECV does RFE but uses CV to pick the optimal number of features.
rfecv = RFECV(
    estimator=LogisticRegression(max_iter=2000),
    step=1, cv=3, scoring="accuracy", min_features_to_select=1,
)
t0 = time.time()
rfecv.fit(X_tr_s, y_tr)
dt = time.time() - t0
print(f"RFECV best n_features: {rfecv.n_features_}")
print(f"RFECV cv score: {rfecv.cv_results_['mean_test_score'].max():.4f}")
print(f"Fit time: {dt:.2f} s")
print(f"Ranking (1 = selected): {rfecv.ranking_[:10]}...")

# PREDICT BEFORE RUNNING: Will RFECV pick MORE than 5 but FEWER than 30
# features? (Hint: yes, it balances accuracy against feature count.)
# Your guess:

# =========================================================================
# PART H: SequentialFeatureSelector
# =========================================================================
print("\n" + "=" * 70)
print("PART H: SequentialFeatureSelector (forward selection)")
print("=" * 70)
# SFS greedily adds (forward=True) or removes (forward=False) one feature
# per step, using cross-validation to score each candidate subset.
# WARNING: very slow (n_features * n_steps fits). Use with small feature sets.
sfs = SequentialFeatureSelector(
    LogisticRegression(max_iter=2000),
    n_features_to_select=10,
    direction="forward",
    cv=3, n_jobs=-1,
)
t0 = time.time()
sfs.fit(X_tr_s, y_tr)
dt = time.time() - t0
X_tr_sfs = sfs.transform(X_tr_s)
X_te_sfs = sfs.transform(X_te_s)
print(f"SFS picked {X_tr_sfs.shape[1]} features, fit={dt:.2f} s")
clf = LogisticRegression(max_iter=2000).fit(X_tr_sfs, y_tr)
print(f"SFS test_acc: {clf.score(X_te_sfs, y_te):.4f}")

# PREDICT BEFORE RUNNING: How many fits does SFS do to pick 10 from 30?
# (Rough estimate: ~10 * 20 = 200 fits, since each step tests ~20 candidates.)
# Your guess:

# =========================================================================
# PART I: Pipeline -- the CORRECT way to do feature selection in CV
# =========================================================================
print("\n" + "=" * 70)
print("PART I: Feature selection inside a Pipeline (no leakage)")
print("=" * 70)
# If you do feature selection on the FULL data BEFORE cross-validation,
# information from test folds leaks into the selection step. This makes
# CV scores optimistic. Always put selection inside the Pipeline.
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(f_classif, k=10)),
    ("clf", LogisticRegression(max_iter=2000)),
])

# Correct: fit selection inside each CV fold
scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"Pipeline CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

# WRONG (leaky): select on all data first, then CV. We don't do this; just
# explain why it's a problem.
print("(If you select on the full data BEFORE CV, CV scores will be")
print(" optimistically biased. Never do this.)")

# PREDICT BEFORE RUNNING: Will the pipeline CV score be LOWER or HIGHER
# than the test accuracy from Part A? (Hint: similar, because CV is honest.
# The leaky version would be HIGHER than honest CV.)
# Your guess:

# =========================================================================
# PART J: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART J: Reflection -- section 1.13")
print("=" * 70)

# Q1: In one sentence, what is the difference between feature SELECTION
#     and feature EXTRACTION?
# Your answer:
#
# Q2: When would you use SelectKBest vs RFE vs SelectFromModel?
# Your answer:
#
# Q3: What is the cost trade-off between SelectKBest and SequentialFeatureSelector?
# Your answer:
#
# Q4: Why must feature selection be inside a Pipeline for CV?
# Your answer:
#
# Q5: You have 1000 features and 200 samples. Which method would you try
#     FIRST? (Hint: use an L1 model + SelectFromModel -- it scales to high
#     dimensions. SelectKBest + mutual_info also scales.)
# Your answer: