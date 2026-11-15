# Day 24: Multiclass and Multioutput
# Covers scikit-learn User Guide §1.12
#
# --- MULTICLASS ---
# Many binary classifiers (e.g. LinearSVC) can't natively do >2 classes.
# Meta-strategies wrap the binary model:
#   OneVsRestClassifier (OvR): K binary classifiers, one per class.
#     - Fast (K fits), but the "negative" set is huge when K is large.
#     - predict_proba: normalize the K "positive" probabilities.
#   OneVsOneClassifier (OvO): K*(K-1)/2 binary classifiers, one per PAIR.
#     - Slower for large K but each fit uses only 2 classes (less data).
#     - Voting breaks ties.
#   OutputCodeClassifier: Error-correcting-output-code. Each class is
#     assigned a unique binary CODE of length `code_size * n_classes`,
#     and we train `code_size * n_classes` binary classifiers. Predict
#     by finding the code closest to the classifier outputs.
#
# --- MULTIOUTPUT ---
# When Y has MULTIPLE columns (multiple targets to predict simultaneously).
#   MultiOutputClassifier: fits one classifier PER target (independent).
#   MultiOutputRegressor : fits one regressor PER target (independent).
#   ClassifierChain      : chains binary classifiers; each classifier gets
#                          the predictions of earlier ones as EXTRA features.
#                          Exploits label correlations.
#   RegressorChain       : same idea for regression.
#
# Key intuition:
#   Multiclass   = predict ONE label from K options.
#   Multioutput  = predict MULTIPLE labels at once (a vector of labels).
#   Multi-label  = each sample can have MULTIPLE labels; fits via binary
#                  relevance (essentially K independent binary classifiers).

from sklearn.multiclass import (
    OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier,
)
from sklearn.multioutput import (
    MultiOutputClassifier, MultiOutputRegressor,
    ClassifierChain, RegressorChain,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score
import numpy as np
import time

# =========================================================================
# PART A: OvR vs OvO vs native multiclass on iris
# =========================================================================
print("=" * 70)
print("PART A: OvR vs OvO vs native multiclass")
print("=" * 70)

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# LinearSVC is binary-only -> needs a wrapper
print("(LinearSVC needs OvR/OvO; LogisticRegression is native multiclass.)")

for name, clf in [
    ("LinearSVC + OvR", OneVsRestClassifier(LinearSVC(max_iter=10000))),
    ("LinearSVC + OvO", OneVsOneClassifier(LinearSVC(max_iter=10000))),
    ("LinearSVC + ECOC", OutputCodeClassifier(
        LinearSVC(max_iter=10000), code_size=2, random_state=42)),
    ("LogisticRegression (native)", LogisticRegression(max_iter=2000)),
]:
    t0 = time.time()
    clf.fit(X_tr_s, y_tr)
    dt = time.time() - t0
    print(f"{name:<32}: train={clf.score(X_tr_s, y_tr):.4f}, "
          f"test={clf.score(X_te_s, y_te):.4f}, fit={dt*1000:.1f} ms")

# PREDICT BEFORE RUNNING: Will OvR and OvO produce SIMILAR test accuracy?
# (Hint: both solve the same problem, but OvO trains on pairs, which can
# help with small data. On iris, differences are tiny.)
# Your guess:

# =========================================================================
# PART B: How many classifiers does each strategy train?
# =========================================================================
print("\n" + "=" * 70)
print("PART B: Number of binary fits per strategy")
print("=" * 70)
K = 3  # iris classes
print(f"K = {K} classes")
print(f"  OvR:  K = {K} binary classifiers (each class vs the rest)")
print(f"  OvO:  K*(K-1)/2 = {K*(K-1)//2} classifiers (one per pair)")
print(f"  ECOC: code_size * K = 2 * {K} = {2*K} binary classifiers")

# Access individual estimators
ovr = OneVsRestClassifier(LinearSVC(max_iter=10000)).fit(X_tr_s, y_tr)
print(f"\nOvR fitted {len(ovr.estimators_)} binary estimators")
ovo = OneVsOneClassifier(LinearSVC(max_iter=10000)).fit(X_tr_s, y_tr)
print(f"OvO fitted {len(ovo.estimators_)} binary estimators")
ecoc = OutputCodeClassifier(LinearSVC(max_iter=10000), code_size=2,
                             random_state=42).fit(X_tr_s, y_tr)
print(f"ECOC fitted {len(ecoc.estimators_)} binary estimators")

# PREDICT BEFORE RUNNING: For 10 classes, which strategy would need the most
# binary fits -- OvR, OvO, or ECOC with code_size=2? (Hint: OvR=10,
# OvO=45, ECOC=20.)
# Your guess:

# =========================================================================
# PART C: multi-label classification (MultiOutputClassifier)
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Multi-label (MultiOutputClassifier)")
print("=" * 70)
# Multi-label: each sample can have MULTIPLE labels simultaneously.
# Example: a news article can be about sports AND politics.
X_ml, y_ml = make_multilabel_classification(
    n_samples=500, n_features=20, n_classes=5, n_labels=2,
    allow_unlabeled=True, random_state=42
)
print(f"X shape: {X_ml.shape}")
print(f"y shape: {y_ml.shape}")  # (500, 5) -- 5 binary labels per sample

X_ml_tr, X_ml_te, y_ml_tr, y_ml_te = train_test_split(
    X_ml, y_ml, test_size=0.3, random_state=42
)

# MultiOutputClassifier: independent binary classifier per label
moc = MultiOutputClassifier(LogisticRegression(max_iter=2000))
moc.fit(X_ml_tr, y_ml_tr)
y_pred_moc = moc.predict(X_ml_te)
print(f"\nMultiOutputClassifier:")
print(f"  hamming_loss: {hamming_loss(y_ml_te, y_pred_moc):.4f} "
      f"(fraction of wrong labels)")
print(f"  jaccard_score (macro): "
      f"{jaccard_score(y_ml_te, y_pred_moc, average='macro'):.4f}")
print(f"  n estimators (one per label): {len(moc.estimators_)}")

# PREDICT BEFORE RUNNING: With 5 labels, how many LogisticRegression models
# did MultiOutputClassifier train? (Hint: one per label.)
# Your guess:

# =========================================================================
# PART D: ClassifierChain -- exploit label correlations
# =========================================================================
print("\n" + "=" * 70)
print("PART D: ClassifierChain")
print("=" * 70)
# ClassifierChain: classifier for label k uses labels 0..k-1 as features.
# If labels are correlated, the chain can exploit this -- sometimes better
# than independent MultiOutputClassifier.
chain = ClassifierChain(LogisticRegression(max_iter=2000), random_state=42)
chain.fit(X_ml_tr, y_ml_tr)
y_pred_chain = chain.predict(X_ml_te)
print(f"ClassifierChain:")
print(f"  hamming_loss: {hamming_loss(y_ml_te, y_pred_chain):.4f}")
print(f"  jaccard_score (macro): "
      f"{jaccard_score(y_ml_te, y_pred_chain, average='macro'):.4f}")

# Compare: MOC vs Chain
print(f"\nCompare:")
print(f"  MultiOutput jaccard: "
      f"{jaccard_score(y_ml_te, y_pred_moc, average='macro'):.4f}")
print(f"  Chain       jaccard: "
      f"{jaccard_score(y_ml_te, y_pred_chain, average='macro'):.4f}")
print("(ClassiferChain wins when labels are correlated. With random labels,")
print(" the chain can be WORSE due to error propagation.)")

# PREDICT BEFORE RUNNING: Will Chain beat MultiOutput on this synthetic
# multi-label data? (Hint: make_multilabel_classification labels are NOT
# explicitly correlated, so chain may not help and can be worse.)
# Your guess:

# =========================================================================
# PART E: MultiOutputRegressor
# =========================================================================
print("\n" + "=" * 70)
print("PART E: MultiOutputRegressor")
print("=" * 70)
# Multi-output regression: y has multiple continuous targets.
rng = np.random.RandomState(42)
X_mr = rng.randn(300, 5)
# Two targets, both depending on first 3 features, with different coefficients
y_mr = np.column_stack([
    X_mr[:, 0] + 2 * X_mr[:, 1] - X_mr[:, 2] + rng.randn(300) * 0.1,
    -2 * X_mr[:, 0] + X_mr[:, 3] + X_mr[:, 4] + rng.randn(300) * 0.1,
])
print(f"X_mr shape: {X_mr.shape}")
print(f"y_mr shape: {y_mr.shape}")  # (300, 2)

X_mr_tr, X_mr_te, y_mr_tr, y_mr_te = train_test_split(
    X_mr, y_mr, test_size=0.3, random_state=42
)

# MultiOutputRegressor: independent regression per target
mor = MultiOutputRegressor(LinearRegression())
mor.fit(X_mr_tr, y_mr_tr)
y_pred_mor = mor.predict(X_mr_te)
# Score: R^2 averaged across targets
print(f"MultiOutputRegressor:")
print(f"  R^2 per target: {mor.score(X_mr_te, y_mr_te, ):.4f}")
print(f"  n estimators: {len(mor.estimators_)}")
for i, est in enumerate(mor.estimators_):
    print(f"  target {i} coef: {np.round(est.coef_, 3)}")

# RegressorChain: sequential with previous predictions as features
rc = RegressorChain(LinearRegression(), random_state=42)
rc.fit(X_mr_tr, y_mr_tr)
y_pred_rc = rc.predict(X_mr_te)
print(f"\nRegressorChain R^2: {rc.score(X_mr_te, y_mr_te):.4f}")
print(f"  n estimators: {len(rc.estimators_)}")

# PREDICT BEFORE RUNNING: On INDEPENDENT targets (y1 and y2 here have
# distinct coefficient vectors), will Chain help or match MultiOutput?
# (Hint: with no correlation, chain adds noise -- but the effect on linear
# regression is often negligible.)
# Your guess:

# =========================================================================
# PART F: Manual OvR prediction (proves the math)
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Manual OvR prediction")
print("=" * 70)
# OvR: train K binary classifiers (class k vs rest). predict = argmax over
# the K "positive" probabilities.
ovr_manual = OneVsRestClassifier(LogisticRegression(max_iter=2000))
ovr_manual.fit(X_tr_s, y_tr)

# Take one test point; compute prob for each class manually
x_test = X_te_s[0]
y_true = y_te[0]
print(f"Test point (first 3 features): {np.round(x_test[:3], 3)}")
print(f"True class: {y_true}")
print(f"n_estimators (K classifiers): {len(ovr_manual.estimators_)}")

probas_manual = []
for k, est in enumerate(ovr_manual.estimators_):
    p = est.predict_proba([x_test])[0, 1]  # P(class k | x)
    probas_manual.append(p)
    print(f"  binary classifier {k}: P(class {k}) = {p:.4f}")

pred_manual = int(np.argmax(probas_manual))
print(f"Manual OvR prediction: {pred_manual}")
print(f"Sklearn OvR prediction: {ovr_manual.predict([x_test])[0]}")
print(f"Sklearn OvR proba:      {ovr_manual.predict_proba([x_test])[0]}")

# PREDICT BEFORE RUNNING: Will manual and sklearn OvR match EXACTLY?
# (Hint: yes, sklearn just normalizes the K probabilities and takes argmax.
# Small differences if you don't normalize the same way.)
# Your guess:

# =========================================================================
# PART G: Hamming loss -- the right metric for multi-label
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Multi-label evaluation metrics")
print("=" * 70)
# For multi-label, plain accuracy is misleading (any wrong label = wrong
# sample). Use hamming_loss or jaccard_score.
y_true_ml = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
y_pred_ml = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]])

print("y_true:", y_true_ml.tolist())
print("y_pred:", y_pred_ml.tolist())
print(f"hamming_loss:    {hamming_loss(y_true_ml, y_pred_ml):.4f}")
print(f"jaccard (micro): "
      f"{jaccard_score(y_true_ml, y_pred_ml, average='micro'):.4f}")
print(f"jaccard (macro): "
      f"{jaccard_score(y_true_ml, y_pred_ml, average='macro'):.4f}")
print(f"exact_match (subset accuracy): "
      f"{np.mean(np.all(y_true_ml == y_pred_ml, axis=1)):.4f}")

# PREDICT BEFORE RUNNING: For multi-label classification, what does
# exact_match_ratio mean? (Hint: fraction of SAMPLES where ALL labels
# are correct.)
# Your guess:

# =========================================================================
# PART H: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART H: Reflection -- section 1.12")
print("=" * 70)

# Q1: What is the difference between multiclass and multi-label?
# Your answer:
#
# Q2: When would you choose OvR over OvO? (Hint: think of K.)
# Your answer:
#
# Q3: What is the key difference between MultiOutputClassifier and
#     ClassifierChain?
# Your answer:
#
# Q4: Why is hamming_loss preferred over accuracy for multi-label?
# Your answer:
#
# Q5: When would you prefer ClassifierChain over MultiOutputClassifier?
# Your answer: