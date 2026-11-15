# Day 28: Probability Calibration
# Covers scikit-learn User Guide §1.16
#
# A classifier is CALIBRATED if its predict_proba scores match reality:
# among all samples predicted at p=0.8, roughly 80% should be positive.
#
# Many classifiers are NOT calibrated:
#   - Naive Bayes: often over-confident (probabilities near 0 or 1).
#   - SVM (hinge loss): often under-confident (probabilities near 0.5).
#   - Random Forest: average of leaf votes, often under-confident.
#
# Two calibration methods:
#
# 1. Platt scaling (method='sigmoid'):
#    Fit a 1D logistic regression on the raw decision scores.
#    p_calibrated = 1 / (1 + exp(A * score + B))
#    - Few parameters (A, B). Works well with small calibration data.
#    - Assumes the miscalibration is sigmoid-shaped.
#
# 2. Isotonic regression (method='isotonic'):
#    Non-parametric monotone fit from score -> probability.
#    - More flexible. Needs MORE data (docs warn against <1000 samples).
#    - Can overfit on small calibration sets.
#    - We covered the math on Day 27.
#
# The docs' calibration example shows:
#   - LogisticRegression is often already well calibrated.
#   - GaussianNB (with redundant features) is over-confident.
#   - LinearSVC is under-confident.
#   - Isotonic fixes both when calibration data is plentiful.
#   - Sigmoid fixes the under-confident case but struggles with the
#     over-confident transposed-sigmoid case.
#
# Brier score: mean((p_hat - y)^2). Lower = better. Combines calibration
# and sharpness. Use it to compare calibrated vs uncalibrated.

from sklearn.calibration import CalibratedClassifierCV, calibration_curve, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# PART A: Toy Example -- calibrate a LinearSVC
# =========================================================================
print("=" * 70)
print("PART A: Toy Example -- LinearSVC calibration")
print("=" * 70)

X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=10, n_redundant=5,
    random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# LinearSVC has no predict_proba natively; CalibratedClassifierCV gives it.
base_svc = LinearSVC(max_iter=10000, random_state=42)
base_svc.fit(X_tr, y_tr)

# Sigmoid calibration
cal_sig = CalibratedClassifierCV(
    LinearSVC(max_iter=10000, random_state=42), method="sigmoid", cv=5
)
cal_sig.fit(X_tr, y_tr)

# Isotonic calibration
cal_iso = CalibratedClassifierCV(
    LinearSVC(max_iter=10000, random_state=42), method="isotonic", cv=5
)
cal_iso.fit(X_tr, y_tr)

print(f"LinearSVC accuracy (no proba): {base_svc.score(X_te, y_te):.4f}")
print(f"Sigmoid-calibrated accuracy:   {cal_sig.score(X_te, y_te):.4f}")
print(f"Isotonic-calibrated accuracy:  {cal_iso.score(X_te, y_te):.4f}")

# Brier scores
brier_sig = brier_score_loss(y_te, cal_sig.predict_proba(X_te)[:, 1])
brier_iso = brier_score_loss(y_te, cal_iso.predict_proba(X_te)[:, 1])
print(f"\nBrier score (sigmoid):   {brier_sig:.4f}")
print(f"Brier score (isotonic):  {brier_iso:.4f}")
print("(Lower Brier = better calibrated + sharper.)")

# PREDICT BEFORE RUNNING: Will accuracy change much after calibration?
# (Hint: calibration usually does not change ACCURACY, only the
# probabilities. Argmax is usually the same.)
# Your guess:

# =========================================================================
# PART B: Calibration curve (reliability diagram)
# =========================================================================
print("\n" + "=" * 70)
print("PART B: Calibration curve")
print("=" * 70)
# calibration_curve bins predictions and compares mean predicted vs
# observed frequency in each bin. Perfect calibration = diagonal line.
y_prob_sig = cal_sig.predict_proba(X_te)[:, 1]
prob_true, prob_pred = calibration_curve(y_te, y_prob_sig, n_bins=10)
print(f"{'mean_pred':>10} {'frac_pos':>10} {'gap':>10}")
for mp, fp in zip(prob_pred, prob_true):
    print(f"{mp:>10.4f} {fp:>10.4f} {abs(mp - fp):>10.4f}")
print("(Smaller gap = better calibrated.)")

# Plot and save
fig, ax = plt.subplots(figsize=(8, 6))
CalibrationDisplay.from_predictions(y_te, y_prob_sig, n_bins=10, ax=ax)
plt.savefig("data/day28_calibration_curve.png", dpi=100)
plt.close()
print("(Saved calibration curve -> data/day28_calibration_curve.png)")

# PREDICT BEFORE RUNNING: For a perfectly calibrated classifier, what
# should prob_true - prob_pred be for every bin? (Hint: 0. The points lie
# on the diagonal.)
# Your guess:

# =========================================================================
# PART C: Over-confident vs under-confident
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Over-confident (NB) vs under-confident (SVC)")
print("=" * 70)
# Use breast cancer (binary, clean)
Xb, yb = load_breast_cancer(return_X_y=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    Xb, yb, test_size=0.2, random_state=42, stratify=yb
)
scaler = StandardScaler()
Xb_tr_s = scaler.fit_transform(Xb_tr)
Xb_te_s = scaler.transform(Xb_te)

# GaussianNB (often over-confident)
gnb = GaussianNB().fit(Xb_tr_s, yb_tr)
p_gnb = gnb.predict_proba(Xb_te_s)[:, 1]

# SVC with sigmoid probability (often under-confident)
svc = SVC(kernel="rbf", gamma="auto", probability=True, random_state=42)
svc.fit(Xb_tr_s, yb_tr)
p_svc = svc.predict_proba(Xb_te_s)[:, 1]

# LogisticRegression (often well-calibrated)
lr = LogisticRegression(max_iter=2000).fit(Xb_tr_s, yb_tr)
p_lr = lr.predict_proba(Xb_te_s)[:, 1]

for name, p in [("GaussianNB", p_gnb), ("SVC (Platt)", p_svc),
                ("LogisticRegression", p_lr)]:
    brier = brier_score_loss(yb_te, p)
    print(f"{name:<20}: Brier = {brier:.4f}, "
          f"mean proba = {p.mean():.4f}, "
          f"frac actual pos = {yb_te.mean():.4f}")

# PREDICT BEFORE RUNNING: Which will have the LOWEST Brier score --
# GaussianNB, SVC, or LogisticRegression? (Hint: LR is often best on
# tabular data with mostly linear structure.)
# Your guess:

# =========================================================================
# PART D: Calibrating GaussianNB (fixing over-confidence)
# =========================================================================
print("\n" + "=" * 70)
print("PART D: Calibrating GaussianNB")
print("=" * 70)
# GN is over-confident: it pushes probabilities toward 0 and 1.
# Calibration should pull them back toward the diagonal.
cal_gnb_sig = CalibratedClassifierCV(
    GaussianNB(), method="sigmoid", cv=5
).fit(Xb_tr_s, yb_tr)
cal_gnb_iso = CalibratedClassifierCV(
    GaussianNB(), method="isotonic", cv=5
).fit(Xb_tr_s, yb_tr)

for name, p in [
    ("GNB raw", p_gnb),
    ("GNB sigmoid", cal_gnb_sig.predict_proba(Xb_te_s)[:, 1]),
    ("GNB isotonic", cal_gnb_iso.predict_proba(Xb_te_s)[:, 1]),
]:
    print(f"{name:<16}: Brier = {brier_score_loss(yb_te, p):.4f}, "
          f"min={p.min():.4f}, max={p.max():.4f}")

# PREDICT BEFORE RUNNING: Will calibration IMPROVE the Brier score for
# GaussianNB? (Hint: yes, if GN is over-confident, calibration should
# reduce the Brier score.)
# Your guess:

# =========================================================================
# PART E: Sigmoid vs isotonic -- when to use which
# =========================================================================
print("\n" + "=" * 70)
print("PART E: Sigmoid vs isotonic (data size matters)")
print("=" * 70)
# Docs: isotonic needs >1000 calibration samples or it overfits.
# Let's test on a SMALL calibration set.
X_small, y_small = make_classification(
    n_samples=200, n_features=10, n_informative=5, random_state=42
)
X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(
    X_small, y_small, test_size=0.5, random_state=42
)

for method in ["sigmoid", "isotonic"]:
    cal = CalibratedClassifierCV(
        GaussianNB(), method=method, cv=3
    ).fit(X_s_tr, y_s_tr)
    p = cal.predict_proba(X_s_te)[:, 1]
    print(f"{method:<10} (n_train=100): Brier = {brier_score_loss(y_s_te, p):.4f}")

# Now with a LARGER calibration set
X_large, y_large = make_classification(
    n_samples=5000, n_features=10, n_informative=5, random_state=42
)
X_l_tr, X_l_te, y_l_tr, y_l_te = train_test_split(
    X_large, y_large, test_size=0.2, random_state=42
)

for method in ["sigmoid", "isotonic"]:
    cal = CalibratedClassifierCV(
        GaussianNB(), method=method, cv=5
    ).fit(X_l_tr, y_l_tr)
    p = cal.predict_proba(X_l_te)[:, 1]
    print(f"{method:<10} (n_train=4000): Brier = {brier_score_loss(y_l_te, p):.4f}")

# PREDICT BEFORE RUNNING: On the SMALL data, will isotonic BEAT sigmoid?
# (Hint: docs warn isotonic overfits with <1000 samples. So sigmoid should
# win or tie on small data.)
# Your guess:

# =========================================================================
# PART F: Manual Platt scaling (proves the math)
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Manual Platt scaling")
print("=" * 70)
# Platt scaling fits a logistic regression to the DECISION SCORES.
# For SVC, decision_function gives the raw margin. We then fit:
#     P(y=1 | score) = 1 / (1 + exp(A * score + B))
# sklearn's CalibratedClassifierCV does this internally with cross-val.

from sklearn.linear_model import LogisticRegression as LR

# Get raw decision scores from LinearSVC
lsvc = LinearSVC(max_iter=10000, random_state=42).fit(X_tr, y_tr)
scores_tr = lsvc.decision_function(X_tr)
scores_te = lsvc.decision_function(X_te)

# Fit Platt: logistic regression on scores
platt = LR().fit(scores_tr.reshape(-1, 1), y_tr)
p_manual = platt.predict_proba(scores_te.reshape(-1, 1))[:, 1]

# Compare with CalibratedClassifierCV
cal_svc = CalibratedClassifierCV(
    LinearSVC(max_iter=10000, random_state=42), method="sigmoid", cv=5
).fit(X_tr, y_tr)
p_sklearn = cal_svc.predict_proba(X_te)[:, 1]

print(f"Manual Platt Brier:     {brier_score_loss(y_te, p_manual):.4f}")
print(f"Sklearn sigmoid Brier:  {brier_score_loss(y_te, p_sklearn):.4f}")
print(f"Max prob diff: {np.max(np.abs(p_manual - p_sklearn)):.4f}")
# Differences arise because sklearn uses cross-validation for calibration
# (avoids leakage), while our manual version fits on the same training set.

# PREDICT BEFORE RUNNING: Will the manual and sklearn probabilities match
# EXACTLY? (Hint: no -- sklearn uses cross-val to avoid overfitting the
# calibrator on the same data it was trained on.)
# Your guess:

# =========================================================================
# PART G: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Reflection -- section 1.16")
print("=" * 70)

# Q1: In one sentence, what does it mean for a classifier to be calibrated?
# Your answer:
#
# Q2: What is the difference between Platt scaling and isotonic regression?
# Your answer:
#
# Q3: When would you prefer sigmoid over isotonic? (Hint: data size.)
# Your answer:
#
# Q4: What does the Brier score measure? What are its components?
#     (Hint: calibration + sharpness.)
# Your answer:
#
# Q5: Why does CalibratedClassifierCV use cross-validation?
# Your answer: