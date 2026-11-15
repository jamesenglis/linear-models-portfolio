# Day 26: Semi-Supervised Learning
# Covers scikit-learn User Guide §1.14
#
# Semi-supervised learning: SOME labels are missing. We use the unlabeled
# data to better capture the shape of the distribution.
#
# Label convention: unlabeled samples are marked with y = -1 [citation:1][citation:6].
#
# Three estimators:
#
# 1. SelfTrainingClassifier
#    - Wraps any classifier with predict_proba.
#    - Iteratively predicts pseudo-labels for unlabeled data, adds the most
#      confident ones to the labeled set, refits.
#    - Key params: threshold (confidence cutoff) or k_best (top-k per iter),
#      max_iter, criterion.
#    - IMPORTANT: base classifier must be well calibrated [citation:3][citation:11].
#
# 2. LabelPropagation
#    - Graph-based. Builds a similarity graph over ALL samples (labeled + unlabeled).
#    - Hard clamping (alpha=0): original labels are FIXED.
#    - Uses raw similarity matrix.
#
# 3. LabelSpreading
#    - Same graph idea, but SOFT clamping (alpha in [0,1]): labels can be
#      adjusted slightly, more robust to noise [citation:1][citation:5].
#    - Uses normalized graph Laplacian.
#
# Kernels for LabelPropagation / LabelSpreading:
#   'rbf' : dense fully-connected graph. Good for small/medium data.
#   'knn' : sparse graph. Better for larger data [citation:1].
#
# Mental model: use semi-supervised when labeled data is expensive but
# unlabeled data is plentiful, and the data has cluster/manifold structure.

from sklearn.semi_supervised import (
    SelfTrainingClassifier, LabelPropagation, LabelSpreading,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# =========================================================================
# PART A: Setup -- create semi-supervised data (mask some labels as -1)
# =========================================================================
print("=" * 70)
print("PART A: Setup -- mask labels")
print("=" * 70)

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize (important for graph-based methods and SVC)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Keep only a small fraction of labels; mark rest as -1
rng = np.random.RandomState(42)
n_labeled = 15  # out of ~105 training samples
mask = rng.rand(len(y_tr)) < (n_labeled / len(y_tr))
y_tr_semi = np.where(mask, y_tr, -1)

print(f"Total training samples: {len(y_tr)}")
print(f"Labeled (kept): {np.sum(y_tr_semi != -1)}")
print(f"Unlabeled (-1): {np.sum(y_tr_semi == -1)}")

# Baseline: supervised-only on labeled subset
mask_labeled = y_tr_semi != -1
rf_sup = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sup.fit(X_tr_s[mask_labeled], y_tr[mask_labeled])
print(f"\nSupervised baseline (only {n_labeled} labels): "
      f"test acc = {rf_sup.score(X_te_s, y_te):.4f}")

# PREDICT BEFORE RUNNING: With only 15 labeled samples on 3-class iris,
# will supervised accuracy be HIGH (>0.9), MEDIUM (0.7-0.9), or LOW (<0.7)?
# Your guess:

# =========================================================================
# PART B: SelfTrainingClassifier
# =========================================================================
print("\n" + "=" * 70)
print("PART B: SelfTrainingClassifier")
print("=" * 70)

# Wrap a RandomForest (has native predict_proba)
self_training = SelfTrainingClassifier(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold=0.75,
    criterion="threshold",
    max_iter=10,
    verbose=False,
)
self_training.fit(X_tr_s, y_tr_semi)
print(f"SelfTraining test acc: {self_training.score(X_te_s, y_te):.4f}")
print(f"Iterations: {self_training.n_iter_}")
print(f"Termination: {self_training.termination_condition_}")
print(f"Pseudo-labeled samples: {np.sum(self_training.labeled_iter_ > 0)}")

# Check which samples got pseudo-labeled
pseudo_labeled = self_training.labeled_iter_ > 0
print(f"\nPseudo-label accuracy (on originally unlabeled):")
pseudo_true = y_tr[pseudo_labeled]
pseudo_pred = self_training.transduction_[pseudo_labeled]
print(f"  {accuracy_score(pseudo_true, pseudo_pred):.4f} "
      f"({np.sum(pseudo_labeled)} samples)")

# PREDICT BEFORE RUNNING: Will SelfTraining BEAT the supervised baseline?
# (Hint: usually yes, because it adds confident pseudo-labels from
# unlabeled data.)
# Your guess:

# =========================================================================
# PART C: SelfTraining with calibrated SVC
# =========================================================================
print("\n" + "=" * 70)
print("PART C: SelfTraining with calibrated SVC")
print("=" * 70)
# SVC has predict_proba only when probability=True, but even then it may
# be poorly calibrated. Wrap in CalibratedClassifierCV [citation:10].
base_svc = CalibratedClassifierCV(
    SVC(kernel="rbf", gamma="auto", random_state=42), cv=3
)
self_training_svc = SelfTrainingClassifier(
    estimator=base_svc,
    threshold=0.75,
    max_iter=10,
)
self_training_svc.fit(X_tr_s, y_tr_semi)
print(f"SelfTraining(SVC) test acc: {self_training_svc.score(X_te_s, y_te):.4f}")
print(f"Iterations: {self_training_svc.n_iter_}")

# PREDICT BEFORE RUNNING: Will SVC-based SelfTraining beat the RF-based one?
# (Hint: SVC with RBF is strong on iris, but calibration adds overhead.
# Results are often similar.)
# Your guess:

# =========================================================================
# PART D: LabelPropagation vs LabelSpreading
# =========================================================================
print("\n" + "=" * 70)
print("PART D: LabelPropagation vs LabelSpreading")
print("=" * 70)

for name, model in [
    ("LabelPropagation (knn)", LabelPropagation(kernel="knn", n_neighbors=7)),
    ("LabelSpreading (knn)", LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2)),
    ("LabelPropagation (rbf)", LabelPropagation(kernel="rbf", gamma=20)),
    ("LabelSpreading (rbf)", LabelSpreading(kernel="rbf", gamma=20, alpha=0.2)),
]:
    model.fit(X_tr_s, y_tr_semi)
    test_acc = model.score(X_te_s, y_te)
    print(f"{name:<26}: test acc = {test_acc:.4f}, "
          f"n_iter = {model.n_iter_}")

# PREDICT BEFORE RUNNING: Will LabelSpreading BEAT LabelPropagation?
# (Hint: LabelSpreading is more robust to noise because of soft clamping.
# On clean iris, differences may be small.)
# Your guess:

# =========================================================================
# PART E: The alpha parameter in LabelSpreading (clamping)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: LabelSpreading alpha (clamping factor)")
print("=" * 70)
# alpha=0: hard clamping (like LabelPropagation)
# alpha=1: no clamping (labels can change completely)
print(f"{'alpha':>8} {'test_acc':>10}")
for alpha in [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]:
    ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=alpha)
    ls.fit(X_tr_s, y_tr_semi)
    print(f"{alpha:>8} {ls.score(X_te_s, y_te):>10.4f}")

# PREDICT BEFORE RUNNING: As alpha goes from 0 to 1, does accuracy go UP,
# DOWN, or peak in the middle? (Hint: alpha=0 = hard clamp = trusts labels;
# alpha=1 = no clamp = can change everything, including true labels.)
# Your guess:

# =========================================================================
# PART F: LabelSpreading on digits (harder, higher-dim)
# =========================================================================
print("\n" + "=" * 70)
print("PART F: LabelSpreading on digits (many classes)")
print("=" * 70)
X_d, y_d = load_digits(return_X_y=True)
X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(
    X_d, y_d, test_size=0.3, random_state=42, stratify=y_d
)
scaler_d = StandardScaler()
X_d_tr_s = scaler_d.fit_transform(X_d_tr)
X_d_te_s = scaler_d.transform(X_d_te)

# Keep only 30 labels
rng_d = np.random.RandomState(42)
n_labeled_d = 30
mask_d = rng_d.rand(len(y_d_tr)) < (n_labeled_d / len(y_d_tr))
y_d_tr_semi = np.where(mask_d, y_d_tr, -1)
print(f"Total: {len(y_d_tr)}, labeled: {np.sum(y_d_tr_semi != -1)}, "
      f"unlabeled: {np.sum(y_d_tr_semi == -1)}")

# Supervised baseline
rf_d = RandomForestClassifier(n_estimators=100, random_state=42)
rf_d.fit(X_d_tr_s[mask_d], y_d_tr[mask_d])
print(f"Supervised (30 labels): test acc = {rf_d.score(X_d_te_s, y_d_te):.4f}")

# LabelSpreading with knn kernel (sparse, faster)
ls_d = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2)
ls_d.fit(X_d_tr_s, y_d_tr_semi)
print(f"LabelSpreading (knn):  test acc = {ls_d.score(X_d_te_s, y_d_te):.4f}")

# LabelSpreading with rbf kernel (dense, slower)
ls_d_rbf = LabelSpreading(kernel="rbf", gamma=0.01, alpha=0.2)
ls_d_rbf.fit(X_d_tr_s, y_d_tr_semi)
print(f"LabelSpreading (rbf):  test acc = {ls_d_rbf.score(X_d_te_s, y_d_te):.4f}")

# PREDICT BEFORE RUNNING: On digits (10 classes, 64 features), will
# LabelSpreading beat the supervised baseline trained on only 30 labels?
# (Hint: this is exactly the semi-supervised sweet spot.)
# Your guess:

# =========================================================================
# PART G: Manual self-training logic (proves the idea)
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Manual self-training step")
print("=" * 70)
# Do ONE self-training iteration by hand on the iris semi data.
# Step 1: Train on labeled only.
base = RandomForestClassifier(n_estimators=100, random_state=42)
base.fit(X_tr_s[mask_labeled], y_tr[mask_labeled])

# Step 2: Predict on unlabeled.
unlabeled_mask = y_tr_semi == -1
X_unlabeled = X_tr_s[unlabeled_mask]
y_unlabeled_true = y_tr[unlabeled_mask]

# Step 3: Get probabilities, keep those above threshold.
probas = base.predict_proba(X_unlabeled)
max_probas = probas.max(axis=1)
confident_mask = max_probas >= 0.75
print(f"Unlabeled samples: {len(X_unlabeled)}")
print(f"Confident (threshold=0.75): {np.sum(confident_mask)}")

# Step 4: Add confident pseudo-labels to training set.
X_aug = np.vstack([X_tr_s[mask_labeled], X_unlabeled[confident_mask]])
y_aug = np.concatenate([
    y_tr[mask_labeled], base.predict(X_unlabeled[confident_mask])
])

# Step 5: Refit.
base_refit = RandomForestClassifier(n_estimators=100, random_state=42)
base_refit.fit(X_aug, y_aug)
print(f"After 1 manual iteration: test acc = {base_refit.score(X_te_s, y_te):.4f}")
print(f"Baseline (no pseudo-labels): {rf_sup.score(X_te_s, y_te):.4f}")

# PREDICT BEFORE RUNNING: Will the manual refit BEAT the baseline?
# (Hint: one iteration with confident pseudo-labels often helps, but
# less than the full SelfTrainingClassifier loop.)
# Your guess:

# =========================================================================
# PART H: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART H: Reflection -- section 1.14")
print("=" * 70)

# Q1: What does the label -1 mean in scikit-learn semi-supervised learning?
# Your answer:
#
# Q2: What is the key difference between SelfTraining and LabelPropagation?
# Your answer:
#
# Q3: What is the difference between LabelPropagation and LabelSpreading?
# (Hint: hard vs soft clamping.)
# Your answer:
#
# Q4: Why does SelfTrainingClassifier require a well-calibrated base classifier?
# Your answer:
#
# Q5: When would you choose 'knn' kernel over 'rbf' in LabelSpreading?
# Your answer: