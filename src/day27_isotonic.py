# Day 27: Isotonic Regression
# Covers scikit-learn User Guide §1.15
#
# Isotonic regression fits a NON-PARAMETRIC MONOTONIC function:
#     y_hat is non-decreasing in x (or non-increasing, via increasing=False)
#
# Objective (isotonic least squares):
#     minimize  sum_i w_i * (y_i - y_hat_i)^2
#     subject to  y_hat_i <= y_hat_j whenever x_i <= x_j
#
# This is a quadratic program with monotonicity constraints. Solved via
# the Pool Adjacent Violators Algorithm (PAVA):
#   1. Start with each y as its own "block".
#   2. Scan left to right. If two adjacent blocks violate monotonicity,
#      MERGE them and set the merged value to the weighted average.
#   3. Repeat until no violations remain.
#
# Why it matters:
#   - Calibration: map a classifier's raw score to a calibrated probability
#     (monotone by definition). This is how CalibratedClassifierCV with
#     method='isotonic' works.
#   - Dose-response: fit monotone relationships in medicine/economics.
#   - Ranking: convert ranks into monotone scores.
#
# Key hyperparameters:
#   - y_min, y_max: clip the output to a range.
#   - out_of_bounds: 'nan', 'clip', or 'raise' for x outside training range.
#   - increasing: True (default), False, or 'auto'.
#   - sample_weight: optional per-sample weights.
#
# Practical notes:
#   - The fitted function is piecewise constant (steps), not smooth.
#   - It interpolates linearly between fitted points at prediction time.
#   - Great when n is small/medium and the true relationship is monotone.
#   - Bad when the relationship is NOT monotone -- it will still output a
#     monotone fit, so it will underfit.

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score
import numpy as np

# =========================================================================
# PART A: Toy example -- direct API and step plot
# =========================================================================
print("=" * 70)
print("PART A: Toy Example")
print("=" * 70)

# Data: non-monotone, noisy relationship
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 3, 2, 4, 6, 5, 7, 9, 8, 10])

iso = IsotonicRegression()
y_hat = iso.fit_transform(x, y)
print(f"Raw y:   {y}")
print(f"Fitted:  {y_hat.astype(int)}")
# Notice: the fit forces y_hat to be non-decreasing.

# Predict at new points
x_new = np.array([2.5, 5.5, 9.5])
print(f"\nPredictions at {x_new}: {iso.predict(x_new)}")
print(f"Fitted coefficients X_thresholds_: {iso.X_thresholds_}")
print(f"Fitted coefficients y_thresholds_: {iso.y_thresholds_}")

# PREDICT BEFORE RUNNING: The original y is non-monotone (3 -> 2, 6 -> 5,
# 9 -> 8). Will the fitted y_hat be STRICTLY non-decreasing?
# (Hint: yes -- that's the constraint.)
# Your guess:

# =========================================================================
# PART B: PAVA manual check (proves the algorithm)
# =========================================================================
print("\n" + "=" * 70)
print("PART B: Manual PAVA step-by-step")
print("=" * 70)
# Implement a simple PAVA to see what sklearn does internally.

def pava(y):
    """Pool Adjacent Violators Algorithm."""
    blocks = [[v] for v in y]  # each value starts as its own block
    # Store (sum, count) for each block for efficient weighted averaging.
    sums = [v for v in y]
    counts = [1 for _ in y]

    i = 0
    while i < len(sums) - 1:
        if sums[i] / counts[i] <= sums[i + 1] / counts[i + 1]:
            i += 1
        else:
            # Merge blocks i and i+1
            sums[i] = sums[i] + sums[i + 1]
            counts[i] = counts[i] + counts[i + 1]
            sums.pop(i + 1)
            counts.pop(i + 1)
            # Step back to check the previous block for violations
            if i > 0:
                i -= 1
    # Expand back to per-sample values
    result = []
    for s, c in zip(sums, counts):
        result.extend([s / c] * c)
    return np.array(result)

y_pava = pava(y)
print(f"Raw y:       {y}")
print(f"PAVA fit:    {np.round(y_pava, 4)}")
print(f"Sklearn fit: {np.round(y_hat, 4)}")
print(f"Max diff: {np.max(np.abs(y_pava - y_hat)):.2e}")

# PREDICT BEFORE RUNNING: Will manual PAVA match sklearn EXACTLY?
# (Hint: yes, both implement the same algorithm; numerical differences
# only if sample_weight != 1.)
# Your guess:

# =========================================================================
# PART C: Isotonic regression for calibration
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Isotonic calibration of a classifier")
print("=" * 70)
# A classifier outputs SCORES, but these scores are not necessarily
# calibrated probabilities. Isotonic calibration learns a monotone map
# from score -> calibrated probability.
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Uncalibrated SVC (with probability=True gives Platt-scaled proba by default)
svc = SVC(kernel="rbf", gamma="auto", probability=True, random_state=42)
svc.fit(X_tr_s, y_tr)
print(f"SVC (Platt) accuracy: {svc.score(X_te_s, y_te):.4f}")

# Isotonic-calibrated SVC
svc_iso = CalibratedClassifierCV(
    SVC(kernel="rbf", gamma="auto", random_state=42),
    method="isotonic", cv=5,
)
svc_iso.fit(X_tr_s, y_tr)
print(f"SVC (isotonic) accuracy: {svc_iso.score(X_te_s, y_te):.4f}")

# Compare probability quality with Brier score
# Brier = mean((p_hat - y)^2). Lower = better calibrated.
p_svc = svc.predict_proba(X_te_s)[:, 1]
p_iso = svc_iso.predict_proba(X_te_s)[:, 1]
brier_svc = brier_score_loss(y_te, p_svc)
brier_iso = brier_score_loss(y_te, p_iso)
print(f"\nBrier score (lower = better calibration):")
print(f"  SVC(Platt):    {brier_svc:.4f}")
print(f"  SVC(isotonic): {brier_iso:.4f}")

# PREDICT BEFORE RUNNING: Will isotonic calibration give a LOWER Brier
# score than Platt scaling? (Hint: depends on data size; isotonic is more
# flexible but needs more data. With 455 training samples, Platt may win.)
# Your guess:

# =========================================================================
# PART D: out_of_bounds and y_min / y_max
# =========================================================================
print("\n" + "=" * 70)
print("PART D: Out-of-bounds and clipping")
print("=" * 70)
x_small = np.array([1, 2, 3, 4, 5])
y_small = np.array([1, 2, 4, 3, 5])

for oob in ["nan", "clip", "raise"]:
    iso = IsotonicRegression(out_of_bounds=oob)
    iso.fit(x_small, y_small)
    try:
        preds = iso.predict([0, 6])
        print(f"out_of_bounds={oob!r}: preds at [0, 6] = {preds}")
    except Exception as e:
        print(f"out_of_bounds={oob!r}: raised {type(e).__name__}")

# Clip output to a range
iso_clip = IsotonicRegression(y_min=1.5, y_max=3.5)
y_clipped = iso_clip.fit_transform(x_small, y_small)
print(f"\ny_min=1.5, y_max=3.5 clipped fit: {np.round(y_clipped, 2)}")

# PREDICT BEFORE RUNNING: With out_of_bounds='raise', will predict([0])
# raise or return nan? (Hint: raises.)
# Your guess:

# =========================================================================
# PART E: isotonic vs linear regression on monotone data
# =========================================================================
print("\n" + "=" * 70)
print("PART E: Isotonic vs Linear Regression")
print("=" * 70)
# Generate monotone but NONLINEAR data
rng = np.random.RandomState(42)
x_m = np.sort(rng.uniform(0, 10, 200))
y_m = np.log1p(x_m) + rng.randn(200) * 0.1  # monotone + noise
y_m = np.maximum(y_m, 0)  # clip to non-negative

from sklearn.linear_model import LinearRegression
lin = LinearRegression().fit(x_m[:, None], y_m)
y_lin = lin.predict(x_m[:, None])

iso = IsotonicRegression()
y_iso = iso.fit_transform(x_m, y_m)

mse_lin = np.mean((y_m - y_lin) ** 2)
mse_iso = np.mean((y_m - y_iso) ** 2)
print(f"Linear MSE:    {mse_lin:.4f}")
print(f"Isotonic MSE:  {mse_iso:.4f}")
print(f"Best of the two: {'linear' if mse_lin < mse_iso else 'isotonic'}")

# PREDICT BEFORE RUNNING: On log-shaped monotone data, which will fit
# better -- linear or isotonic? (Hint: isotonic, because it doesn't assume
# a linear shape.)
# Your guess:

# =========================================================================
# PART F: Increasing=False
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Decreasing fit (increasing=False)")
print("=" * 70)
x_desc = np.array([1, 2, 3, 4, 5])
y_desc = np.array([5, 4, 3, 2, 1])
y_inc = IsotonicRegression(increasing=True).fit_transform(x_desc, y_desc)
y_dec = IsotonicRegression(increasing=False).fit_transform(x_desc, y_desc)
print(f"Raw y:        {y_desc}")
print(f"increasing=True:  {y_inc.astype(int)}")
print(f"increasing=False: {y_dec.astype(int)}")

# PREDICT BEFORE RUNNING: With increasing=True on decreasing data, what
# will the fitted output look like? (Hint: isotonic constraint forces
# non-decreasing; on strictly decreasing data, the best fit is a constant
# at the mean value = 3.)
# Your guess:

# =========================================================================
# PART G: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Reflection -- section 1.15")
print("=" * 70)

# Q1: In one sentence, what does isotonic regression fit?
# Your answer:
#
# Q2: What is PAVA? What does it do when two adjacent blocks violate
#     monotonicity?
# Your answer:
#
# Q3: How is isotonic regression used for classifier calibration?
# Your answer:
#
# Q4: What does `out_of_bounds='clip'` do when you predict outside the
#     training range?
# Your answer:
#
# Q5: What is the main WEAKNESS of isotonic regression?
#     (Hint: non-monotone true relationships; overfits step noise with
#     small n.)
# Your answer: