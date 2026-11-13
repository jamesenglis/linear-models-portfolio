# Day 14: Linear Discriminant Analysis (LDA) + Quadratic Discriminant Analysis (QDA)
# Covers scikit-learn User Guide §1.2
#
# Both LDA and QDA are PROBABILISTIC classifiers derived from Bayes' rule.
# They model P(X | y=k) as a multivariate Gaussian for each class k.
# Predictions: pick the class k that maximizes the posterior P(y=k | X).
#
# Math (Bayes' rule):
#     P(y=k | x) = P(x | y=k) * P(y=k) / sum_l P(x | y=l) * P(y=l)
#
# Gaussian density per class:
#     P(x | y=k) = (1 / ((2*pi)^(d/2) * |Sigma_k|^(1/2)))
#                   * exp(-0.5 * (x - mu_k)^T * Sigma_k^-1 * (x - mu_k))
#
# Log posterior (constants dropped):
#     log P(y=k | x) = -0.5 * log|Sigma_k|
#                      -0.5 * (x - mu_k)^T * Sigma_k^-1 * (x - mu_k)
#                      + log P(y=k)
#
# LDA:  Sigma_k = Sigma for ALL classes (shared covariance).
#       -> linear decision boundary.
# QDA:  Sigma_k is different for EACH class (per-class covariance).
#       -> quadratic decision boundary (more flexible, more parameters).
#
# The term (x - mu_k)^T * Sigma^-1 * (x - mu_k) is the MAHALANOBIS DISTANCE.
# LDA assigns x to the class with the closest Mahalanobis distance,
# adjusted by the class prior P(y=k).

from sklearn import discriminant_analysis
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# =========================================================================
# PART A: Toy Example (from sklearn docs)
# =========================================================================
print("=" * 70)
print("PART A: Toy Example")
print("=" * 70)

X_toy = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y_toy = np.array([1, 1, 1, 2, 2, 2])

lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X_toy, y_toy)
print(f"LDA predictions on toy: {lda.predict(X_toy)}")
print(f"LDA coefficients: {lda.coef_}")
print(f"LDA intercept: {lda.intercept_}")
print(f"LDA class means: {lda.means_}")

# PREDICT BEFORE RUNNING: On this perfectly separable toy, will LDA get
# 100% accuracy?
# Your guess:

# =========================================================================
# PART B: LDA vs QDA on Iris (real data)
# =========================================================================
print("\n" + "=" * 70)
print("PART B: LDA vs QDA on Iris")
print("=" * 70)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# LDA
lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print(f"LDA train accuracy: {lda.score(X_train, y_train):.4f}")
print(f"LDA test accuracy:  {lda.score(X_test, y_test):.4f}")
print(f"LDA coef shape: {lda.coef_.shape}")  # (n_classes, n_features)

# QDA
qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
print(f"\nQDA train accuracy: {qda.score(X_train, y_train):.4f}")
print(f"QDA test accuracy:  {qda.score(X_test, y_test):.4f}")

# PREDICT BEFORE RUNNING: Iris has 3 classes and 4 features. Which will
# perform BETTER on test -- LDA or QDA? (Hint: Iris classes have different
# variances, but small sample size favors simpler models.)
# Your guess:

# =========================================================================
# PART C: Manual LDA computation (prove the math)
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Manual LDA calculation")
print("=" * 70)
# LDA: 1) compute class means, 2) shared covariance, 3) linear coefficients.
# w_k = Sigma^-1 * mu_k
# b_k = -0.5 * mu_k^T * Sigma^-1 * mu_k + log(prior_k)

# Use only 2 classes for clean math
mask = y_train < 2  # classes 0 and 1
X_bin = X_train[mask]
y_bin = y_train[mask]

classes = np.unique(y_bin)
n_classes = len(classes)
n_features = X_bin.shape[1]

# Step 1: Class means
means = np.array([X_bin[y_bin == k].mean(axis=0) for k in classes])
print(f"Manual class means:\n{means}")

# Step 2: Shared (pooled) covariance
# Weighted by class priors
priors = np.array([np.mean(y_bin == k) for k in classes])
print(f"Manual priors: {priors}")

cov = np.zeros((n_features, n_features))
for i, k in enumerate(classes):
    X_k = X_bin[y_bin == k]
    cov += priors[i] * np.cov(X_k, rowvar=False, bias=True)
print(f"Manual pooled covariance:\n{np.round(cov, 4)}")

# Step 3: Linear coefficients
Sigma_inv = np.linalg.inv(cov)
w_manual = Sigma_inv.dot(means.T).T  # shape (n_classes, n_features)
b_manual = np.array([
    -0.5 * means[i].dot(Sigma_inv).dot(means[i]) + np.log(priors[i])
    for i in range(n_classes)
])
print(f"Manual LDA coef:\n{np.round(w_manual, 4)}")
print(f"Manual LDA intercept: {np.round(b_manual, 4)}")

# Compare with sklearn (fit on the same binary subset)
lda_bin = discriminant_analysis.LinearDiscriminantAnalysis()
lda_bin.fit(X_bin, y_bin)
print(f"\nSklearn LDA coef:\n{np.round(lda_bin.coef_, 4)}")
print(f"Sklearn LDA intercept: {np.round(lda_bin.intercept_, 4)}")
# Should match closely (sklearn may use slightly different covariance
# normalization; differences should be small).

# PREDICT BEFORE RUNNING: Will the manual coefficients match sklearn's
# EXACTLY or just be CLOSE? (Hint: covariance estimator details differ.)
# Your guess:

# =========================================================================
# PART D: LDA as dimensionality reduction (transform)
# =========================================================================
print("\n" + "=" * 70)
print("PART D: LDA transform (supervised dimensionality reduction)")
print("=" * 70)
# LDA can project data to at most (n_classes - 1) dimensions.
# For 3 classes, max 2 components.
lda_3 = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
X_proj = lda_3.fit_transform(X_train, y_train)
print(f"Original shape: {X_train.shape}")
print(f"After LDA transform: {X_proj.shape}")  # (n_samples, 2)
print(f"Explained variance ratio: {lda_3.explained_variance_ratio_}")
print(f"Sum: {lda_3.explained_variance_ratio_.sum():.4f}")

# PREDICT BEFORE RUNNING: With 3 classes, can n_components be 3?
# (Hint: dimension is < n_classes.)
# Your guess:

# =========================================================================
# PART E: QDA manual check (per-class covariance)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: Manual QDA computation (2 classes)")
print("=" * 70)
# QDA: each class has its OWN covariance matrix.
# Decision: log|Sigma_k| + (x - mu_k)^T Sigma_k^-1 (x - mu_k) - 2*log(prior_k)

qda_bin = discriminant_analysis.QuadraticDiscriminantAnalysis()
qda_bin.fit(X_bin, y_bin)
print(f"QDA train accuracy: {qda_bin.score(X_bin, y_bin):.4f}")

# Manual per-class covariance
covs_manual = []
for k in classes:
    X_k = X_bin[y_bin == k]
    covs_manual.append(np.cov(X_k, rowvar=False, bias=True))
    print(f"Class {k} covariance:\n{np.round(covs_manual[-1], 4)}")

# Take 3 test samples and compute the decision score by hand
X_test_bin = X_test[y_test < 2]
y_test_bin = y_test[y_test < 2]
print(f"\nManual QDA on first 3 test samples:")
for i in range(min(3, len(X_test_bin))):
    x = X_test_bin[i]
    scores = []
    for j, k in enumerate(classes):
        cov_k = covs_manual[j]
        mu_k = means[j]
        diff = x - mu_k
        # log|Sigma| + Mahalanobis - 2*log(prior)
        sign, logdet = np.linalg.slogdet(cov_k)
        mahal = diff.dot(np.linalg.inv(cov_k)).dot(diff)
        score = logdet + mahal - 2 * np.log(priors[j])
        scores.append(score)
    pred_manual = classes[np.argmin(scores)]  # lower score = better
    pred_sklearn = qda_bin.predict([x])[0]
    print(f"  sample {i}: manual={pred_manual}, sklearn={pred_sklearn}, "
          f"true={y_test_bin[i]}")
# Predictions should match (small numerical differences possible).

# =========================================================================
# PART F: LDA vs QDA decision boundaries (2D synthetic)
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Decision boundaries -- when LDA fails, QDA wins")
print("=" * 70)
# Generate data where classes have DIFFERENT covariance structures.
# LDA assumes shared covariance -> will fail. QDA -> will succeed.
X_2d, y_2d = make_classification(
    n_samples=400, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.5, random_state=42
)
# Scale class 0 to have a different covariance than class 1
X_2d[y_2d == 0] *= np.array([1.0, 3.0])  # stretch class 0 vertically

X_tr, X_te, y_tr, y_te = train_test_split(
    X_2d, y_2d, test_size=0.3, random_state=42
)

lda_2d = discriminant_analysis.LinearDiscriminantAnalysis().fit(X_tr, y_tr)
qda_2d = discriminant_analysis.QuadraticDiscriminantAnalysis().fit(X_tr, y_tr)

print(f"LDA test accuracy: {lda_2d.score(X_te, y_te):.4f}")
print(f"QDA test accuracy: {qda_2d.score(X_te, y_te):.4f}")
print("(QDA should win here because class covariances differ.)")

# PREDICT BEFORE RUNNING: Will QDA beat LDA on this stretched data?
# Your guess:

# =========================================================================
# PART G: Shrinkage (LDA regularization)
# =========================================================================
print("\n" + "=" * 70)
print("PART G: LDA shrinkage (regularized covariance)")
print("=" * 70)
# Shrinkage mixes the empirical covariance with a scaled identity matrix:
#   Sigma_shrunk = (1 - alpha) * Sigma_empirical + alpha * mu * I
# Useful when n_samples is small relative to n_features, or when features
# are highly correlated. Only works with solver='lsqr' or 'eigen'.
print(f"{'shrinkage':>12} {'train_acc':>10} {'test_acc':>10}")
for shrink in [None, 0.1, 0.3, 0.5, "auto"]:
    lda_s = discriminant_analysis.LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage=shrink
    )
    lda_s.fit(X_train, y_train)
    print(f"{str(shrink):>12} {lda_s.score(X_train, y_train):>10.4f} "
          f"{lda_s.score(X_test, y_test):>10.4f}")

# PREDICT BEFORE RUNNING: Will shrinkage IMPROVE or HURT test accuracy on
# Iris? (Hint: Iris has only 4 features and 105 training samples -- not
# badly conditioned. Shrinkage may not help much.)
# Your guess:

# =========================================================================
# PART H: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART H: Reflection -- section 1.2")
print("=" * 70)

# Q1: In one sentence, what is the difference between LDA and QDA?
# Your answer:
#
# Q2: What is the Mahalanobis distance, and why does it matter for LDA?
# Your answer:
#
# Q3: You have 1000 samples, 10 classes, and 5 features. Which would you
#     try first -- LDA or QDA? Why?
# Your answer:
#
# Q4: What does LDA's `transform` do, and what is the maximum number of
#     output dimensions for a 4-class problem?
# Your answer:
#
# Q5: Why does QDA need more data than LDA to work well?
# Your answer: