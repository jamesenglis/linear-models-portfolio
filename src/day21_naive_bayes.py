# Day 21: Naive Bayes
# Covers scikit-learn User Guide §1.9
#
# Naive Bayes applies Bayes' rule with a "naive" independence assumption:
# given the class, all features are treated as INDEPENDENT.
#
#     P(y=k | x) = P(y=k) * prod_j P(x_j | y=k) / P(x)
#
# The denominator P(x) is the same for all k, so we just take argmax over
# the numerator. The independence assumption is usually FALSE, yet NB
# still works well in practice for classification (especially text).
#
# Four variants (by the likelihood P(x_j | y)):
#   GaussianNB      : each feature ~ Normal(mu_jk, sigma_jk^2)
#                     -> continuous features
#   MultinomialNB   : each feature is a count, likelihood is multinomial
#                     -> text classification (word counts, tf-idf)
#   BernoulliNB     : each feature is binary (0/1)
#                     -> text with binarized counts, presence/absence
#   CategoricalNB   : each feature is a categorical integer (not ordered)
#                     -> categorical features (e.g. eye color 0/1/2)
#
# Key hyperparameter: alpha (Laplace/Lidstone smoothing).
#   With alpha=0 (default is 1.0, i.e. Laplace), P(x_j | y=k) = 0 for
#   any feature not seen with class k -> the entire product becomes 0.
#   alpha > 0 avoids this.
#
# Speed: fit O(n * p), predict O(n_test * p * k). Fastest of all classifiers.
# Trade-off: independence assumption is often violated -> poorly calibrated
# probabilities, but predictions are often surprisingly good.

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.datasets import load_iris, load_wine, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

# =========================================================================
# PART A: Toy Examples -- the API is uniform across all 4 variants
# =========================================================================
print("=" * 70)
print("PART A: Toy Examples")
print("=" * 70)

# GaussianNB: continuous features
X_toy = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y_toy = np.array([1, 1, 1, 2, 2, 2])
gnb = GaussianNB().fit(X_toy, y_toy)
print(f"GaussianNB predictions: {gnb.predict([[-0.8, -1]])}")
print(f"GaussianNB proba:       {gnb.predict_proba([[-0.8, -1]])}")

# MultinomialNB: counts
X_counts = np.array([[2, 1, 0], [3, 0, 1], [0, 2, 3], [1, 1, 2]])
y_counts = np.array([0, 0, 1, 1])
mnb = MultinomialNB().fit(X_counts, y_counts)
print(f"\nMultinomialNB predictions: {mnb.predict([[2, 1, 0]])}")
print(f"MultinomialNB proba:       {mnb.predict_proba([[2, 1, 0]])}")

# BernoulliNB: binary features
X_bin = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
y_bin = np.array([0, 1, 0, 1])
bnb = BernoulliNB().fit(X_bin, y_bin)
print(f"\nBernoulliNB predictions: {bnb.predict([[1, 0, 1]])}")
print(f"BernoulliNB proba:       {bnb.predict_proba([[1, 0, 1]])}")

# CategoricalNB: integer categories (not ordered)
X_cat = np.array([[0, 1, 0], [1, 0, 1], [2, 1, 0], [0, 2, 1], [1, 1, 1], [2, 0, 0]])
y_cat = np.array([0, 1, 1, 0, 1, 0])
cnb = CategoricalNB().fit(X_cat, y_cat)
print(f"\nCategoricalNB predictions: {cnb.predict([[1, 1, 1]])}")
print(f"CategoricalNB proba:       {cnb.predict_proba([[1, 1, 1]])}")

# PREDICT BEFORE RUNNING: For any of these, will the prediction be the class
# with the HIGHEST probability from predict_proba? (Hint: yes -- it is
# argmax over predict_proba for all four.)
# Your guess:

# =========================================================================
# PART B: GaussianNB on Iris
# =========================================================================
print("\n" + "=" * 70)
print("PART B: GaussianNB on Iris")
print("=" * 70)
X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

gnb = GaussianNB().fit(X_tr, y_tr)
print(f"Train acc: {gnb.score(X_tr, y_tr):.4f}")
print(f"Test acc:  {gnb.score(X_te, y_tr if False else y_te):.4f}")

# Compare with LogisticRegression and SVC (Days 9, 16)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
lr = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
svc = SVC(kernel="rbf", C=1.0, gamma=0.7).fit(X_tr, y_tr)
print(f"LogisticRegression test acc: {lr.score(X_te, y_te):.4f}")
print(f"SVC(RBF) test acc:           {svc.score(X_te, y_te):.4f}")

# PREDICT BEFORE RUNNING: On Iris (3 classes, 4 features), will GaussianNB
# match or beat LogisticRegression? (Hint: Iris features are roughly
# Gaussian per class, so GaussianNB has a good prior.)
# Your guess:

# =========================================================================
# PART C: Laplace smoothing (alpha) -- why it matters
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Laplace smoothing (alpha)")
print("=" * 70)
# MultinomialNB on a small count dataset where one class has an
# unseen feature for a test point. With alpha=0, the product is zero.
X_small = np.array([
    [5, 0, 2],
    [4, 1, 0],
    [0, 3, 4],
    [1, 2, 3],
])
y_small = np.array([0, 0, 1, 1])
X_test_small = np.array([[5, 0, 0]])  # third feature never seen in class 1

print(f"{'alpha':>8} {'pred':>6} {'proba class 0':>14} {'proba class 1':>14}")
for alpha in [0.0001, 0.001, 1.0, 10.0]:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_small, y_small)
    proba = mnb.predict_proba(X_test_small)[0]
    print(f"{alpha:>8} {mnb.predict(X_test_small)[0]:>6} "
          f"{proba[0]:>14.6f} {proba[1]:>14.6f}")

print("\nNote: as alpha grows, probabilities become more uniform (closer to")
print("the class prior). Large alpha => less confident predictions.")

# PREDICT BEFORE RUNNING: With alpha=0.0001, will proba[0] and proba[1]
# both be > 0.01, or will one be very close to 0? (Hint: very small alpha
# makes the zero-count feature dominate.)
# Your guess:

# =========================================================================
# PART D: Text classification -- MultinomialNB vs BernoulliNB
# =========================================================================
print("\n" + "=" * 70)
print("PART D: Text classification (20 newsgroups)")
print("=" * 70)
# Use a small subset for speed
categories = ["alt.atheism", "talk.religion.misc"]
data = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True, random_state=42
)
print(f"Data loaded: {len(data.data)} documents")

# Split into train/test first
docs_train, docs_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
)

# MultinomialNB on word counts
vec_counts = CountVectorizer(max_features=5000, stop_words="english")
X_tr_counts = vec_counts.fit_transform(docs_train)
X_te_counts = vec_counts.transform(docs_test)

mnb_text = MultinomialNB(alpha=0.1)
mnb_text.fit(X_tr_counts, y_train)
print(f"MultinomialNB (counts) test acc:  "
      f"{mnb_text.score(X_te_counts, y_test):.4f}")

# BernoulliNB on binary counts
vec_bin = CountVectorizer(
    max_features=5000, stop_words="english", binary=True
)
X_tr_bin = vec_bin.fit_transform(docs_train)
X_te_bin = vec_bin.transform(docs_test)

bnb_text = BernoulliNB(alpha=0.1)
bnb_text.fit(X_tr_bin, y_train)
print(f"BernoulliNB (binary) test acc:     "
      f"{bnb_text.score(X_te_bin, y_test):.4f}")

# MultinomialNB on TF-IDF (note: still fits, though the model "assumes" counts)
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_tr_tfidf = tfidf.fit_transform(docs_train)
X_te_tfidf = tfidf.transform(docs_test)
mnb_tfidf = MultinomialNB(alpha=0.1)
mnb_tfidf.fit(X_tr_tfidf, y_train)
print(f"MultinomialNB (tf-idf) test acc:   "
      f"{mnb_tfidf.score(X_te_tfidf, y_test):.4f}")

# PREDICT BEFORE RUNNING: Which will win -- MultinomialNB(counts),
# BernoulliNB(binary), or MultinomialNB(tf-idf)? (Hint: tf-idf typically
# wins on classification. Counts and binary are close.)
# Your guess:

# =========================================================================
# PART E: Manual GaussianNB prediction (proves the math)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: Manual GaussianNB prediction")
print("=" * 70)
# Take the first test point. Compute log P(y=k | x) manually.
#    log P(y=k | x) ~ log P(y=k) + sum_j log P(x_j | y=k)
# For Gaussian likelihood:
#    log P(x_j | y=k) = -0.5 * log(2*pi*var_jk)
#                      - (x_j - mu_jk)^2 / (2*var_jk)

gnb_manual = GaussianNB()
gnb_manual.fit(X_tr, y_tr)
x = X_te[0]
classes = gnb_manual.classes_

print(f"Test point: {x}")
print(f"True class: {y_te[0]}")

for i, c in enumerate(classes):
    log_prior = np.log(gnb_manual.class_prior_[i])
    log_lik = -0.5 * np.sum(np.log(2 * np.pi * gnb_manual.var_[i]))
    log_lik += -0.5 * np.sum(
        (x - gnb_manual.theta_[i]) ** 2 / gnb_manual.var_[i]
    )
    total = log_prior + log_lik
    print(f"  class {c}: log_prior={log_prior:.4f}, "
          f"log_lik={log_lik:.4f}, total={total:.4f}")

# Normalize over classes to get probabilities
log_joint = []
for i in range(len(classes)):
    log_prior = np.log(gnb_manual.class_prior_[i])
    log_lik = -0.5 * np.sum(np.log(2 * np.pi * gnb_manual.var_[i]))
    log_lik += -0.5 * np.sum(
        (x - gnb_manual.theta_[i]) ** 2 / gnb_manual.var_[i]
    )
    log_joint.append(log_prior + log_lik)
log_joint = np.array(log_joint)
log_norm = log_joint - log_joint.max()
proba_manual = np.exp(log_norm) / np.exp(log_norm).sum()

print(f"\nManual probabilities:  {np.round(proba_manual, 6)}")
print(f"Sklearn probabilities: {np.round(gnb_manual.predict_proba([x])[0], 6)}")

# PREDICT BEFORE RUNNING: Will manual and sklearn agree to 4+ decimal places?
# Your guess:

# =========================================================================
# PART F: Naive Bayes vs calibrated models
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Naive Bayes calibration")
print("=" * 70)
# NB is famous for BADLY calibrated probabilities: it tends to output
# probabilities near 0 or 1, even when it's not confident. Log-loss
# exposes this, even when accuracy is fine.

wine = load_wine(return_X_y=True)
Xw, yw = wine
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
    Xw, yw, test_size=0.3, random_state=42, stratify=yw
)
scaler = StandardScaler()
Xw_tr_s = scaler.fit_transform(Xw_tr)
Xw_te_s = scaler.transform(Xw_te)

from sklearn.linear_model import LogisticRegression
for name, model, X_tr_use, X_te_use in [
    ("GaussianNB (raw)",       GaussianNB(),        Xw_tr, Xw_te),
    ("GaussianNB (scaled)",    GaussianNB(),        Xw_tr_s, Xw_te_s),
    ("LogisticRegression",     LogisticRegression(max_iter=2000), Xw_tr_s, Xw_te_s),
]:
    model.fit(X_tr_use, yw_tr)
    y_prob = model.predict_proba(X_te_use)
    acc = model.score(X_te_use, yw_te)
    ll = log_loss(yw_te, y_prob, labels=model.classes_)
    print(f"{name:<22}: test_acc={acc:.4f}, test_log_loss={ll:.4f}")

print("\n(Lower log-loss = better calibrated. NB usually loses here even")
print(" when accuracy is fine.)")

# PREDICT BEFORE RUNNING: Will GaussianNB have LOWER accuracy AND HIGHER
# log-loss than LogisticRegression, or same accuracy but higher log-loss?
# Your guess:

# =========================================================================
# PART G: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Reflection -- section 1.9")
print("=" * 70)

# Q1: In one sentence, what is the "naive" in Naive Bayes?
# Your answer:
#
# Q2: When would you use MultinomialNB over GaussianNB?
# Your answer:
#
# Q3: What does the `alpha` parameter do? Why is it needed?
# Your answer:
#
# Q4: If Naive Bayes is "wrong" (independence assumption violated), why
#     does it still work well for classification?
# Your answer:
#
# Q5: Why is NB's predicted probability often badly calibrated?
# Your answer: