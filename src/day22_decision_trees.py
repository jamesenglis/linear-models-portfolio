# Day 22: Decision Trees
# Covers scikit-learn User Guide §1.10
#
# Decision trees partition the feature space with axis-aligned splits.
# Each internal node tests ONE feature against a threshold; leaves give
# the prediction (majority class for classification, mean for regression).
#
# SPLIT CRITERIA (classification):
#   gini    : Gini impurity = 1 - sum_k p_k^2
#             Lower is purer (0 = pure leaf).
#   entropy : H = -sum_k p_k * log2(p_k)
#             Also 0 for pure; usually gives slightly different trees.
#
# SPLIT CRITERIA (regression):
#   squared_error : variance reduction (default)
#   absolute_error: MAE reduction (slower)
#   friedman_mse  : modified MSE (used by gradient boosting internally)
#   poisson       : Poisson deviance (for counts)
#
# STOPPING CRITERIA (avoid overfitting):
#   max_depth           : maximum tree depth
#   min_samples_split   : minimum samples to allow a split
#   min_samples_leaf    : minimum samples required at each leaf
#   max_leaf_nodes      : maximum number of leaves
#   min_impurity_decrease: minimum impurity improvement to split
#
# PRUNING: cost-complexity pruning (ccp_alpha).
#   After training, prune subtrees whose complexity (size) does not
#   improve impurity enough. Larger ccp_alpha = more aggressive pruning.
#
# Pros: interpretable, non-parametric, handles mixed types and nonlinearity.
# Cons: high variance (small data changes -> very different tree).
#       Fix: ensemble (RandomForest, GradientBoosting) -> Day 23.

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# PART A: Toy Example -- iris, fit and inspect the tree
# =========================================================================
print("=" * 70)
print("PART A: Toy Example -- iris tree")
print("=" * 70)

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Full tree (no pruning) -- overfits
tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_tr, y_tr)
print(f"Full tree: depth={tree_full.get_depth()}, leaves={tree_full.get_n_leaves()}")
print(f"Full tree train acc: {tree_full.score(X_tr, y_tr):.4f}")
print(f"Full tree test acc:  {tree_full.score(X_te, y_te):.4f}")

# Pruned tree -- depth=3
tree_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_pruned.fit(X_tr, y_tr)
print(f"\nDepth-3 tree: depth={tree_pruned.get_depth()}, "
      f"leaves={tree_pruned.get_n_leaves()}")
print(f"Depth-3 train acc: {tree_pruned.score(X_tr, y_tr):.4f}")
print(f"Depth-3 test acc:  {tree_pruned.score(X_te, y_te):.4f}")

# Text representation
print("\nTree structure (depth 3):")
print(export_text(tree_pruned, feature_names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width"
]))

# Save a plot of the tree
plt.figure(figsize=(12, 8))
plot_tree(tree_pruned, feature_names=[
    "sepal_length", "sepal_width", "petal_length", "petal_width"
], class_names=["setosa", "versicolor", "virginica"], filled=True)
plt.tight_layout()
plt.savefig("data/day22_iris_tree.png", dpi=100)
print("\n(Saved tree plot -> data/day22_iris_tree.png)")

# PREDICT BEFORE RUNNING: Will the full tree have 100% TRAINING accuracy?
# (Hint: a tree can always memorize its training data if not stopped.)
# Your guess:

# =========================================================================
# PART B: Overfitting with depth
# =========================================================================
print("\n" + "=" * 70)
print("PART B: Depth sweep on iris")
print("=" * 70)
print(f"{'max_depth':>10} {'train_acc':>10} {'test_acc':>10} {'n_leaves':>10}")
for depth in [1, 2, 3, 4, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_tr, y_tr)
    d_str = str(depth) if depth is not None else "None"
    print(f"{d_str:>10} {tree.score(X_tr, y_tr):>10.4f} "
          f"{tree.score(X_te, y_te):>10.4f} {tree.get_n_leaves():>10}")

# PREDICT BEFORE RUNNING: As max_depth increases, will train_acc rise
# monotonically (always or never)? Will test_acc rise monotonically or peak
# then fall?
# Your guess:

# =========================================================================
# PART C: Gini vs Entropy
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Gini vs entropy criterion")
print("=" * 70)
for criterion in ["gini", "entropy", "log_loss"]:
    tree = DecisionTreeClassifier(
        criterion=criterion, max_depth=3, random_state=42
    )
    tree.fit(X_tr, y_tr)
    print(f"criterion={criterion:<10}: train={tree.score(X_tr, y_tr):.4f}, "
          f"test={tree.score(X_te, y_tr if False else y_te):.4f}")

# PREDICT BEFORE RUNNING: Will gini and entropy give EXACTLY the same tree?
# (Hint: often they do, because both rank splits similarly on clean data.
# Occasionally they differ by a split.)
# Your guess:

# =========================================================================
# PART D: Cost-complexity pruning (ccp_alpha)
# =========================================================================
print("\n" + "=" * 70)
print("PART D: Cost-complexity pruning")
print("=" * 70)
# Step 1: find candidate alphas
tree_big = DecisionTreeClassifier(random_state=42)
tree_big.fit(X_tr, y_tr)
path = tree_big.cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas = path.ccp_alphas[:-1]  # drop the last one (single node)
print(f"Number of candidate alphas: {len(ccp_alphas)}")
print(f"Alpha range: {ccp_alphas[0]:.6f} to {ccp_alphas[-1]:.6f}")

# Step 2: fit a tree at each alpha and record test acc
print(f"\n{'ccp_alpha':>12} {'n_leaves':>10} {'train_acc':>10} {'test_acc':>10}")
for alpha in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]:
    tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    tree.fit(X_tr, y_tr)
    print(f"{alpha:>12} {tree.get_n_leaves():>10} "
          f"{tree.score(X_tr, y_tr):>10.4f} {tree.score(X_te, y_te):>10.4f}")

# PREDICT BEFORE RUNNING: As ccp_alpha grows, will n_leaves increase or
# decrease? (Larger alpha = more pruning = fewer leaves.)
# Your guess:

# =========================================================================
# PART E: Manual gini calculation (proves the math)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: Manual Gini and entropy calculation")
print("=" * 70)
# Consider a node with 100 samples split 50/50 between 2 classes.
def gini_impurity(class_counts):
    total = sum(class_counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in class_counts]
    return 1.0 - sum(p ** 2 for p in probs)

def entropy_impurity(class_counts):
    total = sum(class_counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in class_counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)

print(f"{'class_counts':>20} {'gini':>10} {'entropy':>10}")
for counts in [[50, 50], [80, 20], [100, 0], [33, 33, 34], [10, 10, 80]]:
    print(f"{str(counts):>20} {gini_impurity(counts):>10.4f} "
          f"{entropy_impurity(counts):>10.4f}")

# A split is chosen by the impurity DECREASE:
#   delta = impurity(parent) - sum(weighted impurity(children))
def split_gain(parent, left, right, criterion="gini"):
    if criterion == "gini":
        imp_parent = gini_impurity(parent)
        imp_left = gini_impurity(left)
        imp_right = gini_impurity(right)
    else:
        imp_parent = entropy_impurity(parent)
        imp_left = entropy_impurity(left)
        imp_right = entropy_impurity(right)
    n = sum(parent)
    n_l = sum(left)
    n_r = sum(right)
    return imp_parent - (n_l / n) * imp_left - (n_r / n) * imp_right

# Consider parent=[50, 50] (2 classes). Compare two candidate splits:
split1 = ([50, 50], [40, 10], [10, 40])  # split[0] = parent, [1],[2] = children
split2 = ([50, 50], [30, 20], [20, 30])
print(f"\nGini gain for split [40,10] / [10,40]: "
      f"{split_gain(*split1, 'gini'):.4f}")
print(f"Gini gain for split [30,20] / [20,30]: "
      f"{split_gain(*split2, 'gini'):.4f}")
print("(Higher gain wins. The first split is cleaner.)")

# PREDICT BEFORE RUNNING: Will split 1 or split 2 have higher gini gain?
# Your guess:

# =========================================================================
# PART F: Decision tree REGRESSION
# =========================================================================
print("\n" + "=" * 70)
print("PART F: Decision tree regression")
print("=" * 70)
Xr, yr = load_diabetes(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)

print(f"{'max_depth':>10} {'train_R2':>10} {'test_R2':>10} {'train_MSE':>10}")
for depth in [2, 3, 5, 10, None]:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(Xr_tr, yr_tr)
    train_mse = mean_squared_error(yr_tr, tree.predict(Xr_tr))
    d_str = str(depth) if depth is not None else "None"
    print(f"{d_str:>10} {tree.score(Xr_tr, yr_tr):>10.4f} "
          f"{tree.score(Xr_te, yr_te):>10.4f} {train_mse:>10.2f}")

# Compare with Ridge from Day 2
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0).fit(Xr_tr, yr_tr)
print(f"\nRidge test R^2: {ridge.score(Xr_te, yr_te):.4f}")

# PREDICT BEFORE RUNNING: Will the best decision tree (tuned depth) beat
# Ridge on diabetes? (Hint: single trees rarely beat well-tuned linear
# models on tabular data with mostly linear structure.)
# Your guess:

# =========================================================================
# PART G: Feature importance
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Feature importance")
print("=" * 70)
# importance = total impurity decrease contributed by each feature,
# normalized to sum to 1. Gives a rough ranking of feature usefulness.
feature_names = load_iris().feature_names
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
importances = tree.feature_importances_
print("Iris feature importances:")
for name, imp in sorted(
    zip(feature_names, importances), key=lambda x: -x[1]
):
    print(f"  {name:<15}: {imp:.4f}")

print("\ndiabetes feature importances (tree depth=5):")
feature_names_d = load_diabetes().feature_names
tree_d = DecisionTreeRegressor(max_depth=5, random_state=42).fit(Xr_tr, yr_tr)
for name, imp in sorted(
    zip(feature_names_d, tree_d.feature_importances_), key=lambda x: -x[1]
):
    print(f"  {name:<15}: {imp:.4f}")

# PREDICT BEFORE RUNNING: Will petal_width or sepal_length be the most
# important feature for iris? (Hint: petal_width is almost perfect for
# setosa separation.)
# Your guess:

# =========================================================================
# PART H: Reflection
# =========================================================================
print("\n" + "=" * 70)
print("PART H: Reflection -- section 1.10")
print("=" * 70)

# Q1: In one sentence, how does a decision tree split the feature space?
# Your answer:
#
# Q2: What is Gini impurity? What does it equal for a pure node?
# Your answer:
#
# Q3: Why do trees overfit as they grow, and how do you stop them?
# Your answer:
#
# Q4: What does ccp_alpha control?
# Your answer:
#
# Q5: Why are single decision trees rarely the final model?
#     (Hint: variance. Day 23 ensembles will fix this.)
# Your answer: