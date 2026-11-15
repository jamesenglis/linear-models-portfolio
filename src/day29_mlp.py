# Day 29 (FINAL of section 1.1-1.17): Neural Network Models (Supervised)
# Covers scikit-learn User Guide §1.17
#
# MLP = Multi-Layer Perceptron. Fully connected feed-forward neural net.
# Stacked layers: input -> hidden_1 -> ... -> hidden_k -> output.
# Each layer: z = W x + b, a = activation(z).
#
# Scikit-learn provides:
#   MLPClassifier : classification
#   MLPRegressor  : regression
#
# What they are:
#   - Universal function approximators (with enough units and data).
#   - Trained with backpropagation + SGD/Adam.
#   - NOT deep learning (no CNNs, no RNNs, no attention) -- just MLPs.
#     For deep architectures, use PyTorch or TensorFlow.
#
# Key hyperparameters:
#   hidden_layer_sizes : tuple of integers. (100,) = 1 hidden layer of 100
#                        units. (100, 50) = 2 hidden layers.
#   activation         : 'relu' (default), 'tanh', 'logistic', 'identity'.
#   solver             : 'adam' (default), 'sgd', 'lbfgs'.
#   alpha              : L2 regularization strength.
#   learning_rate_init : initial learning rate (adam/sgd only).
#   learning_rate      : 'constant', 'invscaling', 'adaptive'.
#   max_iter           : maximum number of epochs.
#   early_stopping     : hold out a fraction for validation.
#   validation_fraction: fraction held out (default 0.1).
#   n_iter_no_change   : patience for early stopping.
#   batch_size         : mini-batch size (default min(200, n_samples)).
#
# Key practical requirements:
#   - ALWAYS SCALE DATA. MLPs are extremely scale-sensitive.
#   - More hidden units / layers = more capacity, but also more overfitting.
#   - MLPs are BLACK BOXES. For tabular data, try ensembles first
#     (Day 23: RandomForest, HistGradientBoosting often beat MLPs).
#   - Use MLPs when: features are continuous, data is large, and you've
#     tried simpler models. For images and text, use PyTorch/TF.
#
# Loss:
#   Classification: cross-entropy. Output layer uses softmax (multiclass)
#                   or logistic (binary).
#   Regression:     squared error. Output layer is linear.

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import time

# =========================================================================
# PART A: Toy Example -- MLPClassifier
# =========================================================================
print("=" * 70)
print("PART A: Toy Example -- XOR problem")
print("=" * 70)
# XOR is not linearly separable. A linear classifier CANNOT solve it.
# But an MLP with a hidden layer can.
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

mlp = MLPClassifier(
    hidden_layer_sizes=(5,), activation="relu",
    solver="adam", max_iter=5000, random_state=42
)
mlp.fit(X_xor, y_xor)
print(f"Predictions: {mlp.predict(X_xor)}")
print(f"True:        {y_xor}")
print(f"Accuracy:    {mlp.score(X_xor, y_xor):.4f}")
print(f"n_iter_:     {mlp.n_iter_} (epochs)")
print(f"Loss curve (last): {mlp.loss_curve_[-1]:.6f}")

# Linear baseline for comparison
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=2000).fit(X_xor, y_xor)
print(f"\nLogisticRegression (linear) accuracy: {lr.score(X_xor, y_xor):.4f}")
print("(LR cannot solve XOR; MLP can.)")

# PREDICT BEFORE RUNNING: Will logistic regression get 100% on XOR?
# (Hint: XOR is the classic example of a NON-linearly separable problem.)
# Your guess:

# =========================================================================
# PART B: MLPClassifier on Iris
# =========================================================================
print("\n" + "=" * 70)
print("PART B: MLPClassifier on Iris")
print("=" * 70)

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

mlp = MLPClassifier(
    hidden_layer_sizes=(100,), activation="relu", solver="adam",
    alpha=1e-4, learning_rate_init=0.001, max_iter=1000,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=20, random_state=42,
)
t0 = time.time()
mlp.fit(X_tr_s, y_tr)
dt = time.time() - t0
print(f"MLP(100,) train acc: {mlp.score(X_tr_s, y_tr):.4f}")
print(f"MLP(100,) test acc:  {mlp.score(X_te_s, y_te):.4f}")
print(f"n_iter_: {mlp.n_iter_}, fit time: {dt*1000:.0f} ms")
print(f"Final loss: {mlp.loss_curve_[-1]:.4f}")

# Compare with LogisticRegression (Day 9)
lr = LogisticRegression(max_iter=2000).fit(X_tr_s, y_tr)
print(f"\nLogisticRegression test acc: {lr.score(X_te_s, y_te):.4f}")

# PREDICT BEFORE RUNNING: Will MLP beat LogisticRegression on Iris?
# (Hint: Iris is mostly linear; MLP may match, but rarely dominates.)
# Your guess:

# =========================================================================
# PART C: Architecture comparison (hidden_layer_sizes)
# =========================================================================
print("\n" + "=" * 70)
print("PART C: Architecture comparison")
print("=" * 70)
# Use breast cancer (harder, better for comparing architectures).
Xb, yb = load_breast_cancer(return_X_y=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    Xb, yb, test_size=0.2, random_state=42, stratify=yb
)
scaler_b = StandardScaler()
Xb_tr_s = scaler_b.fit_transform(Xb_tr)
Xb_te_s = scaler_b.transform(Xb_te)

architectures = [
    (10,),
    (50,),
    (100,),
    (100, 50),
    (100, 100, 50),
    (500,),
]
print(f"{'hidden_layers':<22} {'train_acc':>10} {'test_acc':>10} {'n_params':>10} {'fit_ms':>8}")
for arch in architectures:
    mlp = MLPClassifier(
        hidden_layer_sizes=arch, activation="relu", solver="adam",
        alpha=1e-4, max_iter=500, early_stopping=True, random_state=42,
    )
    t0 = time.time()
    mlp.fit(Xb_tr_s, yb_tr)
    dt = time.time() - t0
    # Number of parameters (weights + biases)
    n_params = sum(coef.size for coef in mlp.coefs_) + sum(
        b.size for b in mlp.intercepts_
    )
    arch_str = str(arch)
    print(f"{arch_str:<22} {mlp.score(Xb_tr_s, yb_tr):>10.4f} "
          f"{mlp.score(Xb_te_s, yb_te):>10.4f} {n_params:>10} {dt*1000:>8.0f}")

# PREDICT BEFORE RUNNING: Will the DEEPEST network (100,100,50) give the
# HIGHEST test accuracy? (Hint: no -- too much capacity overfits, especially
# with 455 training samples. The sweet spot is usually 1-2 hidden layers.)
# Your guess:

# =========================================================================
# PART D: Activation functions
# =========================================================================
print("\n" + "=" * 70)
print("PART D: Activation functions")
print("=" * 70)
print(f"{'activation':<14} {'train_acc':>10} {'test_acc':>10}")
for activation in ["relu", "tanh", "logistic", "identity"]:
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,), activation=activation, solver="adam",
        alpha=1e-4, max_iter=500, early_stopping=True, random_state=42,
    )
    mlp.fit(Xb_tr_s, yb_tr)
    print(f"{activation:<14} {mlp.score(Xb_tr_s, yb_tr):>10.4f} "
          f"{mlp.score(Xb_te_s, yb_te):>10.4f}")

# PREDICT BEFORE RUNNING: Which activation will perform WORST?
# (Hint: 'identity' makes the whole network LINEAR, defeating the point.
# It reduces to logistic regression.)
# Your guess:

# =========================================================================
# PART E: Solvers (adam vs sgd vs lbfgs)
# =========================================================================
print("\n" + "=" * 70)
print("PART E: Solvers")
print("=" * 70)
print(f"{'solver':<8} {'train_acc':>10} {'test_acc':>10} {'fit_ms':>8} {'epochs':>8}")
for solver in ["adam", "sgd", "lbfgs"]:
    mlp = MLPClassifier(
        hidden_layer_sizes=(50,), activation="relu", solver=solver,
        alpha=1e-4, max_iter=500, random_state=42,
        # early_stopping not supported with lbfgs on all sklearn versions;
        # disable for fair comparison:
        early_stopping=False,
    )
    t0 = time.time()
    mlp.fit(Xb_tr_s, yb_tr)
    dt = time.time() - t0
    print(f"{solver:<8} {mlp.score(Xb_tr_s, yb_tr):>10.4f} "
          f"{mlp.score(Xb_te_s, yb_te):>10.4f} {dt*1000:>8.0f} {mlp.n_iter_:>8}")

# PREDICT BEFORE RUNNING: Which will converge in the FEWEST iterations?
# (Hint: lbfgs uses second-order information -- usually fewest iterations,
# though each is more expensive.)
# Your guess:

# =========================================================================
# PART F: alpha (L2 regularization) sweep
# =========================================================================
print("\n" + "=" * 70)
print("PART F: alpha (L2 regularization)")
print("=" * 70)
print(f"{'alpha':>10} {'train_acc':>10} {'test_acc':>10}")
for alpha in [1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,), activation="relu", solver="adam",
        alpha=alpha, max_iter=500, early_stopping=True, random_state=42,
    )
    mlp.fit(Xb_tr_s, yb_tr)
    print(f"{alpha:>10} {mlp.score(Xb_tr_s, yb_tr):>10.4f} "
          f"{mlp.score(Xb_te_s, yb_te):>10.4f}")

# PREDICT BEFORE RUNNING: As alpha grows, will train accuracy go UP or DOWN?
# (Hint: more regularization = harder to fit training data = lower train
# accuracy. Test accuracy may go up if overfitting was the issue.)
# Your guess:

# =========================================================================
# PART G: Early stopping -- loss curve
# =========================================================================
print("\n" + "=" * 70)
print("PART G: Early stopping (loss curve)")
print("=" * 70)
mlp = MLPClassifier(
    hidden_layer_sizes=(100,), activation="relu", solver="adam",
    alpha=1e-4, max_iter=2000, early_stopping=True,
    validation_fraction=0.2, n_iter_no_change=15, random_state=42,
)
mlp.fit(Xb_tr_s, yb_tr)
print(f"Stopped after {mlp.n_iter_} epochs")
print(f"Final loss: {mlp.loss_curve_[-1]:.4f}")
print(f"Test accuracy: {mlp.score(Xb_te_s, yb_te):.4f}")
print(f"Loss curve (first 5): {np.round(mlp.loss_curve_[:5], 4)}")
print(f"Loss curve (last 5):  {np.round(mlp.loss_curve_[-5:], 4)}")
print("(If early_stopping=True, the training STOPS when validation score")
print(" has not improved for n_iter_no_change epochs.)")

# PREDICT BEFORE RUNNING: Will the loss curve be MONOTONICALLY decreasing?
# (Hint: usually, but not always -- SGD with mini-batches can oscillate.)
# Your guess:

# =========================================================================
# PART H: MLPRegressor on diabetes
# =========================================================================
print("\n" + "=" * 70)
print("PART H: MLPRegressor")
print("=" * 70)
Xr, yr = load_diabetes(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)
scaler_r = StandardScaler()
Xr_tr_s = scaler_r.fit_transform(Xr_tr)
Xr_te_s = scaler_r.transform(Xr_te)

print(f"{'arch':<20} {'train_R2':>10} {'test_R2':>10} {'train_MSE':>12}")
for arch in [(50,), (100,), (100, 50)]:
    mlp = MLPRegressor(
        hidden_layer_sizes=arch, activation="relu", solver="adam",
        alpha=1e-4, max_iter=1000, early_stopping=True, random_state=42,
    )
    mlp.fit(Xr_tr_s, yr_tr)
    train_mse = mean_squared_error(yr_tr, mlp.predict(Xr_tr_s))
    print(f"{str(arch):<20} {mlp.score(Xr_tr_s, yr_tr):>10.4f} "
          f"{mlp.score(Xr_te_s, yr_te):>10.4f} {train_mse:>12.2f}")

# Baseline: Ridge
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0).fit(Xr_tr_s, yr_tr)
print(f"\nRidge test R^2: {ridge.score(Xr_te_s, yr_te):.4f}")

# PREDICT BEFORE RUNNING: Will the MLP BEAT Ridge on diabetes? (Hint: often
# NO. Diabetes is mostly linear, and Ridge is a strong baseline. This is
# the classic lesson: for tabular data, try linear / tree models first.)
# Your guess:

# =========================================================================
# PART I: Reflection -- completing the Supervised Learning chapter
# =========================================================================
print("\n" + "=" * 70)
print("PART I: Reflection -- section 1.17 (and completion of Chapter 1)")
print("=" * 70)

# Q1: What is an MLP? What does hidden_layer_sizes=(100, 50) mean?
# Your answer:
#
# Q2: Why do we need a non-linear activation like relu?
# Your answer:
#
# Q3: In practice, when would you choose an MLP over RandomForest or
#     HistGradientBoosting for tabular data?
#     (Hint: usually not. MLPs win when data is large, features are
#     continuous, and you have time to tune.)
# Your answer:
#
# Q4: Why does scikit-learn's MLPClassifier NOT replace PyTorch/TensorFlow?
#     (Hint: no GPU, no custom layers, no RNNs/CNNs, no autograd.)
# Your answer:
#
# Q5: What does `early_stopping=True` do and why is it critical for MLPs?
# Your answer:

# Portfolio completeness note: This is the final day of the Supervised
# Learning chapter (§1.1 through §1.17). You now have all 17 subsections
# covered with handwritten math checks.