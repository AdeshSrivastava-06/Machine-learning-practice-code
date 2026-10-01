"""
CALIBRATION CURVE

A Calibration Curve is used to check whether the probability predicted
by a classification model is reliable or not.

For example, if a model predicts 100 cases with 80% probability of
being positive, then approximately 80 of those cases should actually
be positive if the model is well calibrated.

Why is it needed?
- Accuracy only tells us how many predictions are correct.
- Calibration tells us whether the predicted probabilities are trustworthy.
- It is useful when we care about the confidence of a prediction, such
  as in medical diagnosis, fraud detection, and risk prediction.

How does it work?
The predicted probabilities are divided into different ranges (bins).
The average predicted probability is compared with the actual fraction
of positive cases in each bin.

A well-calibrated model will have its calibration curve close to the
diagonal line (Perfect Calibration).

Brier Score can also be used to measure the quality of predicted
probabilities. A lower Brier Score generally indicates better
probabilistic predictions.
"""
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score


# 1. Load real dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Features:", data.feature_names)


# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# 4. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# 5. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# 6. Calibration Curve
prob_true, prob_pred = calibration_curve(
    y_test,
    y_prob,
    n_bins=10,
    strategy="uniform"
)


# 7. Brier Score
brier = brier_score_loss(y_test, y_prob)

print("Brier Score:", brier)


# 8. Plot Calibration Curve
plt.figure(figsize=(8, 6))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="Random Forest"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")

plt.title("Calibration Curve - Breast Cancer Dataset")

plt.legend()
plt.grid()

plt.show()
