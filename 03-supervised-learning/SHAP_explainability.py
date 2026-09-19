"""
SHAP Explainability

SHAP (SHapley Additive exPlanations) explains individual predictions
by assigning each feature a contribution value — how much it pushed
the prediction up or down from the baseline (average prediction).

Unlike plain feature_importances_ (which only tells you what matters
GLOBALLY, across the whole model), SHAP can explain a single
prediction: "why did the model predict THIS for THIS patient/row?"

"""

import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# 1. Load data and train model

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))


# 2. Build a SHAP explainer
#    TreeExplainer is fast and exact for tree-based models
#    (RandomForest, XGBoost, LightGBM, CatBoost, DecisionTree)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Different shap versions return this differently for binary classification:
#   - older versions: a list [class_0_values, class_1_values]
#   - newer versions: a single ndarray of shape (n_samples, n_features, n_classes)
# Either way, we want the 2D (n_samples, n_features) array for class 1.
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
elif np.asarray(shap_values).ndim == 3:
    shap_values_class1 = shap_values[:, :, 1]
else:
    shap_values_class1 = shap_values

print(f"\nSHAP values shape: {np.array(shap_values_class1).shape}")
print(f"(rows = test samples, columns = features)")


# 3. Global explanation: which features matter most overall?
#    Mean absolute SHAP value per feature = average impact on
#    prediction magnitude, across all test samples

mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[::-1][:10]

print("\nTop 10 most important features (by mean |SHAP value|):")
for idx in top_features_idx:
    print(f"  {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")

# Summary plot: shows feature importance AND the direction of effect
# (red = high feature value, blue = low feature value)
shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved shap_summary_plot.png")


# 4. Local explanation: why did the model predict THIS for ONE sample?

sample_idx = 0
sample = X_test[sample_idx]
prediction = model.predict([sample])[0]
prediction_proba = model.predict_proba([sample])[0]

print(f"\nExplaining test sample #{sample_idx}")
print(f"  Model predicted class: {prediction} (probabilities: {prediction_proba})")
print(f"  Actual class: {y_test[sample_idx]}")
base_value = explainer.expected_value
if isinstance(base_value, np.ndarray):
    base_value = base_value[1] if base_value.shape[0] > 1 else base_value[0]
print(f"  Base value (average prediction): {base_value:.4f}")

# Show top contributing features for this one prediction
sample_shap = shap_values_class1[sample_idx]
contribution_order = np.argsort(np.abs(sample_shap))[::-1][:5]
print("  Top 5 features driving this prediction:")
for idx in contribution_order:
    direction = "pushed UP" if sample_shap[idx] > 0 else "pushed DOWN"
    print(f"    {feature_names[idx]} = {sample[idx]:.3f} -> {direction} by {abs(sample_shap[idx]):.4f}")
