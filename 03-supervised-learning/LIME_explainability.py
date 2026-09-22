"""
LIME Explainability

LIME (Local Interpretable Model-agnostic Explanations) explains a
single prediction by fitting a simple, interpretable model (linear
regression) around that ONE prediction only — it perturbs the input
slightly, sees how the model's output changes, and uses that local
behavior to approximate feature contributions.

Difference from SHAP:
    - SHAP is based on game theory (Shapley values) and gives
      mathematically consistent, additive contributions.
    - LIME is model-agnostic and local: it approximates the model's
      decision boundary near one point with a simple linear model.
      Faster and easier to reason about, but explanations can be
      less stable (different runs may give slightly different values
      due to random perturbation sampling).

Works with ANY model (not just tree-based), since it treats the
model as a black box — only needs .predict_proba().
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular


# 1. Load data and train model

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))


# 2. Build a LIME explainer for tabular data

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
    discretize_continuous=True,
    random_state=42,
)


# 3. Explain a single prediction

sample_idx = 0
sample = X_test[sample_idx]
prediction = model.predict([sample])[0]
prediction_proba = model.predict_proba([sample])[0]

print(f"\nExplaining test sample #{sample_idx}")
print(f"  Model predicted: {class_names[prediction]} (probabilities: {prediction_proba})")
print(f"  Actual class: {class_names[y_test[sample_idx]]}")

explanation = explainer.explain_instance(
    sample,
    model.predict_proba,
    num_features=8,
    num_samples=5000,   # number of perturbed samples LIME generates around this point
)

print("\nTop features driving this prediction:")
for feature_condition, weight in explanation.as_list():
    direction = "supports predicted class" if weight > 0 else "against predicted class"
    print(f"  {feature_condition}: weight={weight:.4f} ({direction})")


# 4. Explain a second sample the model got WRONG (if any exist)
#    Useful for understanding failure cases

test_preds = model.predict(X_test)
wrong_idxs = np.where(test_preds != y_test)[0]

if len(wrong_idxs) > 0:
    wrong_idx = wrong_idxs[0]
    wrong_sample = X_test[wrong_idx]

    print(f"\nExplaining a MISCLASSIFIED sample (test index #{wrong_idx})")
    print(f"  Model predicted: {class_names[test_preds[wrong_idx]]}, actual: {class_names[y_test[wrong_idx]]}")

    wrong_explanation = explainer.explain_instance(
        wrong_sample, model.predict_proba, num_features=8, num_samples=5000
    )
    for feature_condition, weight in wrong_explanation.as_list():
        print(f"  {feature_condition}: weight={weight:.4f}")
else:
    print("\nNo misclassified samples in test set to explain.")
