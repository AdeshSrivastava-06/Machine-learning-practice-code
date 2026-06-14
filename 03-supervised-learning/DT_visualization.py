
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from dtreeviz import model
import matplotlib.pyplot as plt
import pandas as pd


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_train, y_train)

plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.show()

X_train_df = pd.DataFrame(X_train, columns=iris.feature_names)


viz = model(clf,
            X_train=X_train_df,
            y_train=y_train,
            feature_names=iris.feature_names,
            target_name="iris",
            class_names=list(iris.target_names))

viz.view().save("decision_tree.svg")

