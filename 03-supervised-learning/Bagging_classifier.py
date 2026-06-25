from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X,y = make_classification(n_samples=10000, n_features=10 , n_informative=3)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

print('Accuracy of Decision Tree Classifier',accuracy_score(y_test,y_pred))

"""Bagging"""

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    random_state=42)

bag.fit(X_train,y_train)
y_pred_bag = bag.predict(X_test)

print('Accuracy of Bagging Classifier',accuracy_score(y_test,y_pred_bag))


"""Pasting- same as bagging but with bootstrap=False"""

bag_paste = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=False,
    random_state=42,
    verbose=1,
    n_jobs=-1
    )

bag_paste.fit(X_train,y_train)
y_pred_paste = bag_paste.predict(X_test)

print('Accuracy of Pasting Classifier',accuracy_score(y_test,y_pred_paste))

"""Random Subspaces- same as bagging but with max_features<1.0  - random subset of features for each base estimator"""

bag_subspace = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=1.0,
    bootstrap=False,
    max_features=0.5,
    bootstrap_features=True,
    random_state=42,
)

bag_subspace.fit(X_train,y_train)
y_pred_subspace = bag_subspace.predict(X_test)
print('Accuracy of Random Subspace Classifier',accuracy_score(y_test,y_pred_subspace))


"""Random Patches- same as bagging but with max_samples<1.0 and max_features<1.0 - random subset of samples and features for each base estimator here we are using both random subspace and random sampling"""

bag_patch = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    max_features=0.5,
    bootstrap_features=True,
    random_state=42,
)

bag_patch.fit(X_train,y_train)
y_pred_patch = bag_patch.predict(X_test)
print('Accuracy of Random Patch Classifier',accuracy_score(y_test,y_pred_patch))


"""OOB score- Out of bag score- when bootstrap=True, each base estimator is trained on a random subset of the training data, so some samples are not used for training that base estimator. These samples are called out-of-bag samples. The OOB score is the average accuracy of the base estimators on their respective OOB samples. It can be used as an estimate of the generalization error of the ensemble without needing a separate validation set."""

bag_oob = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    oob_score=True,
    random_state=42,
)
bag_oob.fit(X_train,y_train)
y_pred_oob = bag_oob.predict(X_test)
print('Accuracy of Bagging Classifier with OOB score',accuracy_score(y_test,y_pred_oob))
print('OOB Score of Bagging Classifier',bag_oob.oob_score_)


"""Applying GRID SEARCH CV to find the best hyperparameters for bagging classifier"""

from sklearn.model_selection import GridSearchCV

parameters = {
    'n_estimators': [100, 200, 500],
    'max_samples': [0.25, 0.5, 1.0],
    'bootstrap': [True, False],
    'max_features': [0.5, 1.0],
    'bootstrap_features': [True, False]
}

search = GridSearchCV(
    BaggingClassifier(),parameters,cv=5)
search.fit(X_train,y_train)
print('Best Hyperparameters for Bagging Classifier',search.best_params_)    

"""In summary Bagging is an ensemble method which uses the same base estimator multiple times on different random subsets of the training data to create an ensemble of models. It can be used for both classification and regression tasks and it helps to reduce overfitting and improve the generalization performance of the model.

methods of bagging:
1. Bagging- random sampling with replacement
2. Pasting- random sampling without replacement
3. Random Subspaces- random subset of features for each base estimator
4. Random Patches- random subset of samples and features for each base estimator
5. OOB (Out-of-Bag) Score- estimate of the generalization error using out-of-bag samples
"""
