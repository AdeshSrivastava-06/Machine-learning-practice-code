from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.data, housing.target

print('Dataset features:',housing.feature_names)
print('Dataset target:',housing.target_names)
print('Dataset shape:',X.shape,y.shape)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print('Training set shape:',X_train.shape,y_train.shape)
print('Test set shape:',X_test.shape,y_test.shape)

lr = LinearRegression()
dt = DecisionTreeRegressor()
knn = KNeighborsRegressor()

lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
print('R2 score of Linear Regression',r2_score(y_test,y_pred_lr))

dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)
print('R2 score of Decision Tree Regressor',r2_score(y_test,y_pred_dt))

knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)
print('R2 score of KNN Regressor',r2_score(y_test,y_pred_knn))

from sklearn.ensemble import BaggingRegressor

bag_reg = BaggingRegressor(random_state=1)

bag_reg.fit(X_train,y_train)
y_pred_bag = bag_reg.predict(X_test)

print('R2 score of Bagging Regressor',r2_score(y_test,y_pred_bag))

print('training set score:',bag_reg.score(X_train,y_train))
print('test set score:',bag_reg.score(X_test,y_test))

"""here we have used the default hyperparameters of bagging regressor, we can tune the hyperparameters using GridSearchCV"""

"""using the OOB (Out-of-Bag) score to estimate the generalization error of the model"""

bag_reg_oob = BaggingRegressor(random_state=1,oob_score=True)
bag_reg_oob.fit(X_train,y_train)
print('OOB score of Bagging Regressor',bag_reg_oob.oob_score_)
bag_reg_pred = bag_reg_oob.predict(X_test)
print('R2 score of Bagging Regressor with OOB score',r2_score(y_test,bag_reg_pred))
