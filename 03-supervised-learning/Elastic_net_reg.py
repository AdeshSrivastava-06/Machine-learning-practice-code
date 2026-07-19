from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet,LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X,y = load_diabetes(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2,test_size=0.2)

"""apply linear regression"""
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test,y_pred_lr)
print("Linear Regression R2 Score:", r2_lr)

"""apply ridge regression"""
ridge = Ridge(alpha=0.1)
ridge.fit(X_train,y_train)
y_pred_ridge = ridge.predict(X_test)
r2_ridge = r2_score(y_test,y_pred_ridge)
print("Ridge Regression R2 Score:", r2_ridge)

"""apply lasso regression"""
lasso = Lasso(alpha=0.01)
lasso.fit(X_train,y_train)
y_pred_lasso = lasso.predict(X_test)
r2_lasso = r2_score(y_test,y_pred_lasso)
print("Lasso Regression R2 Score:", r2_lasso)

"""apply elastic net regression"""
elastic_net = ElasticNet(alpha=0.005,l1_ratio=0.9)
elastic_net.fit(X_train,y_train)
y_pred_elastic_net = elastic_net.predict(X_test)
r2_elastic_net = r2_score(y_test,y_pred_elastic_net)
print("Elastic Net Regression R2 Score:", r2_elastic_net)

"""we can get the best value by hyperparameter tuning using GridSearchCV or RandomizedSearchCV"""
"""we have 2 options to perform elastic net the current code and the sdgd regression with elastic net penalty"""
