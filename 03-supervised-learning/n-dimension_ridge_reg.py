from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np  
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X,y = load_diabetes(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

reg = Ridge(alpha=0.1,solver='cholesky')

"""solver can be 'svd','cholesky','lsqr','sparse_cg','sag','saga' here gradient descent based solvers are 'sag' and 'saga' rest are direct solvers"""

reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

print("R2 Score for sklearn Ridge:", r2_score(y_test,y_pred))

print(reg.coef_)
print(reg.intercept_)

class CustomRidge:
   
   def __init__(self,alpha=1.0):
       self.alpha = alpha
       self.m = None
       self.b = None
       
   def fit(self,X_train,y_train): 
        
        X_train = np.insert(X_train,0,1,axis=1)
        I = np.identity(X_train.shape[1])
        I[0][0] = 0
        result = np.linalg.inv(X_train.T.dot(X_train) + self.alpha * I).dot(X_train.T).dot(y_train)
        self.b = result[0]
        self.m = result[1:]
        
   def predict(self,X_test):
       return np.dot(X_test,self.m) + self.b
   
reg = CustomRidge(alpha=0.1)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("R2 Score for CustomRidge:", r2_score(y_test,y_pred))
print(reg.m)
print(reg.b)
