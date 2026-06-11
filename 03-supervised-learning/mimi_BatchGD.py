from sklearn.datasets import load_diabetes

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X,y = load_diabetes(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_scaled,y_train)

print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(X_test_scaled)
print("R2 Score for Linear Regression using sklearn:", r2_score(y_test,y_pred))

import random

class MBGDRegressor:
    
    def __init__(self,batch_size,learning_rate=0.01,epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
    def fit(self,X_train_scaled,y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train_scaled.shape[1])
        
        for i in range(self.epochs):
            
            for j in range(int(X_train_scaled.shape[0]/self.batch_size)):
                
                idx = random.sample(range(X_train_scaled.shape[0]),self.batch_size)
                
                y_hat = np.dot(X_train_scaled[idx],self.coef_) + self.intercept_

                intercept_der = -2 * np.mean(y_train[idx] - y_hat)
                
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)

                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train_scaled[idx]) / self.batch_size
                
                self.coef_ = self.coef_ - (self.lr * coef_der)

    def predict(self,X_test_scaled):
        return np.dot(X_test_scaled,self.coef_) + self.intercept_
    
mbgd = MBGDRegressor(batch_size=35,learning_rate=0.01,epochs=100)
print(mbgd.fit(X_train_scaled,y_train))   
y_pred_mbgd = mbgd.predict(X_test_scaled)
r = r2_score(y_test,y_pred_mbgd)

print("R2 Score for custom MBGDRegressor:--",r)

"""from sklearn"""

from sklearn.linear_model import SGDRegressor

sgd  = SGDRegressor(learning_rate='constant',eta0=0.01,loss='squared_error')
batch_size = 35

for i in range(100):
    idx = random.sample(range(X_train_scaled.shape[0]),batch_size)
    sgd.partial_fit(X_train_scaled[idx],y_train[idx])
    
print("coeficients\n:--",sgd.coef_
      ,"intercept\n:--",sgd.intercept_)

print("R2 Score for sklearn SGDRegressor:--",r2_score(y_test,sgd.predict(X_test_scaled)))