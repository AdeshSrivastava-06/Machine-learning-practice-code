from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score


x,y = make_regression(n_samples=100,n_features=1,n_informative=1,n_targets=1,noise=20,random_state=42)

plt.scatter(x,y)
plt.show()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2) 

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

print('lr coef-',lr.coef_)
print('lr intercept--',lr.intercept_)
y_pred = lr.predict(x_test)
from sklearn.metrics import r2_score

print('r2-score using sklearn:--',r2_score(y_test,y_pred))

m = 47.13

class GDRegressor:
    
    def __init__(self,learning_rate,epochs):
        self.m = 100
        self.b = -120
        self.lr= learning_rate
        self.epochs = epochs
        
    def fit(self,x,y):
        #calculate the b using GD
        for i in range(self.epochs):
            
            loss_slope_b = -2*np.sum(y - self.m*x.ravel() - self.b)
            
            self.b = self.b - (self.lr * loss_slope_b)
            
            loss_slope_m = -2*np.sum((y - self.m*x.ravel() - self.b)*x.ravel())
            
            self.m = self.m - (self.lr * loss_slope_m)
        print(self.b,self.m)
        
    def predict(self,x):
        return self.m*x + self.b
            
              
gd = GDRegressor(0.001,80)
gd.fit(x_train,y_train)

y_pred = gd.predict(x_test)
from sklearn.metrics import r2_score

print('r2-score using GD:--',r2_score(y_test,y_pred))

