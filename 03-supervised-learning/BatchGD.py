from sklearn.datasets import load_diabetes

import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X,y = load_diabetes(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

reg = LinearRegression()
reg.fit(x_train_scaled,y_train)

print("Coefficients:--",reg.coef_)
print("Intercept:--",reg.intercept_)

y_pred = reg.predict(x_test_scaled)
print("R2 Score for Linear Regression using sklearn:", r2_score(y_test,y_pred))

class GDRegressor:
    
    def __init__(self,learning_rate = 0.01,epochs=100):
        self.coef_=None
        self.intercept_=None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,x_train_scaled,y_train):
        
        self.intercept_ = 0
        self.coef_=np.ones(x_train_scaled.shape[1])
        
        for i in range(self.epochs):
            
            y_hat = np.dot(x_train_scaled,self.coef_) + self.intercept_
            
            intercept_der = -2 * np.mean(y_train - y_hat)
            
            self.intercept_ = self.intercept_ - (self.lr*intercept_der)

            coef_der = -2 * np.dot((y_train - y_hat),x_train_scaled)/x_train_scaled.shape[0]
            self.coef_ = self.coef_ - (self.lr*coef_der)
    
    def predict(self,x_test_scaled):
        return np.dot(x_test_scaled,self.coef_) + self.intercept_
    
gdr = GDRegressor(epochs=1000,learning_rate=0.01)
gdr.fit(x_train_scaled,y_train)

y_pred = gdr.predict(x_test_scaled)
print("R2 Score for custom GDRegressor:", r2_score(y_test,y_pred))

"""from sklearn"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=0.001,
    max_iter=1500,
    tol=None,
    random_state=2
)

sgd.fit(x_train_scaled,y_train)
y_pred3 = sgd.predict(x_test_scaled)

print("Intercept:-",sgd.intercept_)
print("Coefficients:--",sgd.coef_)
print("R2-score for sklearn SGDRegressor:--",r2_score(y_test,y_pred3))