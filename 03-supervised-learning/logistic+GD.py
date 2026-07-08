from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=20)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Data')
plt.show()

from sklearn.linear_model import LogisticRegression

lor = LogisticRegression(penalty=None,solver='sag')
lor.fit(X,y)
print("Coefficients:", lor.coef_)
print("Intercept:", lor.intercept_)

m1 = -(lor.coef_[0][0]/lor.coef_[0][1])
b1 = -(lor.intercept_/lor.coef_[0][1])

x_input = np.linspace(-3,3,100)
y_input = m1*x_input + b1

def gd(X, y):
    
    X=np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    
    lr=0.5
    for _ in range(5000):
        y_hat = sigmoid(np.dot(X, weights))
        weights = weights + lr*(np.dot(y-y_hat,X)/X.shape[0])
        
    return weights[1:], weights[0]

def sigmoid(z):
    return 1/(1+np.exp(-z))

coef_, intercept_ = gd(X,y)
print("Coefficients:", coef_)   
print("Intercept:", intercept_)
m2 = -(coef_[0]/coef_[1])
b2 = -(intercept_/coef_[1])

x_input1 = np.linspace(-3,3,100)
y_input1 = m2*x_input1 + b2

plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,label='Sklearn Logistic Regression',color='red')
plt.plot(x_input1,y_input1,label='Gradient Descent Logistic Regression',color='blue')
plt.legend()
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
plt.ylim(-3,3)
plt.show()

"""In this code we have used batch gradient descent to optimize the weights of the logistic regression model. i.e one update is done after calculating the average gradient over the entire training dataset. This approach can be computationally expensive for large datasets, but it provides a stable convergence towards the optimal weights."""
