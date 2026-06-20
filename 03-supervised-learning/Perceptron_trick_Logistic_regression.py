from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data = load_breast_cancer()
X = data.data
y = data.target

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X = pca.fit_transform(X)

plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',edgecolors='k',s=100)
plt.title('Breast Cancer Dataset (PCA Reduced)')
plt.show()

# USING PERCEPTRON (STEP FUNCTION)

def step(z):
    return 1 if z >= 0 else 0

def perceptron(X,y):
    
    X = np.insert(X,0,1,axis=1)
    weights = np.zeros(X.shape[1])
    lr = 0.01
    
    for _ in range(1000):
        
        j = np.random.randint(0,X.shape[0])
        y_hat = step(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]
        
    return weights

weights_p = perceptron(X,y)

intercept_p = weights_p[0]
coef_p = weights_p[1:]

m_p = -(coef_p[0]/coef_p[1])
b_p = -(intercept_p/coef_p[1])

x_input = np.linspace(X[:,0].min(),X[:,0].max(),100)
y_input = m_p*x_input + b_p


# SKLEARN LOGISTIC REGRESSION

# sklearn logistic reg used sigmoid + cross entropy loss function to find the optimal decision boundary

lor = LogisticRegression()
lor.fit(X,y)

coef_lr = lor.coef_[0]
intercept_lr = lor.intercept_[0]

m_lr = -(coef_lr[0]/coef_lr[1])
b_lr = -(intercept_lr/coef_lr[1])

y_input_lr = m_lr*x_input + b_lr


# ---- SGD STYLE LOGISTIC REGRESSION(USING ONLY SIGMOID FUNCTION)

# sigmoid function is used in logistic regression to map predicted values to probabilities

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sgd_logistic(X,y):
    
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1
    
    for _ in range(1000):
        
        j = np.random.randint(0,X.shape[0])
        y_hat = sigmoid(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]
        
    return weights

weights_sgd = sgd_logistic(X,y)

intercept_sgd = weights_sgd[0]
coef_sgd = weights_sgd[1:]

m_sgd = -(coef_sgd[0]/coef_sgd[1])
b_sgd = -(intercept_sgd/coef_sgd[1])

y_input_sgd = m_sgd*x_input + b_sgd


# BATCH GRADIENT DESCENT LOGISTIC REGRESSION

# In this code we have used batch gradient descent to optimize the weights of the logistic regression model.
# i.e one update is done after calculating the average gradient over the entire training dataset.
# This approach can be computationally expensive for large datasets, but it provides stable convergence.

def gd(X,y):
    
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    
    lr = 0.5
    
    for _ in range(5000):
        
        y_hat = sigmoid(np.dot(X,weights))
        weights = weights + lr*(np.dot(X.T,(y-y_hat))/X.shape[0])
        
    return weights

weights_gd = gd(X,y)

intercept_gd = weights_gd[0]
coef_gd = weights_gd[1:]

m_gd = -(coef_gd[0]/coef_gd[1])
b_gd = -(intercept_gd/coef_gd[1])

y_input_gd = m_gd*x_input + b_gd


#  ACCURACY COMPARISON

X_bias = np.insert(X,0,1,axis=1)

pred_p = []
for i in range(len(X)):
    pred_p.append(step(np.dot(X_bias[i],weights_p)))

pred_sgd = []
for i in range(len(X)):
    pred_sgd.append(1 if sigmoid(np.dot(X_bias[i],weights_sgd))>=0.5 else 0)

pred_gd = []
for i in range(len(X)):
    pred_gd.append(1 if sigmoid(np.dot(X_bias[i],weights_gd))>=0.5 else 0)

print("Perceptron Accuracy:",accuracy_score(y,pred_p))
print("Sklearn Logistic Accuracy:",lor.score(X,y))
print("SGD Logistic Accuracy:",accuracy_score(y,pred_sgd))
print("Batch GD Logistic Accuracy:",accuracy_score(y,pred_gd))


#  FINAL DECISION BOUNDARY COMPARISON

plt.figure(figsize=(10,6))

plt.plot(x_input,y_input,color='black',label='Perceptron')
plt.plot(x_input,y_input_lr,color='red',label='Sklearn Logistic Regression')
plt.plot(x_input,y_input_sgd,color='brown',label='SGD Logistic Regression')
plt.plot(x_input,y_input_gd,color='blue',label='Batch GD Logistic Regression')

plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',edgecolors='k',s=100)

plt.title('Comparison of Decision Boundaries')

plt.legend()

plt.show()
