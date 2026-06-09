from sklearn.datasets import make_regression

import numpy as np

x,y = make_regression(n_samples=4,n_features=1,n_informative=1,n_targets=1,noise=80,random_state=13)

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.show()

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x,y)

print(f"reg.coef_: {reg.coef_}")
print(f"reg.intercept_: {reg.intercept_}")

plt.scatter(x,y)
plt.plot(x,reg.predict(x),color = "red")
plt.show()

"""lets apply gradient descent assuming slope is constant m = 78.35 and lets assume the starting value for the intercept is b = 0"""

"""1st iteration"""
y_pred = ((78.85 * x)  + 0).reshape(4)

plt.scatter(x,y)
plt.plot(x,reg.predict(x),color="red",label='OLS')
plt.plot(x,y_pred,color='blue',label='b=0')
plt.legend()
plt.show()

m = 78.35
b = 0

loss_slope = -2*np.sum(y - m*x.ravel() - b)
print(f"loss-slope-- {loss_slope}")

learning_rate = 0.1

step_size = loss_slope *learning_rate

print(f"step-size-- {step_size}")

"""calculating the new intercept"""
b = b - step_size
print(f"new intercept-- {b}")


y_pred1 = ((78.35 * x) + b).reshape(4)

plt.scatter(x,y)
plt.plot(x,reg.predict(x),color='red',label='OLS')
plt.plot(x,y_pred1,color='#00a65a',label='b = {}'.format(b))
plt.plot(x,y_pred,color='#A3E4D7',label='b = 0')
plt.legend()
plt.show()

"""iteration 2"""

loss_slope = -2 * np.sum(y - m*x.ravel() - b)

step_size = loss_slope*learning_rate

b = b - step_size
print(f"new intercept-- {b}")
y_pred2 = ((78.35 * x) + b).reshape(4)

plt.scatter(x,y)
plt.plot(x,reg.predict(x),color='red',label='OLS')
plt.plot(x,y_pred2,color='#00a65a',label='b = {}'.format(b))
plt.plot(x,y_pred1,color='#A3E4D7',label='b = {}'.format(b))
plt.plot(x,y_pred,color='#A3E4D7',label='b = 0')
plt.legend()
plt.show()

"""iteration 3"""

loss_slope = -2 * np.sum(y - m*x.ravel() - b)
step_size = loss_slope*learning_rate

b = b - step_size
print(f"3rd b-- {b}")


y_pred3 = ((78.35 * x) + b).reshape(4)

plt.figure(figsize=(12,8))
plt.scatter(x,y)
plt.plot(x,reg.predict(x),color='red',label='OLS')
plt.plot(x,y_pred3,color='#00a65a',label='b = {}'.format(b))
plt.plot(x,y_pred2,color='#A3E4D7',label='b = {}'.format(b))
plt.plot(x,y_pred1,color='#A3E4D7',label='b = {}'.format(b))
plt.plot(x,y_pred,color='#A3E4D7',label='b = 0')
plt.legend()
plt.show()


b = -100
m = 78.35
lr = 0.1

epochs = 10

for i in range(epochs):
  loss_slope = -2 * np.sum(y - m*x.ravel() - b)
  b = b - (lr * loss_slope)

  y_pred = m * x + b

  plt.plot(x,y_pred)

plt.scatter(x,y)
plt.show()

"""The above code represents the GD process for calculating the intercept b. We can also calculate the slope m using the same process. but here we have assumed the slope to be constant."""