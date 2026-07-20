import numpy as np
from sklearn.linear_model import Ridge  
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_diabetes

data = load_diabetes()
#print(data.DESCR)

x = data.data
y = data.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score
print("MSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

"""using Ridge Regression"""

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.01)  # You can adjust the alpha value for regularization strength

ridge.fit(x_train, y_train)
y_pred_ridge = ridge.predict(x_test)    

print("MSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print("R2 Score:", r2_score(y_test, y_pred_ridge))



"""making our own data"""

m = 100
x1 = 5 * np.random.rand(m, 1) - 2
x2 = 0.7 * x1 ** 2 - 2 * x1 + 3 + np.random.randn(m, 1)

plt.scatter(x1, x2)
plt.show()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def get_preds_ridge(x1,x2,alpha):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=16)),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(x1, x2)
    return model.predict(x1)

y_pred_ridge = get_preds_ridge(x1, x2, alpha=0.01)
print("MSE-polynomial-ridge:", np.sqrt(mean_squared_error(x2, y_pred_ridge)))
print("R2 Score-polynomial-ridge:", r2_score(x2, y_pred_ridge))

alphas = [0,20,200]
cs = ['r','g','b']

plt.figure(figsize=(12, 8))
plt.plot(x1,x2,'b+',label='Data_points')

for alpha, c in zip(alphas, cs):
    preds = get_preds_ridge(x1, x2, alpha)
    plt.plot(sorted(x1[:, 0]), preds[np.argsort(x1[:, 0])], c, label='Alpha: {}'.format(alpha))
    
plt.legend()
plt.show()

"""here we can see that as we increase the alpha value, the model becomes less complex and fits the data less closely, which can help to prevent overfitting.the blue line (alpha=0) fits the data very closely, while the red line (alpha=20) and green line (alpha=200) are smoother and less sensitive to the noise in the data."""
