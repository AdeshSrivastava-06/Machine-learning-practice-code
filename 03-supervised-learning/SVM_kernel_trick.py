import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d

from sklearn.datasets._samples_generator import make_circles

X,y = make_circles(n_samples=100, factor=.1, noise=.1)
plt.scatter(X[:,0], X[:,1], c=y,s=50,cmap='autumn')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

zero_one_colourmap = ListedColormap(('blue', 'red'))
def plot_decision_boundary(X, y, clf):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                                 stop = X_set[:, 0].max() + 1, 
                                 step = 0.01),
                       np.arange(start = X_set[:, 1].min() - 1, 
                                 stop = X_set[:, 1].max() + 1, 
                                 step = 0.01))
  
    plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), 
                                             X2.ravel()]).T).reshape(X1.shape),
               alpha = 0.75, 
               cmap = zero_one_colourmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = (zero_one_colourmap)(i), label = j)
    plt.title('SVM Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    return plt.show()

plot_decision_boundary(X, y, model)

def plot_3d_plot(x,y):
    r = np.exp(-(x**2).sum(1)) # RBF kernel transformation
    ax = plt.subplot(projection='3d')
    ax.scatter(x[:,0], x[:,1], r, c=y, cmap='autumn',s=100)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    plt.show()
    
plot_3d_plot(X,y)

"""now we will train the SVM with the RBF kernel and see how it performs"""

model_rbf = svm.SVC(kernel='rbf', gamma='scale')
model_rbf.fit(x_train, y_train)
y_pred_rbf = model_rbf.predict(x_test)
print("RBF Kernel Accuracy:", accuracy_score(y_test, y_pred_rbf))
plot_decision_boundary(X, y, model_rbf)

"""we can aslo use polynomial kernel it means that we will transform the data into a higher dimensional space using polynomial features just like logistic regression"""

model_poly = svm.SVC(kernel='poly', degree=2, gamma='scale')
model_poly.fit(x_train, y_train)
y_pred_poly = model_poly.predict(x_test)
print("Polynomial Kernel Accuracy:", accuracy_score(y_test, y_pred_poly))

plot_decision_boundary(X, y, model_poly)

"""we can see that the RBF kernel and polynomial kernel both perform much better than the linear kernel on this dataset because they can capture the non-linear relationships between the features and the target variable. The RBF kernel is particularly effective for this dataset because it can capture the circular decision boundary, while the polynomial kernel can also capture non-linear relationships but may not be as effective as the RBF kernel in this case.
RBF=Radial Basis Function, it is a popular kernel function used in SVMs that can capture non-linear relationships between features and the target variable. It works by transforming the data into a higher-dimensional space where it can find a linear decision boundary that separates the classes effectively. The RBF kernel is particularly effective for datasets with circular or spherical decision boundaries, as it can capture the complex relationships between the features and the target variable."""
