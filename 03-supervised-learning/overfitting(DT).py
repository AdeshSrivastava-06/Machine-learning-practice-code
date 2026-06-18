import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def analyzer(max_depth):
    data = pd.read_csv('Social_Network_Ads.csv')
    X = data.iloc[:,2:4].values
    y = data.iloc[:,-1].values
    
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X,y)
    
    a = np.arange(start=X[:,0].min()-1, stop=X[:,0].max()+1, step=0.1)
    b = np.arange(start=X[:,1].min()-1, stop=X[:,1].max()+1, step=100)
    
    a,b = np.meshgrid(a,b)
    
    plt.contourf(a,b,clf.predict(np.array([a.ravel(),b.ravel()]).T).reshape(a.shape), alpha=0.75, cmap=plt.cm.brg)
    
    plt.scatter(X[:,0],X[:,1],c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.brg)
    
    plt.title('Decision Tree Classifier (max_depth={})'.format(max_depth))  
    
    plt.show()
    
analyzer(max_depth=None)

"""this code demonstrates overfitting in a Decision Tree Classifier by setting max_depth to None, which allows the tree to grow until all leaves are pure. The resulting decision boundary is very complex and may not generalize well to unseen data, indicating overfitting."""