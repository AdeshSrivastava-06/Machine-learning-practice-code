import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(LogisticRegression)
df = sns.load_dataset('iris')

encode = LabelEncoder()
df['species'] = encode.fit_transform(df['species'])

df = df[['sepal_length','petal_length','species']]

x = df.iloc[:,0:2]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

clf = LogisticRegression(solver='lbfgs', max_iter=1000)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test,y_pred))

query = np.array([[3.4,2.2]])
print(clf.predict_proba(query))
print(clf.predict(query))

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(x.values, y.values, clf, legend=2)

# Adding axes annotations
plt.xlabel('sepal length [cm]')
plt.xlabel('petal length [cm]')
plt.title('Softmax on Iris')

plt.show() 

#Solver is an optimization algorithm that is used to find the best parameters for the model. In this case, we are using lbfgs which is a quasi-Newton method that is used for optimization. It is a good choice for small datasets and it is also faster than other solvers. We also set max_iter to 1000 which is the maximum number of iterations that the solver will run before stopping.
#Its called softmax Logistic regression because it is used for multi-class classification problems. It uses the softmax function to calculate the probabilities of each class and then it selects the class with the highest probability.
