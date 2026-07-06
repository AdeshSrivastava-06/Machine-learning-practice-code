import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)

df['species'] = data.target

print(df.columns)

encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

print(df.head())

import seaborn as sns
sns.pairplot(df, hue='species')
import matplotlib.pyplot as plt
plt.show()

new_df = df[df['species'] != 0][['sepal length (cm)', 'sepal width (cm)','species']]

x = new_df.iloc[:,0:1]
y = new_df.iloc[:,-1]

print(new_df.head())

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = KNeighborsClassifier()

estimators = [('lr',clf1),('rf',clf2),('knn',clf3)]

for estimator in estimators:
    scores = cross_val_score(estimator[1],x,y,cv=10,scoring='accuracy')
    print(estimator[0],np.round(np.mean(scores),2))

from sklearn.ensemble import VotingClassifier

"""Hard Voting Classifier"""
vc = VotingClassifier(estimators=estimators,voting='hard')
scores1 = cross_val_score(vc,x,y,cv=10,scoring='accuracy')
print('Hard Voting Classifier',np.round(np.mean(scores1),2))

"""Soft Voting Classifier"""
vc = VotingClassifier(estimators=estimators,voting='soft')
scores2 = cross_val_score(vc,x,y,cv=10,scoring='accuracy')  
print('Soft Voting Classifier ',np.round(np.mean(scores2),2))

"""hyperparameter weights- giving more weight to the model which is performing better"""
vc = VotingClassifier(estimators=estimators,voting='soft',weights=[9,2,1])
scores3 = cross_val_score(vc,x,y,cv=10,scoring='accuracy')
print('Soft Voting Classifier with weights',np.round(np.mean(scores3),2))

"""Diffrence bet soft and hard voting- in hard voting the final prediction is based on majority vote but in soft voting the final prediction is based on the average of predicted probabilities of each class by each model"""


"""classifiers of same algorithm but with different hyperparameters"""

from sklearn.datasets import make_classification

X1, y1 = make_classification(n_samples=1000, n_features=20, n_informative=15,n_redundant=5, random_state=2)

from sklearn.svm import SVC #SVC = Support Vector Classifier it means that we are using SVM algorithm for classification task. SVM is a supervised machine learning algorithm that can be used for both classification and regression tasks. It works by finding the hyperplane that best separates the data into different classes.

svm1 = SVC(probability=True, kernel='poly',degree=1)
svm2 = SVC(probability=True, kernel='poly',degree=2)
svm3 = SVC(probability=True, kernel='poly',degree=3)
estimators = [('svm1',svm1),('svm2',svm2),('svm3',svm3)]

for estimator in estimators:
    scores = cross_val_score(estimator[1],X1,y1,cv=10,scoring='accuracy')
    print(estimator[0],np.round(np.mean(scores),2))
    
vc = VotingClassifier(estimators=estimators,voting='soft')
scores4 = cross_val_score(vc,X1,y1,cv=10,scoring='accuracy')
print('Soft Voting Classifier with different hyperparameters',np.round(np.mean(scores4),2))
