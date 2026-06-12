
import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split

from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

df = pd.read_csv('train2.csv',usecols=['Age','Pclass','Fare','Survived'])

print(df.head())

x = df.drop(columns=['Survived'])
y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

knn = KNNImputer(n_neighbors=1,weights='distance')

x_train_trf = knn.fit_transform(x_train)
x_test_trf = knn.transform(x_test)

print(pd.DataFrame(x_train_trf,columns=x_train.columns))

lr = LogisticRegression()

lr.fit(x_train_trf,y_train)
y_pred = lr.predict(x_test_trf)

print(f"Accuracy with KNNImputer: {accuracy_score(y_test,y_pred)}")

"""comaprison with the simpleinputer"""

si = SimpleImputer()

X_train_trf2 = si.fit_transform(x_train)
X_test_trf2 = si.transform(x_test)
lr = LogisticRegression()

lr.fit(X_train_trf2,y_train)

y_pred2 = lr.predict(X_test_trf2)

print(f"Accuracy with SimpleImputer: {accuracy_score(y_test,y_pred2)}")

"""The above code is for the knn imputer and simple imputer and we can see that the accuracy of the knn imputer is better than the simple imputer because it uses the distance to impute the missing values whereas the simple imputer uses the mean or median to impute the missing values"""

#It's a data preprocessing technique that is used to impute the missing values in the dataset using the k-nearest neighbors algorithm. It is a multivariate imputation technique that uses the distance between the samples to impute the missing values. It is a more advanced technique than the simple imputer and it can be used to impute the missing values in the dataset more accurately.

#Disadvantages of KNNImputer:
#1. It can be computationally expensive for large datasets.
#2. It can be sensitive to the choice of k and the distance metric.
#3. Here while deploying the model we have to save the knn imputer object and use it to transform the new data before making predictions. other better alternative is to use pipeline and save the pipeline object which will include the knn imputer and the logistic regression model.