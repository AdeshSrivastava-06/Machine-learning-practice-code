import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
print(df.columns)

x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,0:-1],df.iloc[:,-1],test_size=0.2,random_state=43)

print(x_train.shape)

scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.fit(x_test)

knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(x_train,y_train)

from sklearn.metrics import accuracy_score

y_pred = knn.predict(x_test)

print(accuracy_score(y_test,y_pred))

"""applying cross validation to find the best k-value"""

scores = []

for i in range(1,16):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    scores.append(accuracy_score(y_test,y_pred))
    print(f"{i} = {scores[-1]}")

