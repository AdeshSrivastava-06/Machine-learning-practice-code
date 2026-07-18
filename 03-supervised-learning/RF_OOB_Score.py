import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

df = pd.read_csv('heart.csv')
x = df.iloc[:,0:-1] #iloc 0 to -1 means all rows and all columns except last one
y = df.iloc[:,-1] #iloc -1 means all rows and last column

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
rf = RandomForestClassifier(oob_score=True, random_state=42)

rf.fit(X_train,y_train)
print("OOB Score:", rf.oob_score_)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print(rf.feature_importances_)
