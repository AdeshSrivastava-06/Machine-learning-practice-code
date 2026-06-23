import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import seaborn as sns

df = pd.read_csv('train2.csv')[['Age','Pclass','SibSp','Parch','Survived']]

df = df.dropna()
print(df.head())

X = df.iloc[:,0:4]
y = df.iloc[:,-1]

print(np.mean(cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=20)))

"""applying feature cons"""

X['Family_size'] = X['SibSp'] + X['Parch'] + 1 #here ist +1 because i am counting that person also whose par or child was on the titanic 

def myfunc(num):
    if num == 1:
        #alone
        return 0
    elif num >1 and num <=4:
        # small family
        return 1
    else:
        # large family
        return 2
X['Family_type'] = X['Family_size'].apply(myfunc)

X.drop(columns=['SibSp','Parch','Family_size'],inplace=True)

print(X.head())

print(np.mean(cross_val_score(LogisticRegression(),X,y,scoring='accuracy',cv=20)))

"""feature splitting"""
"""its just creating a new feature from existing cols"""

df = pd.read_csv('train2.csv')

df["Title"] = df['Name'].str.split(', ',expand=True)[1].str.split('.',expand=True)[0]

print(df.head())
