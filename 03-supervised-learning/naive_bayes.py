import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
df = pd.read_csv('play_tennis.csv')
df = df.drop(columns=['day'])
print(df.head())
print(df.describe())

"""When we train our model a lookup table will be generated which will contain the probability of yes or no for each input(in that also for each values of that col)
"""

x=df.drop(columns=['play'])
y=df['play']

le = LabelEncoder()
y = le.fit_transform(y)

cat_cols = x.columns

preprocessor = ColumnTransformer(
    transformers=[
        ('cat',OrdinalEncoder(),cat_cols)
    ]
)

Pipeline1 = Pipeline([
    ('preprocessor',preprocessor),
    ('model',CategoricalNB())
])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

Pipeline1.fit(x_train,y_train)

y_pred = Pipeline1.predict(x_test)
print(accuracy_score(y_test,y_pred))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(Pipeline1,x,y,cv=5)

print(scores)
print("Average accuracy:",scores.mean())


#The logic of the Algo is that it will calculate the probability of each class for each input and then it will multiply the probabilities of each input and then it will select the class with the highest probability.