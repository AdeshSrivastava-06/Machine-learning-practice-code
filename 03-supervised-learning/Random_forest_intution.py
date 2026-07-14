import pandas as pd 
import numpy as np 
from sklearn.datasets import make_classification
import random

X,y = make_classification(n_features=5,n_redundant=0,n_informative=5,n_clusters_per_class=1,n_samples=100) #by default rows are 100

df = pd.DataFrame(X,columns=['col1','col2','col3','col4','col5'])
df['target'] = y
print(df.shape)
print(df.head())

#function for row sampling

def sample_row(df,percent):
    return df.sample(int(percent*df.shape[0]))

def sample_features(df,percent):
    cols = random.sample(df.columns.tolist()[:,-1],int(percent*df.shape[1]-1))
    return df[cols]

def combined_sampling(df,row_percent,col_percent):
    new_df = sample_row(df,row_percent)
    return sample_features(new_df,col_percent)

df1 = sample_row(df,0.2)
df2 = sample_row(df,0.2)
df3 = sample_row(df,0.2)

from sklearn.tree import DecisionTreeClassifier

clf1 = DecisionTreeClassifier()
clf2 = DecisionTreeClassifier()
clf3 = DecisionTreeClassifier()

clf1.fit(df1.iloc[:,0:5],df1.iloc[:,-1])
clf2.fit(df2.iloc[:,0:5],df2.iloc[:,-1])
clf3.fit(df3.iloc[:,0:5],df3.iloc[:,-1])


print(clf1.predict(np.array([0.56,-1.67,-0.65,1.45,1.23]).reshape(1,5)))
print(clf2.predict(np.array([0.56,-1.67,-0.65,1.45,1.23]).reshape(1,5)))
print(clf3.predict(np.array([0.56,-1.67,-0.65,1.45,1.23]).reshape(1,5)))
