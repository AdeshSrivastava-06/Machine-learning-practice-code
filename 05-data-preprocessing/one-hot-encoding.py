import numpy as np
import pandas as pd

df = pd.read_csv('cars.csv')

print(df.head())

print(f"Value counts for 'brand': {df['brand'].value_counts().to_string()}")

print(f"Value counts for 'fuel': {df['fuel'].value_counts().to_string()}")

print(f"Value counts for 'owner': {df['owner'].value_counts().to_string()}")

"""one-hot encoding using pandas"""

print(f"One-hot encoded 'fuel' and 'owner': {pd.get_dummies(df,columns=['fuel','owner']).to_string()}")

"""k-1 encoding"""
print(f"K-1 encoded 'fuel' and 'owner': {pd.get_dummies(df,columns=['fuel','owner'],drop_first=True).to_string()}")


"""using skleran"""
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,0:4],df.iloc[:,-1],test_size=0.2,random_state=0)

from sklearn.preprocessing import OneHotEncoder

ohe =OneHotEncoder(drop='first',sparse_output=False,dtype=np.int32)

x_train_new = ohe.fit_transform(x_train[['fuel','owner']])

print(f"One-hot encoded 'fuel' and 'owner' in training set: {x_train_new}")

x_test_new = ohe.transform(x_test[['fuel','owner']])

print(f"One-hot encoded 'fuel' and 'owner' in test set: {x_test_new}")

"""now we will combine the  numerical columns with the one-hot encoded columns"""

final_df = np.hstack((x_train[['brand','km_driven']].values, x_train_new))

num_cols = ['brand', 'km_driven']
ohe_cols = ohe.get_feature_names_out(['fuel','owner'])
all_cols = np.concatenate([num_cols, ohe_cols])

final_df1 = pd.DataFrame(final_df, columns=all_cols)
print(f"Final DataFrame with one-hot encoded columns: {final_df1.head()}")


"""one-hot encoding with top categories so here i am taking the threshold as 100 if the cars count below 100 will fall under others category"""

counts = df['brand'].value_counts()
threshold = 100

repl = counts[counts <= threshold].index
print(f"Categories to be replaced: {repl}")

brand_dummy = pd.get_dummies(df['brand'].replace(repl,'uncommon'))
print(f"One-hot encoded 'brand' with top categories: {brand_dummy.to_string()}")