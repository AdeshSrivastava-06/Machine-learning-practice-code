import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv('MNIST.csv')
print(df.head())
print("Shape of the data:-" ,df.shape) #the dataset consists of digits in 28 X 28 pixels

plt.imshow(df.iloc[18306, 1:].values.reshape(28,28))
plt.show()

x = df.iloc[:,1:]
y = df.iloc[:,0]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train,y_train)

import time
start = time.time()
y_pred = knn.predict(x_test)

print(time.time() - start)

from sklearn.metrics import accuracy_score

print(f'Accuracy : {accuracy_score(y_test,y_pred)}')

"""now performing PCA"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=300)

x_train_trf = pca.fit_transform(x_train)
x_test_trf = pca.transform(x_test)

print(x_train_trf.shape)

knn = KNeighborsClassifier()
knn.fit(x_train_trf,y_train)
y_pred = knn.predict(x_test_trf)

print(f'Accuracy with PCA: {accuracy_score(y_test,y_pred)}')

#best_accuracy = 0
#best_n_components = 0

#for i in range(1, 785):
#    pca = PCA(n_components=i)
#    x_train_trf = pca.fit_transform(x_train)
#    x_test_trf = pca.transform(x_test)
#    
#    knn = KNeighborsClassifier()
#    knn.fit(x_train_trf, y_train)
#    y_pred = knn.predict(x_test_trf)
#
#    acc = accuracy_score(y_test, y_pred)
#
#    if acc > best_accuracy:
#        best_accuracy = acc
#        best_n_components = i

#print("Highest Accuracy:", best_accuracy)
#print("Best n_components:", best_n_components)

"""transforming to a 2D coordinate system"""

pca = PCA(n_components=2)
x_train_trf = pca.fit_transform(x_train)
x_test_trf = pca.transform(x_test)

import plotly.express as px 

y_train_trf = y_train.astype(str)
fig = px.scatter(x = x_train_trf[:,0],
                 y = x_train_trf[:,1],
                 color=y_train_trf,
                 color_discrete_sequence=px.colors.qualitative.G10)
fig.show()


pca = PCA(n_components=3)
x_train_trf = pca.fit_transform(x_train)
x_test_trf = pca.transform(x_test)

import plotly.express as px 

y_train_trf = y_train.astype(str)
fig = px.scatter_3d(x = x_train_trf[:,0],
                 y = x_train_trf[:,1],
                 z = x_train_trf[:,2],
                 color=y_train_trf)
fig.update_layout(
    margin = dict(l = 20,r=20,t=20,b=20)
)
fig.show()

"""Eigen-values"""
print(pca.explained_variance_) #this value represents how much the pca(components) represents the actula variance of the original data

"""Eigen-vetors"""
print(pca.components_) #shape (3,784)

"""finding the optimum no.of principal components here the Eigen values are used here each componets perecent(lambda1 / (sum of all the lambda) * 100) of Eigen-values is used and represented by lambda and then all the lambda are sequentially added until we get 90%"""

print(pca.explained_variance_ratio_)

pca = PCA(n_components=None)
x_train_trf = pca.fit_transform(x_train)
x_test_trf = pca.transform(x_test)

print(np.cumsum(pca.explained_variance_ratio_))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

"""when PCA does not work
[1]when the variance acress all the axis is approximately the same for all the eigen values
[2]when the both the points are mirror img and the projection of both the data is in the same plane/line
[3]when data follows a certain pattern like for sine curve or y=X^2"""
                
