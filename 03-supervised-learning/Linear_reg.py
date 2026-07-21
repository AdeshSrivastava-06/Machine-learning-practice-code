import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('placement.csv')

print(df.head())
plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')
plt.show()

X = df.iloc[:,0:1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)

from sklearn.metrics import r2_score

y_pred = lr.predict(X_test)
print(r2_score(y_test,y_pred))

print(lr.predict(X_test.iloc[0].values.reshape(1,1)))

plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')
plt.show() 

slope = lr.coef_ # represents how one quatity is dependent on other quantity i.e weightage
print("slope:",slope)
intercept = lr.intercept_ #it represents the offset
print("intercept:",intercept)

print(slope*8.58 + intercept)

