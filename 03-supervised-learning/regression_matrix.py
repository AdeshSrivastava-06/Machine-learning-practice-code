import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('placement.csv')

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

plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

y_pred = lr.predict(X_test)

print("MAE:--",mean_absolute_error(y_test,y_pred))
print("MSE:--",mean_squared_error(y_test,y_pred))
print("RMSE:--",np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2-score:--",r2_score(y_test,y_pred))

r2 = r2_score(y_test,y_pred)
"""adjusted R2 score"""

aR2 = 1 - ((1 - r2)*(40 - 1)/(40-1-1))
print("Adjusted R2 score:--",aR2)

new_df1 = df.copy()
new_df1['random_feature'] = np.random.random(200)

new_df1 = new_df1[['cgpa','random_feature','package']]
new_df1.head()

plt.scatter(new_df1['random_feature'],new_df1['package'])
plt.xlabel('random_feature')
plt.ylabel('Package(in lpa)')
plt.show()


X = new_df1.iloc[:,0:2]
y = new_df1.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lr = LinearRegression()
lr.fit(X_train,y_train)
LinearRegression()
y_pred = lr.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

adR2 = 1 - ((1-r2)*(40-1)/(40-1-2))

print(adR2)


new_df2 = df.copy()

new_df2['iq'] = new_df2['package'] + (np.random.randint(-12,12,200)/10)

new_df2 = new_df2[['cgpa','iq','package']]

plt.scatter(new_df2['iq'],new_df2['package'])
plt.xlabel('iq')
plt.ylabel('Package(in lpa)')
plt.show()

X = new_df2.iloc[:,0:2]
y = new_df2.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

adjr2 = 1 - ((1-r2)*(40-1)/(40-1-2))

print(adjr2)
