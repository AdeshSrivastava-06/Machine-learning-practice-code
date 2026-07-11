from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

housing = fetch_california_housing()
X, y = housing.data, housing.target

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

lr= LinearRegression()
dt= DecisionTreeRegressor()


estimators= [('lr',lr),('dt',dt)]

for estimator in estimators:
    scores= cross_val_score(estimator[1],X,y,cv=10,scoring='r2')
    print(estimator[0],np.round(np.mean(scores),2))
    
from sklearn.ensemble import VotingRegressor
"""Voting Regressor"""
vr= VotingRegressor(estimators=estimators)
scores1= cross_val_score(vr,X,y,cv=10,scoring='r2')
print('Voting Regressor',np.round(np.mean(scores1),2))

"""hyperparameter weights- giving more weight to the model which is performing better"""

vr= VotingRegressor(estimators=estimators,weights=[9,3])
scores2= cross_val_score(vr,X,y,cv=10,scoring='r2')
print('Voting Regressor',np.round(np.mean(scores2),2))
