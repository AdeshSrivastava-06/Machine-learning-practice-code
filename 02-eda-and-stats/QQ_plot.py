#This code is for generating a Q-Q plot to compare the quantiles of the sepal length of the iris dataset with the quantiles of a normal distribution.


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')

sns.kdeplot(data=df,x="sepal_length")
plt.show()

temp = sorted(df['sepal_length'].tolist())

y_quant = []

for i in range(1,101):
    y_quant.append(np.percentile(temp,i))
    
samples = np.random.normal(loc=0,scale=1,size=1000)

x_quant = []

for i in range(1,101):
    x_quant.append(np.percentile(samples,i))
    
sns.scatterplot(x=x_quant,y=y_quant)
plt.show()

"""directly from statsmodel"""

import statsmodels.api as sm
import matplotlib.pyplot as plt

fig = sm.qqplot(df['sepal_length'],line='45',fit=True)

plt.title('QQ plot')
plt.xlabel('Theoretical quantiles')
plt.ylabel('sample quantiles')
plt.show()

