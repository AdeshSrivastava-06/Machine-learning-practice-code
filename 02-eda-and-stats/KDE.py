"""KDE is a non-paramertric density estimation it uses kernel->probalility distribution(mainly the normal dist) and then for every data point it plots a normal curve of a specific bandwith(std) and above the points the no.of curvers decides its position by adding the values og Y"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal

samp_1 = normal(loc=20,scale=5,size=300)
samp_2 = normal(loc=40,scale=5,size=700)
sample = np.hstack((samp_1,samp_2))

#plot histogram bins=50

plt.hist(sample,bins=50)
plt.show()

from sklearn.neighbors import KernelDensity

model = KernelDensity(bandwidth=3,kernel="gaussian")

"""bandwidth is the std of the normal distribution which is used to plot the curve for every data point and then add them to get the final curve"""
"""kernel - guassian is the most common kernel used in KDE but there are other kernels like tophat,epanechnikov,exponential etc."""

sample = sample.reshape((len(sample),1))
model.fit(sample)

values = np.linspace(sample.min(),sample.max(),100)
values = values.reshape((len(values),1))

probabilities = model.score_samples(values)
"""score_sample reaturn the probability values in log therefore we use exponential to cut that log"""
probabilities = np.exp(probabilities)


plt.hist(sample,bins=50,density=True)
plt.plot(values[:],probabilities)
plt.show()


"""same thing we can do with seaborn"""

import seaborn as sns

sns.kdeplot(sample.reshape(1000),bw_adjust=1.5,bw_method="silverman")
plt.show()
