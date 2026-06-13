"""to check if a given dist is a pareto distribution we amke a log-log plot plot
pareto->its a special type of distribution that follows or exhibit a power law(its a func relationship betwnne 2 variables i.e y = k*x^a) use to model dist of wealth,income ,population spread"""

import numpy as np
import matplotlib.pyplot as plt

#define the parameters of the pareto dist
alpha = 3
xm = 1

#create an array of x values
x = np.linspace(0.1,10,1000)

#create the y-values of the pareto dist
y = alpha * (xm**alpha) / (x**(alpha+1))

plt.plot(x,y)
plt.show()

#create the log-log plot
plt.plot(np.log(x),np.log(y))

#add labels
plt.xlabel('X')
plt.ylabel('P(x)')
plt.title('Pareto distribution')
plt.show()

"""using QQ-plot"""


import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

alpha = 2
xm = 1

#generate a set of random data from the pareto dist

x = stats.pareto.rvs(b=alpha,scale=xm,size=1000)

plt.hist(x)
plt.show()

params = stats.pareto.fit(x,floc=0)

#create a pareto dist object
dist = stats.pareto(b=params[0],scale=params[2])

#create a QQ-plot
fig = sm.qqplot(x,dist=dist,line='45')

plt.title('QQ-plot of pareto dist')
plt.xlabel('Theoretical quantiles')
plt.ylabel('sample quantiles')
plt.show()

