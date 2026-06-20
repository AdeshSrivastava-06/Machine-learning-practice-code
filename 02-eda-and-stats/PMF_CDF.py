import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(4)

L = []
for i in range(10000):
    L.append(random.randint(1,6))
    
print(len(L))
print(L[:6])

s = (pd.Series(L).value_counts()/pd.Series(L).value_counts().sum()).sort_index()

print(s)

s.plot(kind="bar")
plt.show()

""" now for 2 dice"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(4)

L = []
for i in range(10000):
    a = random.randint(1,6)
    b = random.randint(1,6)
    L.append(a + b)
    
print(len(L))
print(L[:6])

s = (pd.Series(L).value_counts()/pd.Series(L).value_counts().sum()).sort_index()

print(s)

s.plot(kind="bar")
plt.show()




"""PMF to CDF(it is same as the pdf but shows the probability of occurence of all the possble outcomes p(x <= 5))"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(4)

L = []
for i in range(10000):
    a = random.randint(1,6)
    b = random.randint(1,6)
    L.append(a + b)
    
print(len(L))
print(L[:6])

s = (pd.Series(L).value_counts()/pd.Series(L).value_counts().sum()).sort_index()

a = np.cumsum(s)
print(a)
 
a.plot(kind="bar")
plt.show()
 
s.plot(kind="bar")
plt.show()


#The above code is for the PMF and CDF of the sum of two dice rolls. The PMF (Probability Mass Function) shows the probability of each possible outcome (from 2 to 12), while the CDF (Cumulative Distribution Function) shows the cumulative probability of outcomes up to a certain value.
