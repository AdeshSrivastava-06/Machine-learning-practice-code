import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data_science_job.csv')

print(df.head())

print(df.isnull().mean()*100)
"""here we will not remove all the cols but only that cols whose numm value % is less than 5% according to the CCA principle"""

cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]

print(cols)

"""now we will see if we drop all the null cols then how much % data will be left"""

print(len(df[cols].dropna()) / len(df))

new_df = df[cols].dropna()

"""to check if we have done the right thing by removing the null cols then we can plot a histogram """

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['training_hours'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['training_hours'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['training_hours'].plot.density(color='red')

# data after cca
new_df['training_hours'].plot.density(color='green')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['city_development_index'].plot.density(color='red')

# data after cca
new_df['city_development_index'].plot.density(color='green')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['city_development_index'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['city_development_index'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)
plt.show()

temp = pd.concat([
    #% of obervations per category in origignal data
    df['enrolled_university'].value_counts() / len(df),
    
    new_df['enrolled_university'].value_counts() / len(new_df)
],axis=1)

temp.columns = ['original','cca']
print(temp)

temp = pd.concat([
    #% of obervations per category in origignal data
    df['education_level'].value_counts() / len(df),
    
    new_df['education_level'].value_counts() / len(new_df)
],axis=1)

temp.columns = ['original','cca']
print(temp)
