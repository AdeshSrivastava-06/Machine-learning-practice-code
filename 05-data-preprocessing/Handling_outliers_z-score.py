import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('placement.csv')

print(df.sample(5))

plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(df['cgpa'])

plt.subplot(122)
sns.distplot(df['placement_exam_marks'])

plt.show()

print("Mean value of cgpa",df['cgpa'].mean())
print("Std value of cgpa",df['cgpa'].std())
print("Min value of cgpa",df['cgpa'].min())
print("Max value of cgpa",df['cgpa'].max())

Highest_boundary = df['cgpa'].mean() + 3*df['cgpa'].std()

Lowest_boundary = df['cgpa'].mean() - 3*df['cgpa'].std()

print("Highest Boundary--",Highest_boundary)
print("Lowest Boundary--",Lowest_boundary)

print(df[(df['cgpa'] > Highest_boundary) | (df['cgpa'] < Lowest_boundary)])

"""now we will remove the oulier"""
"""trimming"""

new_df = df[(df['cgpa'] > 5.11) & (df['cgpa'] < 8.80)]

print(new_df)



"""approch 2-> calculate the z-score"""

df['cgpa_zcore'] = (df['cgpa'] - df['cgpa'].mean()) / df['cgpa'].std()

print(df.head())

"""here the limits of the z-score will be -3 to 3 because the data is normally distributed """

print(df[df['cgpa_zcore'] > 3])
print(df[df['cgpa_zcore'] < -3])

new_df = df[(df['cgpa_zcore'] > -3) & (df['cgpa_zcore'] < 3)]
print(new_df)



"""capping them"""

df['cgpa'] = np.where(
    df['cgpa'] > Highest_boundary,Highest_boundary,
    np.where(
        df['cgpa'] < Lowest_boundary,
        Lowest_boundary,
        df['cgpa']
    )
)
"""in np.where we provide 3conditions 1-> condition, 2-> if the condition is true then by what value to replace it , 3->If its false then what to do"""

print(df)
print(df.describe())
