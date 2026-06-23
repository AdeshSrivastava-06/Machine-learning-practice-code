import pandas as pd 
import numpy as np
from scipy import stats
 
df = pd.read_csv('train.csv')

"""
Null Hypo--> the average of males=avg age of females
Alternate Hypo-->avg age of males > avg age of females"""

pop_male = df[df['Sex'] == 'male']['Age'].dropna()

pop_female = df[df['Sex'] == 'female']['Age'].dropna()

sample_male = pop_male.sample(25)
sample_female = pop_female.sample(25)

alpha = 0.05

from scipy.stats import shapiro

# Perform the Shapiro-Wilk test for both desktop and mobile users
shapiro_male = shapiro(sample_male)
shapiro_female = shapiro(sample_female)

print("Shapiro-Wilk test for desktop users:", shapiro_male)
print("Shapiro-Wilk test for mobile users:", shapiro_female)

from scipy.stats import levene

# Perform Levene's test
levene_test = levene(sample_male, sample_female)
print(levene_test)

t_stats,p_value = stats.ttest_ind(sample_male,sample_female)

print("t-stats:--",t_stats)
print("p-value:--",p_value/2)

if p_value < alpha :
    print("Reject the null hypo")
else:
    print("No strong evidence to reject the null hypo")    
    
    
print("male-mean:--",pop_male.mean())
print("female-mean:--",pop_female.mean())    
