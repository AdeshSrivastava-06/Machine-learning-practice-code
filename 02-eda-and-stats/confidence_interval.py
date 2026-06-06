import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

df['Fare'].plot(kind='kde')
plt.show()

samples = []
sample_std = []

for i in range(50):
    x = df['Fare'].dropna().sample(30).values
    sample_std.append(x.std())
    samples.append(x.tolist())
    
sample = np.array(samples) 

samp_means = sample.mean()

samp_std = np.mean(sample_std)

lower_limit = samp_means.mean() - 2.042*(samp_std/np.sqrt(30))
upper_limit = samp_means.mean() + 2.042*(samp_std/np.sqrt(30))

print(f"lower-limit:- {lower_limit} -- upper limit:-{upper_limit}")

print(samp_means)

#the above code returns the confidence interval for the mean of the 'Fare' column in the dataset using the standard deviation of the samples. The t-value used is 2.042, which corresponds to a 95% confidence level with 29 degrees of freedom (since we are using a sample size of 30). The confidence interval is calculated using the formula:
#CI = mean ± t * (std / sqrt(n))


"""other method"""
import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

samples = []
sample_means = []

for i in range(50):
    x = df['Fare'].dropna().sample(30).values
    sample_means.append(x.mean())
    samples.append(x)

sample_means = np.array(sample_means)

mean_of_means = sample_means.mean()
std_of_means = sample_means.std(ddof=1)

t = 2.042  # 95% CI, df = 29

lower_limit = mean_of_means - t * (std_of_means)
upper_limit = mean_of_means + t * (std_of_means)

print(f"Lower limit: {lower_limit}")
print(f"Upper limit: {upper_limit}")

print(mean_of_means)

# In this method, we calculate the mean of the sample means and the standard deviation of the sample means. The confidence interval is then calculated using the formula:
# CI = mean_of_means ± t * std_of_means