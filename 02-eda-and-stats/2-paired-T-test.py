import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

before = np.array([80, 92, 75, 68, 85, 78, 73, 90, 70, 88, 76, 84, 82, 77, 91])
after = np.array([78, 93, 81, 67, 88, 76, 74, 91, 69, 88, 77, 81, 80, 79, 88])

differences = after - before
print(differences)


plt.hist(differences)
plt.title("Histogram of Weight Differences")
plt.xlabel("Weight Differences (kg)")
plt.ylabel("Frequency")
plt.show()

shapiro_test = stats.shapiro(differences)
print("Shapiro-Wilk test:", shapiro_test)

mean_diff = np.mean(differences)
std_diff = np.std(differences,ddof=1)

n = len(differences)

t_stats = mean_diff/(std_diff/np.sqrt(n))

df = n-1

alpha = 0.05

p_value = stats.t.cdf(t_stats,df)

print("p-value:--",p_value)

if p_value < alpha:
    print("Reject null hypo")
else:
    print("Not enoung evi ")    
