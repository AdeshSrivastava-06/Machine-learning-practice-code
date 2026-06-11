**Exploratory Data Analysis (EDA)** is the process of analyzing and summarizing a dataset before building any model. The goal is to understand the data's structure, spot patterns, detect anomalies, and check assumptions using statistics and visualizations.

---

**Descriptive Statistics**

The first step is understanding basic properties of each column.

- **Mean, Median, Mode** — measures of central tendency that tell you where the data is centered.
- **Variance and Standard Deviation** — measure how spread out the values are around the mean.
- **Min, Max, Range** — give a sense of the boundaries of the data.
- **Percentiles and IQR** — useful for understanding distribution and detecting outliers. IQR is the range between the 25th and 75th percentile.

---

**Data Quality Checks**

- Checking for missing values and their percentage per column.
- Identifying duplicate rows.
- Checking data types — whether numerical columns are stored as strings, etc.

---

**Univariate Analysis**

Looking at one variable at a time. For numerical features, histograms and box plots reveal distribution and outliers. For categorical features, bar charts show frequency of each category.

---

**Bivariate and Multivariate Analysis**

Looking at relationships between two or more variables. Scatter plots show correlation between two numerical features. A correlation matrix with a heatmap shows how all numerical features relate to each other. Box plots by category show how a numerical feature varies across groups.

---

**Key Statistical Concepts**

- **Skewness** — measures asymmetry of a distribution. Positive skew means a long right tail, negative skew means a long left tail.
- **Kurtosis** — measures how heavy or light the tails of a distribution are compared to a normal distribution.
- **Correlation** — a value between -1 and 1 indicating the linear relationship between two variables. Closer to 1 or -1 means strong relationship, near 0 means weak.
- **Outliers** — data points that lie far from the rest. Detected using IQR method or Z-score.
