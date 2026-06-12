# Data Preprocessing

Data preprocessing is the process of cleaning and transforming raw data into a format suitable for model training. Real-world data is often incomplete, inconsistent, or noisy, and feeding it directly into a model leads to poor results.

---

**Handling Missing Values**

- Drop rows or columns if the missing percentage is very high.
- Fill with mean or median for numerical columns.
- Fill with mode for categorical columns.
- Use forward fill or backward fill for time series data.

---

**Handling Outliers**

- Detect using IQR method or Z-score.
- Either remove them, cap them at a boundary value, or transform the feature to reduce their impact.

---

**Encoding Categorical Variables**

- Label Encoding — assigns a number to each category. Suitable for ordinal data.
- One-Hot Encoding — creates a binary column for each category. Suitable for nominal data.

---

**Feature Scaling**

- Normalization (Min-Max Scaling) — scales values to a range of 0 to 1. Used when the distribution is not Gaussian.
- Standardization (Z-score Scaling) — scales values to have mean 0 and standard deviation 1. Used when the distribution is approximately Gaussian.

---

**Handling Imbalanced Data**

- Oversampling the minority class using techniques like SMOTE.
- Undersampling the majority class.
- Using class weights in the model.

---

**Train-Test Split**

Splitting data into training and testing sets before any preprocessing to avoid data leakage. A common split is 80-20 or 70-30.

---

Preprocessing directly impacts model accuracy and generalization. A well-preprocessed dataset often matters more than the choice of algorithm.
