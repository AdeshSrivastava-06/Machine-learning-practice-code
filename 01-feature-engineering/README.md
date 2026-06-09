# Feature Engineering
Feature engineering is the process of using domain knowledge to select, transform, or create new input variables (features) from raw data to improve model performance. It is often considered one of the most important and time-consuming steps in the machine learning pipeline.

Raw data rarely comes in a form that a model can directly learn from effectively. Feature engineering bridges that gap by making the underlying patterns more visible to the algorithm.

**Common techniques include:**

- **Handling missing values** — filling them with mean, median, mode, or a constant, or dropping them entirely depending on context.
- **Encoding categorical variables** — converting text categories into numbers using label encoding or one-hot encoding.
- **Feature scaling** — normalizing or standardizing numerical features so they are on a comparable scale, which matters for algorithms like SVM or KNN.
- **Feature creation** — combining existing features to create new ones, for example extracting day, month, and year from a date column.
- **Binning** — grouping continuous values into discrete buckets, like converting age into age groups.
- **Handling skewness** — applying log or square root transformations to reduce skew in numerical distributions.
- **Dropping irrelevant features** — removing columns with no predictive value or very high missing rates.
