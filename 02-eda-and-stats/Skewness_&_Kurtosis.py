"""
Skewness & Kurtosis

Measures distribution shape for numeric features:
- Skewness: asymmetry (0 = symmetric, +ve = right tail, -ve = left tail)
- Kurtosis: tailedness (0 = normal/mesokurtic using Fisher's definition,
  +ve = heavy tails/leptokurtic, -ve = light tails/platykurtic)
"""

import pandas as pd
from scipy.stats import skew, kurtosis

df = pd.read_csv("train.csv")
numeric_cols = df.select_dtypes(include="number").columns

results = []
for col in numeric_cols:
    data = df[col].dropna()
    sk = skew(data)
    ku = kurtosis(data)  # Fisher's definition (normal = 0)

    if abs(sk) < 0.5:
        skew_label = "approximately symmetric"
    elif abs(sk) < 1:
        skew_label = "moderately skewed"
    else:
        skew_label = "highly skewed"

    if ku > 0.5:
        kurt_label = "heavy-tailed (leptokurtic)"
    elif ku < -0.5:
        kurt_label = "light-tailed (platykurtic)"
    else:
        kurt_label = "normal-tailed (mesokurtic)"

    results.append({
        "feature": col,
        "skewness": round(sk, 3),
        "skew_type": skew_label,
        "kurtosis": round(ku, 3),
        "kurtosis_type": kurt_label
    })

result_df = pd.DataFrame(results).sort_values("skewness", key=abs, ascending=False)
print(result_df.to_string(index=False))

result_df.to_csv("skew_kurtosis_report.csv", index=False)
