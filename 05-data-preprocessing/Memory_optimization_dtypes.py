# Memory optimization by choosing smaller data types.
# Goal: reduce the memory used by a DataFrame, which matters when a CSV file
# is large and does not fit in memory.
# Steps for every CSV file in DATA_FOLDER:
#   1. Integers are stored in the smallest integer type that fits the values
#   2. Floats are stored as float32, which keeps about 7 significant digits
#   3. Text columns with few unique values are stored as the category type
# After the changes, the script reports the largest relative error in the
# numeric columns and checks that the text columns are exactly the same.

from pathlib import Path
import numpy as np
import pandas as pd

# Folder that contains the CSV files. "." means the current folder.
DATA_FOLDER = "."

# float32 is not exact. Set this to False for data that needs full precision,
# such as money values or very precise measurements.
USE_FLOAT32 = True

def is_text(series):
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

def read_csv_safely(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def optimize(df):
    out = df.copy()
    changes = {}
    for col in out.columns:
        before = str(out[col].dtype)
        s = out[col]
        if pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s) and USE_FLOAT32:
            out[col] = s.astype("float32")
        elif is_text(s):
            # A category column is smaller when many rows share few values
            if s.nunique() / max(len(s), 1) < 0.5:
                out[col] = s.astype("category")
        if str(out[col].dtype) != before:
            changes[col] = (before, str(out[col].dtype))
    return out, changes

def compare(original, optimized):
    # Largest relative error over the numeric columns, and exact match for text
    max_error = 0.0
    text_same = True
    for col in original.columns:
        a = original[col]
        b = optimized[col].astype(a.dtype)
        if pd.api.types.is_numeric_dtype(a):
            a_values = a.to_numpy(dtype=float, na_value=np.nan)
            b_values = b.to_numpy(dtype=float, na_value=np.nan)
            diff = np.abs(a_values - b_values) / np.maximum(np.abs(a_values), 1e-12)
            if np.isfinite(diff).any():
                max_error = max(max_error, float(np.nanmax(diff)))
        else:
            text_same = text_same and a.equals(b)
    return max_error, text_same

files = sorted(Path(DATA_FOLDER).glob("*.csv"))
if not files:
    print("No CSV files found in", Path(DATA_FOLDER).resolve())
    print("Change DATA_FOLDER at the top of the script to the folder with your CSV files.")

print(f"{'File':28s} {'Before (KB)':>12s} {'After (KB)':>11s} {'Saved':>7s} {'Max error':>10s} {'Text same':>10s}")
details = {}
for path in files:
    df = read_csv_safely(path)
    optimized, changes = optimize(df)
    before = df.memory_usage(deep=True).sum() / 1024
    after = optimized.memory_usage(deep=True).sum() / 1024
    max_error, text_same = compare(df, optimized)
    details[path.name] = changes
    print(f"{path.name:28s} {before:12.1f} {after:11.1f} {100 * (1 - after / before):6.1f}% "
          f"{max_error:10.1e} {str(text_same):>10s}")

# Show which columns were changed for each file
print()
for name, changes in details.items():
    print(name)
    if not changes:
        print("  no column changed")
    for col, (old, new) in changes.items():
        print(f"  {col}: {old} to {new}")

print()
print("Summary:")
print("Smaller integer types and category columns give the largest savings.")
print("Text columns with few unique values usually save the most memory.")
print("Columns with mostly unique text, such as names, stay as they are.")
print("A max error of about 1e-7 comes from float32 and is too small to matter")
print("for most analysis, but it is not exact.")
