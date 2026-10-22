import pandas as pd
from sklearn.model_selection import train_test_split

# =========================================================
# RANDOM DATA SPLITTING
# =========================================================
# This script performs:
# 1. Loading the prepared dataset
# 2. Separating features and target variable
# 3. Random stratified splitting into:
#    - training set (70%)
#    - validation set (20%)
#    - test set (10%)
# 4. Saving all generated subsets
# =========================================================


# =========================================================
# 1. LOAD PREPARED DATASET
# =========================================================

df = pd.read_csv("AIDS_Prepared_Data.csv")


# =========================================================
# 2. SEPARATE FEATURES AND TARGET VARIABLE
# =========================================================

X = df.drop(columns=['infected'])

y = df['infected']


# =========================================================
# 3. FIRST SPLIT: TRAINING AND TEMPORARY SETS
# =========================================================
# The dataset is divided into:
# - training set: 70%
# - temporary set: 30%
#
# Stratification preserves the class distribution
# of the target variable.

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)


# =========================================================
# 4. SECOND SPLIT: VALIDATION AND TEST SETS
# =========================================================
# The temporary subset is divided into:
# - validation set: 20% of total data
# - test set: 10% of total data
#
# Since the temporary subset contains 30% of data,
# test_size = 1/3 produces the required ratio.

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=1/3,
    stratify=y_temp,
    random_state=42
)


# =========================================================
# 5. REATTACH TARGET VARIABLE
# =========================================================
# The target variable is added back to each subset
# before saving.

X_train['infected'] = y_train.values

X_valid['infected'] = y_valid.values

X_test['infected'] = y_test.values


# =========================================================
# 6. SAVE SPLIT DATASETS
# =========================================================

X_train.to_csv(
    "train_random.csv",
    index=False
)

X_valid.to_csv(
    "valid_random.csv",
    index=False
)

X_test.to_csv(
    "test_random.csv",
    index=False
)


# =========================================================
# 7. OUTPUT INFORMATION
# =========================================================

print("=" * 50)
print("RANDOM SPLIT COMPLETED")
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_valid)}")
print(f"Test set size: {len(X_test)}")
print("=" * 50)