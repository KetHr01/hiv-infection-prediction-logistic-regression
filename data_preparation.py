import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =========================================================
# DATA PREPARATION FOR LOGISTIC REGRESSION
# =========================================================
# This script performs:
# 1. Loading the dataset
# 2. Missing value processing
# 3. Feature engineering
# 4. Feature scaling
# 5. Saving the prepared dataset
# =========================================================


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("AIDS_Classification.csv")


# =========================================================
# 2. DEFINE FEATURE GROUPS
# =========================================================
# Numerical variables are processed using median imputation.
# Categorical variables are processed using mode imputation.

numeric_cols = [
    'time',
    'age',
    'wtkg',
    'karnof',
    'preanti',
    'cd40',
    'cd420',
    'cd80',
    'cd820'
]

categorical_cols = [
    'trt',
    'hemo',
    'homo',
    'drugs',
    'oprior',
    'z30',
    'race',
    'gender',
    'str2',
    'strat',
    'symptom',
    'treat',
    'offtrt'
]


# =========================================================
# 3. HANDLE MISSING VALUES
# =========================================================
# Missing numerical values are replaced by the median.
# Missing categorical values are replaced by the mode.

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# =========================================================
# 4. FEATURE ENGINEERING
# =========================================================
# Additional medical and statistical features are generated
# to improve the predictive ability of logistic regression.

new_features = pd.DataFrame(index=df.index)


# ---------------------------------------------------------
# 4.1 IMMUNOLOGICAL FEATURES
# ---------------------------------------------------------
# Features based on CD4/CD8 immune cell dynamics.

# Initial CD4/CD8 ratio
new_features['cd4_ratio_0'] = (
    df['cd40'] / (df['cd80'] + 1e-5)
)

# CD4/CD8 ratio after 20 weeks
new_features['cd4_ratio_20'] = (
    df['cd420'] / (df['cd820'] + 1e-5)
)

# Absolute CD4 change
new_features['cd4_change'] = (
    df['cd420'] - df['cd40']
)

# Absolute CD8 change
new_features['cd8_change'] = (
    df['cd820'] - df['cd80']
)

# Relative CD4 change
new_features['cd4_pct_change'] = (
    (df['cd420'] - df['cd40']) /
    (df['cd40'] + 1e-5)
)

# Relative CD8 change
new_features['cd8_pct_change'] = (
    (df['cd820'] - df['cd80']) /
    (df['cd80'] + 1e-5)
)

# Indicator of severe immune suppression
new_features['severe_immunodef'] = (
    df['cd40'] < 200
).astype(int)

# Indicator of CD4 decrease
new_features['cd4_dropped'] = (
    new_features['cd4_change'] < 0
).astype(int)

# Combined immune system index
new_features['immune_index'] = (
    (df['cd40'] + df['cd420']) /
    (df['cd80'] + df['cd820'] + 1e-5)
)

# Combined deterioration indicator
new_features['health_decline'] = (
    (
        new_features['cd4_change'] < 0
    ) &
    (
        new_features['cd8_change'] > 0
    )
).astype(int)


# ---------------------------------------------------------
# 4.2 INTERACTION FEATURES
# ---------------------------------------------------------
# Features describing interactions between variables.

# Age and immune status interaction
new_features['age_x_cd40'] = (
    df['age'] * df['cd40']
)

# Age and body weight interaction
new_features['age_x_wtkg'] = (
    df['age'] * df['wtkg']
)

# Previous antiretroviral treatment interaction
new_features['preanti_x_cd40'] = (
    df['preanti'] * df['cd40']
)

# Observation time and Karnofsky score interaction
new_features['time_x_karnof'] = (
    df['time'] * df['karnof']
)

# Combined behavioral risk score
new_features['risk_score'] = (
    df['hemo'] +
    df['homo'] +
    df['drugs']
)

# Gender and drug use interaction
new_features['gender_x_drugs'] = (
    df['gender'] * df['drugs']
)


# ---------------------------------------------------------
# 4.3 POLYNOMIAL AND LOG FEATURES
# ---------------------------------------------------------
# Nonlinear transformations for logistic regression.

poly_cols = ['age', 'cd40', 'cd420']

for col in poly_cols:

    # Quadratic transformation
    new_features[f'{col}_sq'] = df[col] ** 2

    # Logarithmic transformation
    new_features[f'log_{col}'] = np.log1p(df[col])


# ---------------------------------------------------------
# 4.4 TREATMENT ENCODING
# ---------------------------------------------------------
# One-hot encoding for treatment categories.

trt_dummies = pd.get_dummies(
    df['trt'],
    prefix='trt',
    drop_first=True
).astype(int)

new_features = pd.concat(
    [new_features, trt_dummies],
    axis=1
)


# =========================================================
# 5. COMBINE ORIGINAL AND GENERATED FEATURES
# =========================================================

df_final = pd.concat(
    [df, new_features],
    axis=1
)


# =========================================================
# 6. SEPARATE TARGET VARIABLE
# =========================================================

target = df_final['infected']

X = df_final.drop(columns=['infected'])


# =========================================================
# 7. FEATURE STANDARDIZATION
# =========================================================
# Logistic regression requires features on similar scales.

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)


# =========================================================
# 8. CREATE FINAL DATASET
# =========================================================

df_ready = X_scaled.copy()

df_ready['infected'] = target.values


# =========================================================
# 9. SAVE PREPARED DATASET
# =========================================================

df_ready.to_csv(
    "AIDS_Prepared_Data.csv",
    index=False
)


# =========================================================
# 10. OUTPUT INFORMATION
# =========================================================

print("=" * 50)
print("DATA PREPARATION COMPLETED")
print(f"Final dataset shape: {df_ready.shape}")
print("=" * 50)

