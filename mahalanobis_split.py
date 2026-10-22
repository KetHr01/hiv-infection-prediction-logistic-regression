import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis

# =========================================================
# MAHALANOBIS-BASED DATA SPLITTING
# =========================================================
# This script performs:
# 1. Loading the prepared dataset
# 2. Computing Mahalanobis distance
# 3. Ordering observations by similarity
# 4. Stratified splitting into:
#    - training set (70%)
#    - validation set (20%)
#    - test set (10%)
# 5. Saving all generated subsets
# =========================================================


# =========================================================
# 1. LOAD PREPARED DATASET
# =========================================================

try:

    df = pd.read_csv("AIDS_Prepared_Data.csv")

    X = df.drop(columns=['infected'])

    y = df['infected']

except FileNotFoundError:

    print("=" * 50)
    print("ERROR: FILE NOT FOUND")
    print("AIDS_Prepared_Data.csv is missing.")
    print("=" * 50)

    exit()


# =========================================================
# 2. MAHALANOBIS VARIATION SERIES FUNCTION
# =========================================================
# This function:
# - computes the Mahalanobis distance
# - sorts observations from the center
#   to the periphery of the distribution

def get_mahalanobis_ordered_indices(X_class):

    # Compute centroid of the class
    centroid = X_class.mean(axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(X_class.values.T)

    # Use pseudo-inverse for numerical stability
    inv_cov = np.linalg.pinv(cov_matrix)

    distances = []

    for idx in X_class.index:

        d = mahalanobis(
            X_class.loc[idx],
            centroid,
            inv_cov
        )

        distances.append((idx, d))

    # Sort by Mahalanobis distance
    distances.sort(key=lambda x: x[1])

    return [x[0] for x in distances]


# =========================================================
# 3. STRATIFIED MAHALANOBIS SPLITTING FUNCTION
# =========================================================
# Splitting is performed separately for each class
# in order to preserve class balance.

def balanced_mahalanobis_split(
    X,
    y,
    train_ratio=0.7,
    valid_ratio=0.2,
    test_ratio=0.1
):

    train_idx = []

    valid_idx = []

    test_idx = []

    # Process each class separately
    for cls in y.unique():

        class_idx = y[y == cls].index

        X_class = X.loc[class_idx]

        # Get observations ordered by Mahalanobis distance
        ordered_idx = get_mahalanobis_ordered_indices(
            X_class
        )

        # Assign observations according to 7:2:1 ratio
        for i, idx in enumerate(ordered_idx):

            position = i % 10

            # Training subset (70%)
            if position < 7:

                train_idx.append(idx)

            # Validation subset (20%)
            elif position < 9:

                valid_idx.append(idx)

            # Test subset (10%)
            else:

                test_idx.append(idx)

    return train_idx, valid_idx, test_idx


# =========================================================
# 4. PERFORM DATA SPLITTING
# =========================================================

train_idx, valid_idx, test_idx = (
    balanced_mahalanobis_split(
        X,
        y,
        train_ratio=0.7,
        valid_ratio=0.2,
        test_ratio=0.1
    )
)


# =========================================================
# 5. CREATE DATA SUBSETS
# =========================================================
# Observations are shuffled after selection
# to remove ordering effects.

train_df = df.loc[train_idx].sample(
    frac=1,
    random_state=42
)

valid_df = df.loc[valid_idx].sample(
    frac=1,
    random_state=42
)

test_df = df.loc[test_idx].sample(
    frac=1,
    random_state=42
)


# =========================================================
# 6. SAVE SPLIT DATASETS
# =========================================================

train_df.to_csv(
    "train_mahalanobis.csv",
    index=False
)

valid_df.to_csv(
    "valid_mahalanobis.csv",
    index=False
)

test_df.to_csv(
    "test_mahalanobis.csv",
    index=False
)


# =========================================================
# 7. OUTPUT INFORMATION
# =========================================================

total_n = len(df)

print("=" * 50)
print("MAHALANOBIS SPLIT COMPLETED")
print(f"Total observations: {total_n}")
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(valid_df)}")
print(f"Test set size: {len(test_df)}")
print("=" * 50)