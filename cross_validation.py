import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)

import warnings

warnings.filterwarnings('ignore')


# =========================================================
# NESTED CROSS-VALIDATION FOR LOGISTIC REGRESSION
# =========================================================
# This script performs:
# 1. Loading the prepared dataset
# 2. Nested cross-validation (5x5 folds)
# 3. Internal feature selection
# 4. Logistic regression model training
# 5. External fold evaluation
# 6. Output of metrics and model coefficients
# =========================================================


# =========================================================
# 1. LOAD AND PREPARE DATA
# =========================================================

df = pd.read_csv(
    "AIDS_Prepared_Data.csv"
)

X_full = sm.add_constant(
    df.drop(columns=['infected'])
)

y_full = df['infected']


# =========================================================
# 2. CROSS-VALIDATION PARAMETERS
# =========================================================

# Classification probability threshold
threshold = 0.28

# External cross-validation loop
outer_skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Internal cross-validation loop
inner_skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================================
# 3. INITIAL OUTPUT
# =========================================================

print("=" * 80)
print("NESTED CROSS-VALIDATION FOR LOGISTIC REGRESSION")
print("=" * 80)


# =========================================================
# 4. EXTERNAL CROSS-VALIDATION LOOP
# =========================================================
# The outer loop evaluates the generalization ability
# of the final logistic regression model.

for outer_fold, (
    train_idx,
    test_idx
) in enumerate(
    outer_skf.split(X_full, y_full),
    1
):

    # -----------------------------------------------------
    # Split outer training and testing subsets
    # -----------------------------------------------------

    X_train_out = X_full.iloc[train_idx]

    X_test_out = X_full.iloc[test_idx]

    y_train_out = y_full.iloc[train_idx]

    y_test_out = y_full.iloc[test_idx]


    # =====================================================
    # 5. INTERNAL FEATURE SELECTION
    # =====================================================
    # Forward feature selection is performed
    # using internal cross-validation.

    included = ['const']

    remaining = [
        f for f in X_train_out.columns
        if f != 'const'
    ]

    best_inner_f1 = 0


    # -----------------------------------------------------
    # Forward feature selection
    # -----------------------------------------------------

    for feature in remaining:

        inner_f1s = []

        # Internal cross-validation loop
        for in_tr_idx, in_val_idx in inner_skf.split(
            X_train_out,
            y_train_out
        ):

            X_it = X_train_out.iloc[in_tr_idx]

            X_iv = X_train_out.iloc[in_val_idx]

            y_it = y_train_out.iloc[in_tr_idx]

            y_iv = y_train_out.iloc[in_val_idx]

            try:

                model = sm.Logit(
                    y_it,
                    X_it[included + [feature]]
                ).fit(
                    disp=0,
                    method='bfgs'
                )

                preds = (
                    model.predict(
                        X_iv[included + [feature]]
                    ) > threshold
                ).astype(int)

                inner_f1s.append(
                    f1_score(
                        y_iv,
                        preds
                    )
                )

            except:

                inner_f1s.append(0)

        # Average internal F1-score
        mean_f1 = np.mean(inner_f1s)

        # Keep feature if performance improves
        if mean_f1 > best_inner_f1:

            best_inner_f1 = mean_f1

            included.append(feature)


    # =====================================================
    # 6. FINAL MODEL TRAINING FOR CURRENT FOLD
    # =====================================================

    final_model = sm.Logit(
        y_train_out,
        X_train_out[included]
    ).fit(
        disp=0,
        method='bfgs'
    )


    # =====================================================
    # 7. TESTING CURRENT FOLD
    # =====================================================

    test_probs = final_model.predict(
        X_test_out[included]
    )

    test_preds = (
        test_probs > threshold
    ).astype(int)


    # =====================================================
    # 8. OUTPUT RESULTS FOR CURRENT MODEL
    # =====================================================

    print("\n" + "█" * 60)

    print(
        f"MODEL #{outer_fold}"
    )

    print("█" * 60)


    # -----------------------------------------------------
    # Classification metrics
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "TEST SET CLASSIFICATION METRICS"
    )

    print("=" * 60)

    print(
        classification_report(
            y_test_out,
            test_preds,
            digits=4
        )
    )


    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    print("=" * 60)

    print("CONFUSION MATRIX")

    print("=" * 60)

    cm = confusion_matrix(
        y_test_out,
        test_preds
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            'Actual Healthy (0)',
            'Actual Infected (1)'
        ],
        columns=[
            'Predicted Healthy (0)',
            'Predicted Infected (1)'
        ]
    )

    print(cm_df)


    # -----------------------------------------------------
    # Model coefficients
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print(
        f"LOGISTIC REGRESSION COEFFICIENTS "
        f"(Features: {len(included)-1})"
    )

    print("=" * 60)

    coef_df = pd.DataFrame({
        'Feature': included,
        'Coefficient': final_model.params
    })

    coef_df['Coefficient'] = (
        coef_df['Coefficient']
        .astype(float)
        .round(4)
    )

    print(
        coef_df.to_string(index=False)
    )


    # -----------------------------------------------------
    # Logit equation
    # -----------------------------------------------------

    coeffs = final_model.params

    equation = (
        f"z = {coeffs['const']:.4f}"
    )

    for feat, val in coeffs.items():

        if feat != 'const':

            sign = "+" if val >= 0 else "-"

            equation += (
                f" {sign} "
                f"({abs(val):.4f} * {feat})"
            )

    print("\n" + "=" * 60)

    print("LOGIT EQUATION")

    print("=" * 60)

    print(equation)

    print("\n")


# =========================================================
# 9. FINAL OUTPUT
# =========================================================

print("=" * 80)
print("NESTED CROSS-VALIDATION COMPLETED")
print(
    "Select the best model or average the results."
)
print("=" * 80)