import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy import stats

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)

import warnings

warnings.filterwarnings('ignore')


# =========================================================
# LOGISTIC REGRESSION WITH MAHALANOBIS SPLITTING
# =========================================================
# This script performs:
# 1. Loading Mahalanobis-based split datasets
# 2. Stepwise feature selection using F-criteria
# 3. Logistic regression model training
# 4. Validation-based model selection
# 5. Final evaluation on the test dataset
# 6. Output of classification metrics and coefficients
# =========================================================


# =========================================================
# 1. LOAD DATASETS
# =========================================================

train_df = pd.read_csv(
    "train_mahalanobis.csv"
)

valid_df = pd.read_csv(
    "valid_mahalanobis.csv"
)

test_df = pd.read_csv(
    "test_mahalanobis.csv"
)


# =========================================================
# 2. PREPARE FEATURES AND TARGET
# =========================================================
# The constant term is added for logistic regression.

def prepare_xy(df):

    y = df['infected']

    X = df.drop(columns=['infected'])

    X = sm.add_constant(
        X,
        has_constant='add'
    )

    return X, y


X_train, y_train = prepare_xy(train_df)

X_val, y_val = prepare_xy(valid_df)

X_test, y_test = prepare_xy(test_df)


# =========================================================
# 3. STEPWISE FEATURE SELECTION FUNCTION
# =========================================================
# Features are added and removed according to
# inclusion and exclusion F-criteria.

def stepwise_by_f_criteria(
    X_tr,
    y_tr,
    f_in,
    f_out
):

    # Convert F-thresholds to p-value thresholds
    p_in = 1 - stats.f.cdf(
        f_in,
        1,
        len(X_tr) - 1
    )

    p_out = 1 - stats.f.cdf(
        f_out,
        1,
        len(X_tr) - 1
    )

    # Start with intercept only
    included = ['const']

    while True:

        changed = False

        excluded = list(
            set(X_tr.columns) - set(included)
        )

        new_pvals = pd.Series(
            index=excluded,
            dtype=float
        )

        # -------------------------------------------------
        # FORWARD STEP
        # -------------------------------------------------
        # Try adding each excluded feature.

        for col in excluded:

            try:

                res = sm.Logit(
                    y_tr,
                    X_tr[included + [col]]
                ).fit(
                    disp=0,
                    method='bfgs',
                    maxiter=100
                )

                new_pvals[col] = (
                    res.pvalues[col]
                )

            except:

                continue

        # Add feature with the smallest p-value
        if not new_pvals.dropna().empty:

            if new_pvals.min() < p_in:

                included.append(
                    new_pvals.idxmin()
                )

                changed = True

        # -------------------------------------------------
        # BACKWARD STEP
        # -------------------------------------------------
        # Remove features with large p-values.

        try:

            res = sm.Logit(
                y_tr,
                X_tr[included]
            ).fit(
                disp=0,
                method='bfgs',
                maxiter=100
            )

            pvals = res.pvalues.drop(
                'const'
            )

            if not pvals.empty:

                if pvals.max() > p_out:

                    included.remove(
                        pvals.idxmax()
                    )

                    changed = True

        except:

            pass

        # Stop if no changes occurred
        if not changed:

            break

    return included


# =========================================================
# 4. DEFINE F-CRITERIA SCENARIOS
# =========================================================
# Different inclusion and exclusion thresholds
# are evaluated during model selection.

f_scenarios = [
    (5.0, 3.0),
    (4.5, 3.5),
    (3.2, 2.5),
    (3.84, 2.71),
    (3.0, 2.0),
    (2.5, 1.5),
    (1.5, 0.7),
    (1.0, 0.5)
]

# Probability threshold for classification
threshold_opt = 0.28

best_val_f1 = -1

best_features = []

best_params = (0, 0)


# =========================================================
# 5. VALIDATION-BASED MODEL SELECTION
# =========================================================

print("=" * 60)
print("VALIDATION-BASED MODEL SELECTION")
print("=" * 60)

for f_in, f_out in f_scenarios:

    current_features = (
        stepwise_by_f_criteria(
            X_train,
            y_train,
            f_in,
            f_out
        )
    )

    try:

        model = sm.Logit(
            y_train,
            X_train[current_features]
        ).fit(
            disp=0,
            method='bfgs'
        )

        # Validation predictions
        val_probs = model.predict(
            X_val[current_features]
        )

        val_preds = (
            val_probs > threshold_opt
        ).astype(int)

        # Validation F1-score
        val_f1 = f1_score(
            y_val,
            val_preds
        )

        print(
            f"F_in = {f_in:<4} | "
            f"F_out = {f_out:<4} | "
            f"Features = {len(current_features)-1:<2} | "
            f"Validation F1 = {val_f1:.4f}"
        )

        # Save best configuration
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_features = current_features

            best_params = (f_in, f_out)

    except:

        continue


# =========================================================
# 6. FINAL MODEL TRAINING
# =========================================================
# The best feature subset is used
# to train the final logistic regression model.

print("\n" + "=" * 60)
print("BEST MODEL CONFIGURATION")
print("=" * 60)

print(
    f"Best F_in threshold : {best_params[0]}"
)

print(
    f"Best F_out threshold: {best_params[1]}"
)

print(
    f"Selected features   : {len(best_features)-1}"
)

print(
    f"Best validation F1  : {best_val_f1:.4f}"
)

final_model = sm.Logit(
    y_train,
    X_train[best_features]
).fit(
    disp=0,
    method='bfgs'
)


# =========================================================
# 7. TEST SET EVALUATION
# =========================================================

test_probs = final_model.predict(
    X_test[best_features]
)

test_preds = (
    test_probs > threshold_opt
).astype(int)


# =========================================================
# 8. CLASSIFICATION METRICS
# =========================================================

print("\n" + "=" * 60)
print("TEST SET CLASSIFICATION METRICS")
print("=" * 60)

print(
    classification_report(
        y_test,
        test_preds,
        digits=4
    )
)


# =========================================================
# 9. CONFUSION MATRIX
# =========================================================

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
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


# =========================================================
# 10. MODEL COEFFICIENTS
# =========================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 60)

coef_df = pd.DataFrame({
    'Feature': best_features,
    'Coefficient': final_model.params
})

coef_df['Coefficient'] = (
    coef_df['Coefficient']
    .astype(float)
    .round(4)
)

print(coef_df.to_string(index=False))


# =========================================================
# 11. SAVE MODEL COEFFICIENTS
# =========================================================

coef_df.to_csv(
    "best_logistic_model_mahalanobis.csv",
    index=False
)


# =========================================================
# 12. FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION MODELING COMPLETED")
print("=" * 60)