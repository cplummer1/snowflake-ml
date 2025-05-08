# utils/modeling/xgboost_monthly.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split
from xgboost import XGBClassifier

def train_xgboost_with_adasyn(X, y, groups):
    """
    Train an XGBoost classifier on X, y using:
      • BorderlineSMOTE(kind="borderline-1") to focus on “hard” minority points
      • EditedNearestNeighbours to clean up noise
      • RandomizedSearchCV tuning both sampling ratios and XGB params
      • Early stopping on a hold-out fold
    Returns (best_model, imputer).
    """
    # 1) Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 2) carve out an eval set for early stopping
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_imp, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 3) pipeline: Borderline-1 SMOTE → ENN → XGB w/ early stopping
    pipe = ImbPipeline([
        ("smote", BorderlineSMOTE(
            kind="borderline-1",
            random_state=42
        )),
        ("enn", EditedNearestNeighbours()),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            max_delta_step=1,
            early_stopping_rounds=30
        ))
    ])

    # 4) search space
    param_dist = {
        # Borderline-1 SMOTE
        "smote__sampling_strategy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "auto"],
        "smote__k_neighbors":        [1, 3, 5, 7, 9, 11, 13],

        # ENN (you can adjust n_neighbors if you like)
        "enn__n_neighbors":          [3, 5],

        # XGB hyperparameters (narrowed to the region that was strongest)
        "clf__n_estimators":         [100, 200, 300],
        "clf__max_depth":            [3, 4, 5],
        "clf__learning_rate":        [0.01, 0.05],
        "clf__subsample":            [0.7, 0.9],
        "clf__colsample_bytree":     [0.7, 0.9],
        "clf__colsample_bylevel":    [0.7, 0.9],

        "clf__gamma":                [0, 0.1],
        "clf__min_child_weight":     [1, 3],
        "clf__reg_alpha":            [0, 0.1],
        "clf__reg_lambda":           [1, 10],

        # XGB imbalance knob
        "clf__scale_pos_weight":     [1, 2],
    }

    # 5) avoid zero-positive splits
    outer_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    # 6) randomized search (bump to 60 iterations)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=60,
        scoring="roc_auc",
        cv=outer_cv,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    # 7) fit with eval_set for early stopping
    search.fit(
        X_tr, y_tr,
        clf__eval_set=[(X_es, y_es)]
    )
    print("🏆 Best Borderline-1 SMOTE + ENN + XGB params:", search.best_params_)

    # 8) unwrap the XGB model so .feature_importances_ is available
    best_model = search.best_estimator_.named_steps["clf"]

    return best_model, imputer
