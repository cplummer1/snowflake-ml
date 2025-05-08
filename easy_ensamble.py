# utils/modeling/xgboost_monthly.py

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.ensemble import EasyEnsembleClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_xgboost_with_adasyn(X, y, groups=None):
    """
    Train an EasyEnsemble of XGBoost models with randomized hyperparameter search.
    Inputs:
      - X: pd.DataFrame of training features
      - y: pd.Series or array-like of binary targets
      - groups: ignored (kept for signature compatibility)
    Returns:
      - model: the fitted EasyEnsembleClassifier (with best params)
      - imputer: the fitted SimpleImputer
    """

    # 1) Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_imp_arr = imputer.fit_transform(X)
    X_imp = pd.DataFrame(X_imp_arr, columns=X.columns)

    # 2) Base EasyEnsemble of XGB (we'll tune its parameters)
    base = EasyEnsembleClassifier(
        estimator=XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42
        ),
        random_state=42,
        n_jobs=-1
    )

    # 3) Hyperparameter space around (n_estimators=20, max_depth=4, bags=100)
    param_dist = {
        "n_estimators":            [15, 20, 30],               # number of under-sampled bags
        "sampling_strategy":       [0.2, 0.3, 0.4, 0.5, "auto"],
        "estimator__max_depth":    [3, 4, 5],
        "estimator__n_estimators": [80, 100, 120],
        "estimator__learning_rate":    [0.01, 0.05, 0.1],
        "estimator__gamma":            [0, 0.1, 0.3],
        "estimator__min_child_weight": [1, 3, 5],
        "estimator__subsample":        [0.6, 0.8, 1.0],
        "estimator__colsample_bytree": [0.6, 0.8, 1.0],
        "estimator__reg_alpha":        [0, 0.1, 1.0],
        "estimator__reg_lambda":       [1.0, 10.0],
    }

    # 4) Randomized search
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=30,            # number of random sets to try
        scoring="roc_auc",    # or "f2" if you prefer optimizing F₂
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # 5) Fit search
    search.fit(X_imp, y)
    model = search.best_estimator_
    print("🏆 Best params:", search.best_params_)

    # 6) Extract & print average feature importances across the ensemble
    importances = []
    for est in model.estimators_:
        # est is an XGBClassifier here
        if hasattr(est, "feature_importances_"):
            importances.append(est.feature_importances_)
    if importances:
        imp_arr = np.vstack(importances)
        avg_imp = imp_arr.mean(axis=0)
        feat_imp_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": avg_imp
        }).sort_values("Importance", ascending=False)
        print("\n=== Averaged Feature Importances (EasyEnsemble of XGB) ===")
        print(feat_imp_df.to_string(index=False))
    else:
        print("⚠️  Warning: no feature importances could be extracted.")

    return model, imputer
