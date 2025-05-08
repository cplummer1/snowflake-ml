import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_validate
from xgboost import XGBClassifier

def train_xgboost_with_adasyn(X, y, groups):
    # 0) Feature list
    print(f"\n🔍 Features before sampling & tuning ({X.shape[1]} total):")
    print(X.columns.tolist())

    # 1) Impute
    imputer = SimpleImputer(strategy="mean")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 2) Pipeline: ADASYN → XGB
    pipe = ImbPipeline([
        ("sampler", ADASYN(random_state=42)),
        ("clf", XGBClassifier(eval_metric="auc", random_state=42))
    ])

    # 3) Expanded grid
    param_grid = {
        # ADASYN settings
        "sampler__sampling_strategy": [0.2, 0.4, 0.6, 0.8, "auto"],
        "sampler__n_neighbors":       [3, 5, 7],

        # XGB core
        "clf__max_depth":             [3, 4, 5],
        "clf__learning_rate":         [0.01, 0.02, 0.05, 0.1],
        "clf__n_estimators":          [50, 100, 200],
        "clf__subsample":             [0.8],
        "clf__colsample_bytree":      [0.8],

        # Regularization & scale
        "clf__reg_alpha":             [0, 0.1],
        "clf__reg_lambda":            [1, 10],
        "clf__gamma":                 [0, 0.05],
        "clf__min_child_weight":      [1],
        "clf__scale_pos_weight":      [1, 2],
    }

    # 4) More folds for inner & outer CV
    inner_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    outer_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)

    # 5) Inner GridSearchCV
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,      # now 4-fold
        n_jobs=-1,
        verbose=2
    )

    # 6) Outer CV evaluation
    cv_results = cross_validate(
        grid,
        X_imp, y,
        groups=groups,
        cv=outer_cv,      # now 4-fold
        scoring="roc_auc",
        n_jobs=-1,
        return_train_score=False
    )
    print(f"\n🏅 Outer CV AUC scores: {cv_results['test_score']}")
    print(f"🏅 Mean outer CV AUC: {np.mean(cv_results['test_score']):.4f}\n")

    # 7) Fit on full data & report
    grid.fit(X_imp, y, groups=groups)
    print("🎯 Best ADASYN + XGB params:", grid.best_params_)

    return grid.best_estimator_, imputer
