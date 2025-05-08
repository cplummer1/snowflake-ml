import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import ADASYN

def train_xgboost_with_adasyn(X_train, y_train):
    # 1) mean‐impute
    imputer = SimpleImputer(strategy='mean')
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

    # 2) tune ADASYN on a temporary split
    X_temp, X_val, y_temp, y_val = train_test_split(
        X_tr_imp, y_train, test_size=0.4, random_state=42, stratify=y_train
    )
    best_auc = 0
    for k in [2,3,4,5,6,7,8]:
        for strat in [0.2,0.3,0.4,0.5,0.6,0.7,'auto']:
            try:
                ad = ADASYN(sampling_strategy=strat, n_neighbors=k, random_state=42)
                X_res, y_res = ad.fit_resample(X_temp, y_temp)
                m = XGBClassifier(n_estimators=30, max_depth=3,
                                  random_state=42, eval_metric='auc')
                m.fit(X_res, y_res)
                auc = roc_auc_score(y_val, m.predict_proba(X_val)[:,1])
                print(f"ADASYN(k={k}, strat={strat}) → AUC: {auc:.4f}")
                if auc>best_auc:
                    best_auc, best_k, best_s = auc, k, strat
                    best_Xr, best_yr = X_res, y_res
            except:
                pass

    print(f"\n✅ Best ADASYN config: n_neighbors={best_k}, strat={best_s} with AUC={best_auc:.4f}")
    X_res, y_res = best_Xr, best_yr

    # 3) grid‐search XGB on that resampled data
    scale = len(y_res[y_res==0]) / len(y_res[y_res==1]) if len(y_res[y_res==1])>0 else 1
    param_grid = {
        'n_estimators': [50],
        'max_depth':[3,4],
        'learning_rate':[0.02,0.05],
        'subsample':[0.8],'colsample_bytree':[0.8],
        'reg_alpha':[0,0.1],'reg_lambda':[1.0],'gamma':[0,0.05],
        'min_child_weight':[1],'scale_pos_weight':[1,2]
    }
    gs = GridSearchCV(
        estimator=XGBClassifier(eval_metric='auc',random_state=42),
        param_grid=param_grid, scoring='roc_auc',
        cv=3, verbose=1, n_jobs=-1
    )
    gs.fit(X_res, y_res)
    best_model = gs.best_estimator_
    print(f"Best XGB params: {gs.best_params_}")

    return best_model, imputer

