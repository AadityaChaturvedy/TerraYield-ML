import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge

def get_cv_folds(df, unique_years, n_splits=3, test_years_per_fold=2, val_size_years=2):
    """
    Generate chronological time-series train/val/test year splits.

    Parameters
    ----------
    df : pandas.DataFrame
        Input preprocessed crop yield data.
    unique_years : numpy.ndarray
        Sorted array of unique years in the dataset.
    n_splits : int, default=3
        Number of cross-validation folds.
    test_years_per_fold : int, default=2
        Number of consecutive years to use for the testing set in each fold.
    val_size_years : int, default=2
        Number of consecutive years immediately preceding the test set to use for validation.

    Returns
    -------
    list of tuple
        A list of folds, where each fold is a tuple of (actual_train_years, val_years, test_years).
    """
    folds = []
    for fold in range(n_splits):
        end_idx = len(unique_years) - (n_splits - fold) * test_years_per_fold
        if end_idx <= val_size_years + 2:
            continue
        test_years = unique_years[end_idx : end_idx + test_years_per_fold]
        train_years = unique_years[:end_idx]
        val_years = train_years[-val_size_years:]
        actual_train_years = train_years[:-val_size_years]
        folds.append((actual_train_years, val_years, test_years))
    return folds

def clip_target_by_state(df_train, X_train, y_train, target_col, clipping_quantile=0.99):
    """
    Clip training yield values at the specified quantile (e.g., 99th percentile) by state.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training partition of the crop yield dataframe.
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training target yield series.
    target_col : str
        Target yield column name.
    clipping_quantile : float, default=0.99
        Quantile threshold for outlier clipping.

    Returns
    -------
    pandas.Series
        Clipped target yield series.
    """
    # Create temporary df to do group quantile mapping
    train_df = df_train.copy()
    state_q = train_df.groupby("State", observed=True)[target_col].quantile(clipping_quantile)
    ceilings = X_train["State"].map(state_q).astype(float).fillna(15.0)
    return y_train.clip(upper=ceilings)

def fit_models(
    X_train, y_train, X_val, y_val, X_test, y_test,
    cat_features, cb_params, xgb_params, lgb_params
):
    """
    Fit XGBoost, LightGBM, and CatBoost models on train/val, and return predictions and best iterations.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training target yield series.
    X_val : pandas.DataFrame
        Validation feature matrix (used for early stopping).
    y_val : pandas.Series
        Validation target yield series.
    X_test : pandas.DataFrame
        Testing feature matrix.
    y_test : pandas.Series
        Testing target yield series.
    cat_features : list of str
        List of categorical feature column names.
    cb_params : dict
        Hyperparameters for the CatBoost model.
    xgb_params : dict
        Hyperparameters for the XGBoost model.
    lgb_params : dict
        Hyperparameters for the LightGBM model.

    Returns
    -------
    dict
        Dictionary containing predictions on validation and test sets, and best_iter for each model.
    """
    # 1. XGBoost
    xgb_params_fit = {k: v for k, v in xgb_params.items()}
    xgb_params_fit.update({'early_stopping_rounds': 30})
    model_xgb = xgb.XGBRegressor(**xgb_params_fit)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p_xgb_val = model_xgb.predict(X_val)
    p_xgb_test = model_xgb.predict(X_test)

    # 2. LightGBM
    lgb_params_fit = {k: v for k, v in lgb_params.items()}
    lgb_params_fit.update({'verbose': -1, 'subsample_freq': 1})
    model_lgb = lgb.LGBMRegressor(**lgb_params_fit)
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    p_lgb_val = model_lgb.predict(X_val)
    p_lgb_test = model_lgb.predict(X_test)

    # 3. CatBoost
    cb_params_fit = {k: v for k, v in cb_params.items()}
    cb_params_fit.update({'bootstrap_type': 'Bernoulli', 'random_seed': 42, 'verbose': 0, 'early_stopping_rounds': 30})
    model_cb = cb.CatBoostRegressor(**cb_params_fit)
    model_cb.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), verbose=False)
    p_cb_val = model_cb.predict(X_val)
    p_cb_test = model_cb.predict(X_test)

    return {
        'xgb': {'val': p_xgb_val, 'test': p_xgb_test, 'best_iter': getattr(model_xgb, 'best_iteration', xgb_params.get('n_estimators', 500))},
        'lgb': {'val': p_lgb_val, 'test': p_lgb_test, 'best_iter': getattr(model_lgb, 'best_iteration_', lgb_params.get('n_estimators', 200))},
        'cb': {'val': p_cb_val, 'test': p_cb_test, 'best_iter': model_cb.get_best_iteration() if model_cb.get_best_iteration() is not None else cb_params.get('iterations', 350)}
    }

