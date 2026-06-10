import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()

# Default path constants resolved relative to project root
DEFAULT_CLIMATE_PATH = PROJECT_ROOT / "data" / "processed" / "district_climate_data_1997_2022.csv"
DEFAULT_NEIGHBORS_PATH = PROJECT_ROOT / "data" / "processed" / "district_neighbors.json"

# Common default model hyperparameters
DEFAULT_CB_PARAMS = {
    'iterations': 350,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 5.0,
    'subsample': 0.8,
    'bootstrap_type': 'Bernoulli',
    'random_seed': 42,
    'verbose': 0
}

DEFAULT_XGB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 5.0,
    'reg_alpha': 1.0,
    'enable_categorical': True,
    'random_state': 42
}

DEFAULT_LGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_lambda': 2.0,
    'reg_alpha': 0.1,
    'random_state': 42,
    'verbose': -1
}

CROP_PROFILES = {
    # 1. KHARIF RICE (Optuna Tuned)
    "kharif_rice": {
        "crop_name": "rice",
        "season_prefix": "Kharif",
        "target_col": "Kharif_Yield",
        "month_offsets": [4, 5, 6, 7, 8, 9],  # May to Oct
        "month_labels": ["May", "Jun", "Jul", "Aug", "Sep", "Oct"],
        "core_months": ["Jun", "Jul", "Aug", "Sep"],
        "clipping_quantile": 0.99,
        "cb_params": {
            'iterations': 374, 'learning_rate': 0.0595, 'depth': 6, 'l2_leaf_reg': 5.750, 
            'subsample': 0.7083, 'random_strength': 0.3659, 'bootstrap_type': 'Bernoulli', 'random_seed': 42, 'verbose': 0
        },
        "xgb_params": {
            'n_estimators': 728, 'learning_rate': 0.1038, 'max_depth': 6, 'subsample': 0.6383, 
            'colsample_bytree': 0.8438, 'reg_lambda': 6.2578, 'reg_alpha': 4.0354, 'enable_categorical': True, 'random_state': 42
        },
        "lgb_params": {
            'n_estimators': 224, 'learning_rate': 0.0393, 'max_depth': 7, 'num_leaves': 19, 
            'subsample': 0.6004, 'subsample_freq': 1, 'colsample_bytree': 0.6357, 'reg_lambda': 1.4231, 'reg_alpha': 0.0826, 'random_state': 42, 'verbose': -1
        }
    },
    
    # 2. RABI WHEAT (Baseline Tuned)
    "rabi_wheat": {
        "crop_name": "wheat",
        "season_prefix": "Rabi",
        "target_col": "Rabi_Yield",
        "month_offsets": [-3, -2, -1, 0, 1, 2, 3],  # Oct to Apr
        "month_labels": ["Oct_Prev", "Nov_Prev", "Dec_Prev", "Jan", "Feb", "Mar", "Apr"],
        "core_months": [],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 3. KHARIF MAIZE
    "kharif_maize": {
        "crop_name": "maize",
        "season_prefix": "Kharif",
        "target_col": "Kharif_Yield",
        "month_offsets": [4, 5, 6, 7, 8, 9],
        "month_labels": ["May", "Jun", "Jul", "Aug", "Sep", "Oct"],
        "core_months": ["Jun", "Jul", "Aug", "Sep"],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 4. KHARIF GROUNDNUT
    "kharif_groundnut": {
        "crop_name": "groundnut",
        "season_prefix": "Kharif",
        "target_col": "Kharif_Yield",
        "month_offsets": [4, 5, 6, 7, 8, 9],
        "month_labels": ["May", "Jun", "Jul", "Aug", "Sep", "Oct"],
        "core_months": ["Jun", "Jul", "Aug", "Sep"],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 5. KHARIF SOYABEAN
    "kharif_soyabean": {
        "crop_name": "soyabean",
        "season_prefix": "Kharif",
        "target_col": "Kharif_Yield",
        "month_offsets": [4, 5, 6, 7, 8, 9],
        "month_labels": ["May", "Jun", "Jul", "Aug", "Sep", "Oct"],
        "core_months": ["Jun", "Jul", "Aug", "Sep"],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 6. KHARIF COTTON
    "kharif_cotton": {
        "crop_name": "cotton_lint",
        "season_prefix": "Kharif",
        "target_col": "Kharif_Yield",
        "month_offsets": [4, 5, 6, 7, 8, 9],
        "month_labels": ["May", "Jun", "Jul", "Aug", "Sep", "Oct"],
        "core_months": ["Jun", "Jul", "Aug", "Sep"],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 7. KHARIF ARHAR
    "kharif_arhar": {
        "crop_name": "arhar_tur",
        "season_prefix": "Kharif",
        "target_col": "Kharif_Yield",
        "month_offsets": [4, 5, 6, 7, 8, 9],
        "month_labels": ["May", "Jun", "Jul", "Aug", "Sep", "Oct"],
        "core_months": ["Jun", "Jul", "Aug", "Sep"],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 8. RABI POTATO
    "rabi_potato": {
        "crop_name": "potato",
        "season_prefix": "Rabi",
        "target_col": "Rabi_Yield",
        "month_offsets": [-3, -2, -1, 0, 1, 2, 3],
        "month_labels": ["Oct_Prev", "Nov_Prev", "Dec_Prev", "Jan", "Feb", "Mar", "Apr"],
        "core_months": [],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 9. RABI ONION
    "rabi_onion": {
        "crop_name": "onion",
        "season_prefix": "Rabi",
        "target_col": "Rabi_Yield",
        "month_offsets": [-3, -2, -1, 0, 1, 2, 3],
        "month_labels": ["Oct_Prev", "Nov_Prev", "Dec_Prev", "Jan", "Feb", "Mar", "Apr"],
        "core_months": [],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 10. RABI TOBACCO
    "rabi_tobacco": {
        "crop_name": "tobacco",
        "season_prefix": "Rabi",
        "target_col": "Rabi_Yield",
        "month_offsets": [-3, -2, -1, 0, 1, 2, 3],
        "month_labels": ["Oct_Prev", "Nov_Prev", "Dec_Prev", "Jan", "Feb", "Mar", "Apr"],
        "core_months": [],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 11. YEAR SUGARCANE
    "year_sugarcane": {
        "crop_name": "sugarcane",
        "season_prefix": "Year",
        "target_col": "Year_Yield",
        "month_offsets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # Jan to Dec
        "month_labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "core_months": [],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 12. YEAR BANANA
    "year_banana": {
        "crop_name": "banana",
        "season_prefix": "Year",
        "target_col": "Year_Yield",
        "month_offsets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "month_labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "core_months": [],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 13. YEAR GINGER
    "year_ginger": {
        "crop_name": "ginger",
        "season_prefix": "Year",
        "target_col": "Year_Yield",
        "month_offsets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "month_labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "core_months": [],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 14. YEAR TURMERIC
    "year_turmeric": {
        "crop_name": "turmeric",
        "season_prefix": "Year",
        "target_col": "Year_Yield",
        "month_offsets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "month_labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "core_months": [],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    },
    
    # 15. YEAR COCONUT
    "year_coconut": {
        "crop_name": "coconut",
        "season_prefix": "Year",
        "target_col": "Year_Yield",
        "month_offsets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "month_labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "core_months": [],
        "clipping_quantile": 0.99,
        "cb_params": DEFAULT_CB_PARAMS,
        "xgb_params": DEFAULT_XGB_PARAMS,
        "lgb_params": DEFAULT_LGB_PARAMS
    }
}
