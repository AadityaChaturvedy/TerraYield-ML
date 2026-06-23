import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

from src.config import CROP_PROFILES, PROJECT_ROOT
from src.data_loader import load_and_preprocess
from src.features import build_feature_list
from src.models import get_cv_folds, clip_target_by_state, fit_models
from src.tft_model import TemporalFusionTransformer

class KharifRiceDataset(Dataset):
    """
    Dataset wrapper to prepare tabular crop yield data into static and temporal tensors for TFT.
    """
    def __init__(self, df, static_cont_cols, temporal_groups, target_col, state_to_idx, dist_to_idx):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        
        # Categorical maps
        self.states = torch.tensor(self.df['State'].map(state_to_idx).fillna(0).astype(int).values, dtype=torch.long)
        self.districts = torch.tensor(self.df['District_mapped'].map(dist_to_idx).fillna(0).astype(int).values, dtype=torch.long)
        
        # Static continuous features: list of tensors of shape [batch, 1]
        self.static_conts = []
        for col in static_cont_cols:
            vals = torch.tensor(self.df[col].values, dtype=torch.float32).unsqueeze(-1)
            self.static_conts.append(vals)
            
        # Target
        self.targets = torch.tensor(self.df[target_col].values, dtype=torch.float32)
        
        # Temporal sequences
        # June, July, August, September
        self.months = ['Jun', 'Jul', 'Aug', 'Sep']
        self.temporal_seqs = []
        
        for group in temporal_groups:
            # Group is a list of feature prefixes, e.g., ['Precip', 'Precip_Anomaly', 'Precip_Z']
            group_tensors = []
            for m in self.months:
                step_cols = []
                for prefix in group:
                    if prefix == 'CWSI':
                        col_name = f"CWSI_{m}"
                    else:
                        # e.g., Precip_Jun, Precip_Jun_Anomaly, Precip_Jun_Z
                        parts = prefix.split('_')
                        if len(parts) == 1:
                            col_name = f"{parts[0]}_{m}"
                        else:
                            col_name = f"{parts[0]}_{m}_{'_'.join(parts[1:])}"
                    
                    if col_name in self.df.columns:
                        step_cols.append(col_name)
                
                # Check if we have columns for this step
                if step_cols:
                    step_vals = self.df[step_cols].values # [batch, feat_dim]
                    group_tensors.append(torch.tensor(step_vals, dtype=torch.float32))
                
            # Stack along sequence dimension to get [batch, seq_len, feat_dim]
            if group_tensors:
                group_seq = torch.stack(group_tensors, dim=1)
                self.temporal_seqs.append(group_seq)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        static_cont_idx = [tensor[idx] for tensor in self.static_conts]
        temporal_seq_idx = [tensor[idx] for tensor in self.temporal_seqs]
        return (
            self.states[idx],
            self.districts[idx],
            static_cont_idx,
            temporal_seq_idx,
            self.targets[idx]
        )

def collate_fn(batch):
    states = torch.stack([item[0] for item in batch])
    districts = torch.stack([item[1] for item in batch])
    
    # static_cont_idx is a list of tensors of shape [1]
    # We want a list of tensors of shape [batch, 1]
    num_static = len(batch[0][2])
    static_conts = []
    for i in range(num_static):
        static_conts.append(torch.stack([item[2][i] for item in batch]))
        
    # temporal_seq_idx is a list of tensors of shape [seq_len, feat_dim]
    # We want a list of tensors of shape [batch, seq_len, feat_dim]
    num_temporal = len(batch[0][3])
    temporal_seqs = []
    for i in range(num_temporal):
        temporal_seqs.append(torch.stack([item[3][i] for item in batch]))
        
    targets = torch.stack([item[4] for item in batch])
    
    return states, districts, static_conts, temporal_seqs, targets

def train_model(model, train_loader, val_loader, epochs, lr, device, patience=15):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 12
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for states, districts, static_conts, temporal_seqs, targets in train_loader:
            states = states.to(device)
            districts = districts.to(device)
            static_conts = [x.to(device) for x in static_conts]
            temporal_seqs = [x.to(device) for x in temporal_seqs]
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(states, districts, static_conts, temporal_seqs)
            
            loss = criterion(preds, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, districts, static_conts, temporal_seqs, targets in val_loader:
                states = states.to(device)
                districts = districts.to(device)
                static_conts = [x.to(device) for x in static_conts]
                temporal_seqs = [x.to(device) for x in temporal_seqs]
                targets = targets.to(device)
                
                preds = model(states, districts, static_conts, temporal_seqs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * targets.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            # print(f"Early stopping at epoch {epoch+1}")
            break
            
    # Load best weights
    model.load_state_dict(best_model_state)
    return model

def predict(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for states, districts, static_conts, temporal_seqs, _ in loader:
            states = states.to(device)
            districts = districts.to(device)
            static_conts = [x.to(device) for x in static_conts]
            temporal_seqs = [x.to(device) for x in temporal_seqs]
            
            preds = model(states, districts, static_conts, temporal_seqs)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

def main():
    parser = argparse.ArgumentParser(description="Train Temporal Fusion Transformer on Kharif Rice.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--d_model", type=int, default=64, help="Embedding size for the TFT model")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    args = parser.parse_args()

    crop_name = "kharif_rice"
    profile = CROP_PROFILES[crop_name]
    target_col = profile["target_col"]

    print("\n=======================================================")
    print(" TRAINING TEMPORAL FUSION TRANSFORMER (TFT) FOR KHARIF RICE ")
    print("=======================================================\n")

    # Step 1: Loading data
    print("\nStep 1: Loading and preprocessing data...")
    df = load_and_preprocess(
        crop_profile_name=crop_name,
        include_district_feature=True,
        include_extended_months=True,
        include_soil_l1=True,
        include_interactions=False,
        include_yield_trend=False,
        include_lag3=True
    )
    # Defragment DataFrame to avoid huge memory spikes when slicing
    df = df.copy()
    
    # 2. Get features lists
    features = build_feature_list(
        df,
        crop_profile_name=crop_name,
        include_district_feature=True,
        include_extended_months=True,
        include_soil_l1=True,
        include_interactions=False,
        include_yield_trend=False,
        include_lag3=True
    )
    
    # Separate categorical maps
    states_list = sorted(df['State'].unique())
    state_to_idx = {s: i for i, s in enumerate(states_list)}
    
    dist_list = sorted(df['District_mapped'].unique())
    dist_to_idx = {d: i for i, d in enumerate(dist_list)}
    
    # Define Static continuous features
    static_cont_cols = [
        "Kharif_Yield_Lag1", "Kharif_Yield_Lag2", "Kharif_Yield_Lag3",
        "Kharif_Yield_Hist_Mean", "Kharif_Yield_Lag1_Anomaly", "Kharif_Yield_Spatial_Lag1",
        "Kharif_Area_Lag1", "Kharif_Production_Lag1",
        "Post2000", "Post2019", "Year"
    ]
    # Add May and Oct climate features as static features
    for m in ["May", "Oct"]:
        for var in ["Precip", "Temp", "Soil", "SoilL1", "CWSI"]:
            for suffix in ["", "_Anomaly", "_Z"]:
                col = f"{var}_{m}{suffix}"
                if col in df.columns:
                    static_cont_cols.append(col)
                    
    # Define Temporal groups (June, July, August, September sequence)
    temporal_groups = [
        ['Precip', 'Precip_Anomaly', 'Precip_Z'],
        ['Temp', 'Temp_Anomaly', 'Temp_Z'],
        ['Soil', 'Soil_Anomaly', 'Soil_Z'],
        ['SoilL1', 'SoilL1_Anomaly', 'SoilL1_Z'],
        ['NDVI', 'NDVI_Anomaly', 'NDVI_Z'],
        ['EVI', 'EVI_Anomaly', 'EVI_Z'],
        ['CWSI']
    ]
    
    # Map group dimensions
    temporal_dims = []
    months = ['Jun', 'Jul', 'Aug', 'Sep']
    for group in temporal_groups:
        dim = 0
        for prefix in group:
            if prefix == 'CWSI':
                col_name = "CWSI_Jun"
            else:
                parts = prefix.split('_')
                if len(parts) == 1:
                    col_name = f"{parts[0]}_Jun"
                else:
                    col_name = f"{parts[0]}_Jun_{'_'.join(parts[1:])}"
            if col_name in df.columns:
                dim += 1
        temporal_dims.append(dim)

    print(f"Features configurations:")
    print(f"  - Static categorical: State (vocab={len(state_to_idx)}), District (vocab={len(dist_to_idx)})")
    print(f"  - Static continuous: {len(static_cont_cols)} columns")
    print(f"  - Temporal sequence: {len(temporal_groups)} groups with dimensions {temporal_dims} across 4 months")

    unique_years = np.sort(df["Year"].unique())
    folds = get_cv_folds(df, unique_years)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}\n")

    tft_cv_results = []
    tree_cv_results = []

    # Chronological Out-of-Time CV
    for fold_idx, (actual_train_years, val_years, test_years) in enumerate(folds):
        print(f"--- Fold {fold_idx+1} | Test Years: {test_years.min()}-{test_years.max()} ---")
        
        train_mask = df["Year"].isin(actual_train_years)
        val_mask = df["Year"].isin(val_years)
        test_mask = df["Year"].isin(test_years)
        
        print(f"Train size: {train_mask.sum()}, Val size: {val_mask.sum()}, Test size: {test_mask.sum()}")
        import sys
        sys.stdout.flush()
        
        # Split DataFrames
        df_train = df[train_mask].copy()
        df_val = df[val_mask].copy()
        df_test = df[test_mask].copy()
        
        # Scale continuous static variables and target values
        # (We use StandardScaler fitted on training set)
        scaler = StandardScaler()
        df_train[static_cont_cols] = scaler.fit_transform(df_train[static_cont_cols])
        df_val[static_cont_cols] = scaler.transform(df_val[static_cont_cols])
        df_test[static_cont_cols] = scaler.transform(df_test[static_cont_cols])
        
        # We also need to scale the individual temporal columns in the DataFrame
        all_temporal_cols = []
        for m in months:
            for group in temporal_groups:
                for prefix in group:
                    if prefix == 'CWSI':
                        col_name = f"CWSI_{m}"
                    else:
                        parts = prefix.split('_')
                        if len(parts) == 1:
                            col_name = f"{parts[0]}_{m}"
                        else:
                            col_name = f"{parts[0]}_{m}_{'_'.join(parts[1:])}"
                    if col_name in df.columns:
                        all_temporal_cols.append(col_name)
                        
        t_scaler = StandardScaler()
        df_train[all_temporal_cols] = t_scaler.fit_transform(df_train[all_temporal_cols])
        df_val[all_temporal_cols] = t_scaler.transform(df_val[all_temporal_cols])
        df_test[all_temporal_cols] = t_scaler.transform(df_test[all_temporal_cols])
        
        # Fit-clip target (standard pipeline mechanism)
        X_train_clip = df_train[features].copy()
        y_train_clip = clip_target_by_state(df_train, X_train_clip, df_train[target_col], target_col, profile["clipping_quantile"])
        df_train[target_col] = y_train_clip
        
        # Create PyTorch datasets and loaders
        train_dataset = KharifRiceDataset(df_train, static_cont_cols, temporal_groups, target_col, state_to_idx, dist_to_idx)
        val_dataset = KharifRiceDataset(df_val, static_cont_cols, temporal_groups, target_col, state_to_idx, dist_to_idx)
        test_dataset = KharifRiceDataset(df_test, static_cont_cols, temporal_groups, target_col, state_to_idx, dist_to_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Instantiate TFT Model
        model = TemporalFusionTransformer(
            state_vocab_size=len(state_to_idx),
            district_vocab_size=len(dist_to_idx),
            static_cont_dims=[1] * len(static_cont_cols),
            temporal_dims=temporal_dims,
            d_model=args.d_model,
            n_heads=args.n_heads,
            dropout=0.15
        ).to(device)
        
        # Train TFT
        model = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
        
        # Predict with TFT
        preds_tft = predict(model, test_loader, device)
        y_test = df_test[target_col].values
        
        # Compute TFT Metrics
        r2_tft = r2_score(y_test, preds_tft)
        rmse_tft = np.sqrt(mean_squared_error(y_test, preds_tft))
        mae_tft = mean_absolute_error(y_test, preds_tft)
        
        print(f"  TFT | Test R²: {r2_tft:.4f} | RMSE: {rmse_tft:.4f} | MAE: {mae_tft:.4f}")
        
        tft_cv_results.append({
            'fold': fold_idx+1,
            'r2': r2_tft,
            'rmse': rmse_tft,
            'mae': mae_tft,
            'tft_errs': np.abs(y_test - preds_tft)
        })
        
        # ----------------------------------------------------
        # RUN BASELINE ENSEMBLE TREE MODELS (HEAD-TO-HEAD)
        # ----------------------------------------------------
        cat_features_list = ["State", "District_mapped"]
        X_tr = df[train_mask][features].copy()
        y_tr = clip_target_by_state(df[train_mask], X_tr, df[train_mask][target_col], target_col, profile["clipping_quantile"])
        
        X_va, y_va = df[val_mask][features].copy(), df[val_mask][target_col].copy()
        X_te, y_te = df[test_mask][features].copy(), df[test_mask][target_col].copy()
        
        tree_preds = fit_models(
            X_tr, y_tr, X_va, y_va, X_te, y_te,
            cat_features=cat_features_list,
            cb_params=profile["cb_params"],
            xgb_params=profile["xgb_params"],
            lgb_params=profile["lgb_params"]
        )
        
        p_xgb = tree_preds['xgb']['test']
        p_lgb = tree_preds['lgb']['test']
        p_cb = tree_preds['cb']['test']
        p_ens = 0.10 * p_xgb + 0.10 * p_lgb + 0.80 * p_cb
        
        r2_ens = r2_score(y_te, p_ens)
        rmse_ens = np.sqrt(mean_squared_error(y_te, p_ens))
        mae_ens = mean_absolute_error(y_te, p_ens)
        
        print(f"  Ensemble | Test R²: {r2_ens:.4f} | RMSE: {rmse_ens:.4f} | MAE: {mae_ens:.4f}\n")
        
        tree_cv_results.append({
            'fold': fold_idx+1,
            'xgb_r2': r2_score(y_te, p_xgb),
            'lgb_r2': r2_score(y_te, p_lgb),
            'cb_r2': r2_score(y_te, p_cb),
            'ens_r2': r2_ens,
            'ens_rmse': rmse_ens,
            'ens_mae': mae_ens,
            'ens_errs': np.abs(y_te - p_ens)
        })

    # Average metrics
    avg_tft_r2 = np.mean([r['r2'] for r in tft_cv_results])
    avg_tft_rmse = np.mean([r['rmse'] for r in tft_cv_results])
    avg_tft_mae = np.mean([r['mae'] for r in tft_cv_results])
    
    avg_ens_r2 = np.mean([r['ens_r2'] for r in tree_cv_results])
    avg_ens_rmse = np.mean([r['ens_rmse'] for r in tree_cv_results])
    avg_ens_mae = np.mean([r['ens_mae'] for r in tree_cv_results])
    
    avg_cb_r2 = np.mean([r['cb_r2'] for r in tree_cv_results])
    avg_xgb_r2 = np.mean([r['xgb_r2'] for r in tree_cv_results])
    avg_lgb_r2 = np.mean([r['lgb_r2'] for r in tree_cv_results])
    
    from scipy import stats
    all_tft_errors = np.concatenate([r['tft_errs'] for r in tft_cv_results])
    all_ens_errors = np.concatenate([r['ens_errs'] for r in tree_cv_results])
    
    # Paired t-test and Wilcoxon signed-rank test on absolute errors
    t_stat, t_pval = stats.ttest_rel(all_tft_errors, all_ens_errors)
    w_stat, w_pval = stats.wilcoxon(all_tft_errors, all_ens_errors)
    
    # Effect sizes
    mae_diff = np.mean(all_ens_errors) - np.mean(all_tft_errors)
    
    # Cohen's d for paired samples (dz)
    diffs = all_ens_errors - all_tft_errors
    cohens_d = np.mean(diffs) / np.std(diffs, ddof=1)

    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS (TFT vs Ensemble)")
    print("="*80)
    print(f"Paired t-test p-value: {t_pval:.4f}")
    print(f"Wilcoxon signed-rank p-value: {w_pval:.4f}")
    print(f"Mean Absolute Error Difference (Ens - TFT): {mae_diff:.6f}")
    print(f"Cohen's d (dz): {cohens_d:.6f}")

    print("\n" + "="*80)
    print("HEAD-TO-HEAD CHRONOLOGICAL OUT-OF-TIME PERFORMANCE COMPARISON")
    print("="*80)
    
    summary_data = [
        ["TFT (Ours)", f"{tft_cv_results[0]['r2']:.4f}", f"{tft_cv_results[1]['r2']:.4f}", f"{tft_cv_results[2]['r2']:.4f}", f"{avg_tft_r2:.4f}", f"{avg_tft_rmse:.4f}", f"{avg_tft_mae:.4f}"],
        ["CatBoost (Tuned)", f"{tree_cv_results[0]['cb_r2']:.4f}", f"{tree_cv_results[1]['cb_r2']:.4f}", f"{tree_cv_results[2]['cb_r2']:.4f}", f"{avg_cb_r2:.4f}", "-", "-"],
        ["XGBoost (Tuned)", f"{tree_cv_results[0]['xgb_r2']:.4f}", f"{tree_cv_results[1]['xgb_r2']:.4f}", f"{tree_cv_results[2]['xgb_r2']:.4f}", f"{avg_xgb_r2:.4f}", "-", "-"],
        ["LightGBM (Tuned)", f"{tree_cv_results[0]['lgb_r2']:.4f}", f"{tree_cv_results[1]['lgb_r2']:.4f}", f"{tree_cv_results[2]['lgb_r2']:.4f}", f"{avg_lgb_r2:.4f}", "-", "-"],
        ["Robust Ensemble", f"{tree_cv_results[0]['ens_r2']:.4f}", f"{tree_cv_results[1]['ens_r2']:.4f}", f"{tree_cv_results[2]['ens_r2']:.4f}", f"{avg_ens_r2:.4f}", f"{avg_ens_rmse:.4f}", f"{avg_ens_mae:.4f}"]
    ]
    
    headers = ["Model", "Fold 1 R² (17-18)", "Fold 2 R² (19-20)", "Fold 3 R² (21-22)", "Avg R²", "Avg RMSE", "Avg MAE"]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))
    print("\nDone!\n")

if __name__ == "__main__":
    main()
