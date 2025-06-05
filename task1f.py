import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna 
import traceback # For more detailed error printing

def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask): return np.nan if np.any(y_pred != 0) else 0.0
    y_true_masked, y_pred_masked = y_true[mask], y_pred[mask]
    if len(y_true_masked) == 0: return np.nan
    return np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100

def apply_target_encoding(train_df, test_df, column, target_log, fit_map=True, te_map=None, global_mean=None, smoothing=10):
    """ Applies target encoding to a column. """
    if fit_map:
        if global_mean is None: # Calculate global mean only once if not provided
            global_mean = target_log.mean()
        
        agg = target_log.groupby(train_df[column]).agg(['mean', 'count'])
        counts = agg['count']
        means = agg['mean']
        # Smoothed mean
        smooth = (counts * means + global_mean * smoothing) / (counts + smoothing)
        
        te_map = smooth.to_dict()
        train_df[column + '_te'] = train_df[column].map(te_map).fillna(global_mean)
        if test_df is not None:
            test_df[column + '_te'] = test_df[column].map(te_map).fillna(global_mean)
        return train_df, test_df, te_map, global_mean
    else:
        # Apply pre-fitted map
        train_df[column + '_te'] = train_df[column].map(te_map).fillna(global_mean)
        if test_df is not None:
            test_df[column + '_te'] = test_df[column].map(te_map).fillna(global_mean)
        return train_df, test_df


def preprocess_data(train_df_raw_features, test_df_raw_features, y_train_target_for_te=None,
                    fit_transformers=True, scalers_dict=None, medians_dict=None, modes_dict=None, 
                    log_transform_cols_list=None, target_encode_cols_list=None, 
                    target_encode_maps_dict=None, global_target_means_dict=None,
                    processed_train_columns_schema=None):
    """
    Preprocesses feature dataframes.
    Version 7: Incorporates target encoding logic.
    """
    print(f"Preprocessing... Fit: {fit_transformers}, Train shape: {train_df_raw_features.shape}, Test/Val shape: {test_df_raw_features.shape if test_df_raw_features is not None else 'N/A'}")

    X_train = train_df_raw_features.copy()
    X_test = test_df_raw_features.copy() if test_df_raw_features is not None else None
    
    resolution_col_name = 'Resolution'
    for df in [X_train, X_test]:
        if df is None: continue
        if resolution_col_name in df.columns:
            df[resolution_col_name] = pd.to_numeric(df[resolution_col_name], errors='coerce')

    # --- Feature Engineering ---
    for df in [X_train, X_test]:
        if df is None: continue
        # Impute Resolution NaNs before using it in division (use median from training data)
        res_median_val = medians_dict.get(resolution_col_name, 0) if not fit_transformers and medians_dict else (df[resolution_col_name].median() if resolution_col_name in df.columns else 0)
        if resolution_col_name in df.columns:
            df[resolution_col_name] = df[resolution_col_name].fillna(res_median_val)

        if 'Cores' in df.columns and 'EvalAreaInSquareKilometers' in df.columns: df['Cores_x_EvalArea'] = df['Cores'] * df['EvalAreaInSquareKilometers']
        if 'SitesUsed' in df.columns and resolution_col_name in df.columns: df['SitesUsed_div_Resolution'] = df['SitesUsed'] / (df[resolution_col_name] + 1e-6)
        elif 'SitesUsed' in df.columns: df['SitesUsed_div_Resolution'] = 0 
        if 'SitesUsed' in df.columns and 'SitesNotUsed' in df.columns: df['SitesUsed_to_TotalSites_Ratio'] = df['SitesUsed'] / (df['SitesUsed'] + df['SitesNotUsed'] + 1e-6)
        if 'EvalAreaInSquareKilometers' in df.columns and 'SitesUsed' in df.columns: df['EvalArea_per_SiteUsed'] = df['EvalAreaInSquareKilometers'] / (df['SitesUsed'] + 1e-6)
        if 'SubscibersUsed' in df.columns and 'SitesUsed' in df.columns: df['Subscribers_per_SiteUsed'] = df['SubscibersUsed'] / (df['SitesUsed'] + 1e-6)
    
    # Boolean mapping
    bool_like_map = {'True': 1, 'False': 0, True: 1, False: 0} 
    bool_cols_to_map = ['VoiceTrafficUsed', 'MakeVoyagerGrid'] 
    if 'UseNewCatpMode' in X_train.columns:
        if all(X_train['UseNewCatpMode'].dropna().isin([True, False, 'True', 'False'])): bool_cols_to_map.append('UseNewCatpMode')
    for df in [X_train, X_test]:
        if df is None: continue
        for col in bool_cols_to_map:
            if col in df.columns: df[col] = df[col].map(bool_like_map).fillna(0) 

    # Identify column types
    categorical_cols_for_ohe = []
    numerical_cols = []
    
    temp_schema_cols = X_train.columns 
    for col in temp_schema_cols:
        is_target_encoded = target_encode_cols_list and col in target_encode_cols_list
        if is_target_encoded: continue # Skip if it will be target encoded

        if col in bool_cols_to_map: numerical_cols.append(col)
        elif pd.api.types.is_bool_dtype(X_train[col].dtype):
            for df in [X_train, X_test]: 
                if df is not None and col in df.columns: df[col] = df[col].astype(int)
            numerical_cols.append(col)
        elif X_train[col].dtype == 'object' or (X_train[col].nunique(dropna=False) < 30 and not pd.api.types.is_numeric_dtype(X_train[col].dtype)): 
            categorical_cols_for_ohe.append(col)
        elif pd.api.types.is_numeric_dtype(X_train[col].dtype): numerical_cols.append(col)
        else: print(f"Warning: Col '{col}' type {X_train[col].dtype} not categorized. Treating as cat for OHE."); categorical_cols_for_ohe.append(col)
            
    print(f"Numerical cols: {numerical_cols}, Categorical for OHE: {categorical_cols_for_ohe}")
    if target_encode_cols_list: print(f"Target Encoding cols: {target_encode_cols_list}")

    # Imputation (before log/target encoding for numericals)
    if fit_transformers:
        medians_dict = {}
        modes_dict = {}
        if target_encode_cols_list: 
            target_encode_maps_dict = {}
            global_target_means_dict = {}

    for col in numerical_cols:
        if col in X_train.columns: 
            median_val = X_train[col].median() if fit_transformers else medians_dict.get(col, X_train[col].median())
            if fit_transformers: medians_dict[col] = median_val
            X_train[col].fillna(median_val, inplace=True)
        if X_test is not None and col in X_test.columns: 
            X_test[col].fillna(medians_dict.get(col, X_test[col].median()), inplace=True)
    
    if log_transform_cols_list:
        for col in log_transform_cols_list:
            if col in X_train.columns and col in numerical_cols: X_train[col] = np.log1p(X_train[col])
            if X_test is not None and col in X_test.columns and col in numerical_cols: X_test[col] = np.log1p(X_test[col])

    # Target Encoding
    if target_encode_cols_list:
        for col in target_encode_cols_list:
            if col not in X_train.columns: continue
            print(f"Target encoding: {col}")
             # Impute NaNs in the categorical column before target encoding
            cat_mode = X_train[col].mode(dropna=True)
            cat_mode_val = cat_mode[0] if not cat_mode.empty else "Missing_TE_Placeholder"
            if fit_transformers: modes_dict[col + "_orig_cat_impute"] = cat_mode_val # Store imputation for original
            
            X_train[col].fillna(cat_mode_val if fit_transformers else modes_dict.get(col + "_orig_cat_impute", "Missing_TE_Placeholder"), inplace=True)
            if X_test is not None and col in X_test.columns:
                X_test[col].fillna(modes_dict.get(col + "_orig_cat_impute", "Missing_TE_Placeholder"), inplace=True)


            if fit_transformers:
                X_train, X_test, te_map, global_mean = apply_target_encoding(X_train, X_test, col, y_train_target_for_te, fit_map=True)
                target_encode_maps_dict[col] = te_map
                global_target_means_dict[col] = global_mean
            else:
                X_train, X_test = apply_target_encoding(X_train, X_test, col, y_train_target_for_te, fit_map=False, 
                                                        te_map=target_encode_maps_dict.get(col),
                                                        global_mean=global_target_means_dict.get(col))
            # Add new TE feature to numerical_cols list to be scaled
            if (col + '_te') not in numerical_cols: numerical_cols.append(col + '_te')
        # Drop original categorical columns that were target encoded
        X_train.drop(columns=target_encode_cols_list, errors='ignore', inplace=True)
        if X_test is not None: X_test.drop(columns=target_encode_cols_list, errors='ignore', inplace=True)
    
    # Impute OHE categorical columns
    for col in categorical_cols_for_ohe:
        if col in X_train.columns:
            mode_val = X_train[col].mode(dropna=True); mode_val = mode_val[0] if not mode_val.empty else "Missing"
            if fit_transformers: modes_dict[col] = mode_val
            else: mode_val = modes_dict.get(col, mode_val)
            X_train[col].fillna(mode_val, inplace=True); X_train[col] = X_train[col].astype(str) 
        if X_test is not None and col in X_test.columns: 
            test_mode_val = modes_dict.get(col, (X_test[col].mode(dropna=True)[0] if not X_test[col].mode(dropna=True).empty else "Missing"))
            X_test[col].fillna(test_mode_val, inplace=True); X_test[col] = X_test[col].astype(str)

    # One-Hot Encoding for remaining categoricals
    if categorical_cols_for_ohe:
        print(f"OHE for: {categorical_cols_for_ohe}")
        # Ensure columns exist before trying to OHE
        actual_ohe_cols_train = [c for c in categorical_cols_for_ohe if c in X_train.columns]
        if actual_ohe_cols_train:
            X_train = pd.get_dummies(X_train, columns=actual_ohe_cols_train, prefix=actual_ohe_cols_train, prefix_sep='_', dummy_na=False)
        if X_test is not None:
            actual_ohe_cols_test = [c for c in categorical_cols_for_ohe if c in X_test.columns]
            if actual_ohe_cols_test:
                 X_test = pd.get_dummies(X_test, columns=actual_ohe_cols_test, prefix=actual_ohe_cols_test, prefix_sep='_', dummy_na=False)
    
    # Align columns
    if fit_transformers:
        processed_train_columns_schema = X_train.columns.tolist()
    
    if X_test is not None and processed_train_columns_schema is not None:
        for col in processed_train_columns_schema: 
            if col not in X_test.columns: X_test[col] = 0 
        cols_to_drop_from_test = [col for col in X_test.columns if col not in processed_train_columns_schema]
        if cols_to_drop_from_test: X_test.drop(columns=cols_to_drop_from_test, inplace=True)
        X_test = X_test[processed_train_columns_schema] 
    
    if not fit_transformers and processed_train_columns_schema is not None: # Align X_train if it's a validation set
        for col in processed_train_columns_schema:
            if col not in X_train.columns: X_train[col] = 0
        cols_to_drop_from_train_val = [col for col in X_train.columns if col not in processed_train_columns_schema]
        if cols_to_drop_from_train_val: X_train.drop(columns=cols_to_drop_from_train_val, inplace=True)
        X_train = X_train[processed_train_columns_schema]

    print(f"Shape after OHE/TE & align: Train {X_train.shape}, Test {X_test.shape if X_test is not None else 'N/A'}")
    
    # Scaling numerical features (includes original, bools, new engineered, and TE features)
    if fit_transformers:
        scalers_dict = {} 
    
    # Ensure numerical_cols is up-to-date if TE features were added
    final_numerical_cols_to_scale = [nc for nc in numerical_cols if nc in X_train.columns] # Only scale if they exist

    if final_numerical_cols_to_scale:
        print(f"Scaling numerical columns: {final_numerical_cols_to_scale}")
        for col in final_numerical_cols_to_scale:
            if not pd.api.types.is_numeric_dtype(X_train[col]): continue # Skip if not numeric
            if X_test is not None and col in X_test.columns and not pd.api.types.is_numeric_dtype(X_test[col]): continue

            current_scaler = None
            if fit_transformers:
                current_scaler = StandardScaler()
                X_train[[col]] = current_scaler.fit_transform(X_train[[col]])
                scalers_dict[col] = current_scaler 
            else: 
                current_scaler = scalers_dict.get(col)
                if current_scaler: X_train[[col]] = current_scaler.transform(X_train[[col]])
            
            if X_test is not None and col in X_test.columns:
                test_scaler = scalers_dict.get(col) 
                if test_scaler : X_test[[col]] = test_scaler.transform(X_test[[col]])
    
    print("Preprocessing finished.")
    if fit_transformers: 
        return X_train, X_test, scalers_dict, medians_dict, modes_dict, target_encode_maps_dict, global_target_means_dict, processed_train_columns_schema
    else: 
        return X_train, X_test

# Optuna objective function with K-Fold CV
def objective(trial, X_features, y_log_target, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_mapes = []
    
    # Define base columns for log transform (can be refined)
    base_cols_for_log_transform = ['Cores', 'Resolution', 'SitesUsed', 'SitesNotUsed', 'SubscibersUsed', 
                                   'EvalAreaInSquareKilometers'] 
    # Define base columns for target encoding (can be refined)
    base_cols_for_target_encoding = ['SimnetVersion', 'Results', 'Analysis', 'AdditionalLayers', 'Mode', 'Version']


    for fold, (train_idx, val_idx) in enumerate(kf.split(X_features, y_log_target)):
        print(f"\n--- Optuna Trial {trial.number}, Fold {fold+1}/{n_folds} ---")
        X_train_fold, X_val_fold = X_features.iloc[train_idx], X_features.iloc[val_idx]
        y_train_fold, y_val_fold = y_log_target.iloc[train_idx], y_log_target.iloc[val_idx]

        # Determine actual log/TE columns present in this fold's training data
        actual_log_cols = [c for c in base_cols_for_log_transform if c in X_train_fold.columns]
        engineered_cols = ['Cores_x_EvalArea', 'SitesUsed_div_Resolution', 'SitesUsed_to_TotalSites_Ratio', 
                           'EvalArea_per_SiteUsed', 'Subscribers_per_SiteUsed']
        actual_log_cols.extend([ec for ec in engineered_cols if ec in X_train_fold.columns])
        actual_log_cols = list(set(actual_log_cols)) # Unique

        actual_te_cols = [c for c in base_cols_for_target_encoding if c in X_train_fold.columns]
        
        # Preprocess: fit on X_train_fold, transform X_train_fold and X_val_fold
        X_train_fold_processed, X_val_fold_processed, \
        fitted_scalers, fitted_medians, fitted_modes, \
        fitted_te_maps, fitted_global_means, fold_train_schema = \
            preprocess_data(X_train_fold, X_val_fold, y_train_target_for_te=y_train_fold, 
                            fit_transformers=True, log_transform_cols_list=actual_log_cols,
                            target_encode_cols_list=actual_te_cols)

        lgb_params = {
            'objective': 'regression_l1', 'metric': 'mae', 'verbose': -1, 'n_jobs': -1, 'seed': 42 + fold,
            'n_estimators': trial.suggest_int('n_estimators', 1500, 5000, step=500),
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.03),
            'num_leaves': trial.suggest_int('num_leaves', 25, 150),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 70),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.8),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 0.8),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7, step=2),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 5.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 5.0, log=True),
        }
        model = lgb.LGBMRegressor(**lgb_params)
        try:
            model.fit(X_train_fold_processed, y_train_fold,
                      eval_set=[(X_val_fold_processed, y_val_fold)],
                      callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)])
        except Exception as e:
            print(f"Error during model fit in Optuna trial {trial.number}, fold {fold+1}: {e}")
            traceback.print_exc()
            return float('inf') # Return high MAPE if error occurs

        preds_log = model.predict(X_val_fold_processed)
        preds_actual = np.expm1(preds_log); preds_actual = np.maximum(0, preds_actual)
        y_val_actual = np.expm1(y_val_fold)
        
        mape = calculate_mape(y_val_actual, preds_actual)
        if np.isnan(mape):
            print(f"Warning: MAPE is NaN in trial {trial.number}, fold {fold+1}. Returning high value.")
            return float('inf') # Penalize NaN MAPE
        fold_mapes.append(mape)
        print(f"Trial {trial.number}, Fold {fold+1} MAPE: {mape:.2f}%")

    avg_mape = np.mean(fold_mapes)
    print(f"Trial {trial.number} Average CV MAPE: {avg_mape:.2f}%")
    return avg_mape

# --- Main script execution ---
if __name__ == "__main__":
    try:
        full_train_df_with_target = pd.read_csv('traing_data.csv', low_memory=False)
        submission_test_df_features_raw = pd.read_csv('test_data.csv', low_memory=False)
        if 'TaskId' in submission_test_df_features_raw.columns: submission_test_df_features_raw.drop(columns=['TaskId'], inplace=True)
        if 'WorkspaceId' in submission_test_df_features_raw.columns: submission_test_df_features_raw.drop(columns=['WorkspaceId'], inplace=True)
    except Exception as e: print(f"Error loading data: {e}."); exit()

    # Separate features and target for Optuna CV (using the full training data)
    y_full_log_target = np.log1p(full_train_df_with_target['TimeInSeconds'])
    X_full_features = full_train_df_with_target.drop(columns=['TimeInSeconds', 'TaskId', 'WorkspaceId'], errors='ignore')
    
    if y_full_log_target.isnull().any():
        valid_indices = ~y_full_log_target.isnull()
        X_full_features = X_full_features[valid_indices]
        y_full_log_target = y_full_log_target[valid_indices]
    if X_full_features.empty or y_full_log_target.empty: exit("Critical Error: Full data empty before Optuna.")

    # --- Optuna Hyperparameter Search ---
    print("\nStarting Optuna hyperparameter search with K-Fold CV...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce Optuna's default logging
    study = optuna.create_study(direction='minimize') 
    study.optimize(lambda trial: objective(trial, X_full_features, y_full_log_target, n_folds=5), 
                   n_trials=75, timeout=1800) # Increased trials, added timeout (30 mins)

    print("\nOptuna search finished.")
    print(f"Best CV MAPE: {study.best_value:.2f}%")
    best_params_optuna = study.best_params
    print("Best hyperparameters found by Optuna:"); print(best_params_optuna)
    
    # --- Retrain model with best Optuna params on FULL training data ---
    print("\nRetraining model on FULL training data with best Optuna parameters...")
    
    # Determine log/TE columns for the full dataset run
    base_cols_for_log_transform = ['Cores', 'Resolution', 'SitesUsed', 'SitesNotUsed', 'SubscibersUsed', 
                                   'EvalAreaInSquareKilometers'] 
    engineered_cols = ['Cores_x_EvalArea', 'SitesUsed_div_Resolution', 'SitesUsed_to_TotalSites_Ratio', 
                       'EvalArea_per_SiteUsed', 'Subscribers_per_SiteUsed']
    actual_cols_for_log_transform_full = [c for c in base_cols_for_log_transform if c in X_full_features.columns]
    actual_cols_for_log_transform_full.extend([ec for ec in engineered_cols if ec in X_full_features.columns])
    actual_cols_for_log_transform_full = list(set(actual_cols_for_log_transform_full))

    base_cols_for_target_encoding = ['SimnetVersion', 'Results', 'Analysis', 'AdditionalLayers', 'Mode', 'Version']
    actual_te_cols_full = [c for c in base_cols_for_target_encoding if c in X_full_features.columns]
    
    X_train_full_processed, X_submission_test_processed, \
    final_scalers, final_medians, final_modes, \
    final_te_maps, final_global_means, final_train_schema = \
        preprocess_data(X_full_features, submission_test_df_features_raw, y_train_target_for_te=y_full_log_target,
                        fit_transformers=True, log_transform_cols_list=actual_cols_for_log_transform_full,
                        target_encode_cols_list=actual_te_cols_full)

    if X_train_full_processed.empty or y_full_log_target.empty: exit("Critical Error: Full training data empty for final model.")

    final_lgb_params = {
        'objective': 'regression_l1', 'metric': 'mae', 'verbose': -1, 'n_jobs': -1, 'seed': 42,
    }
    final_lgb_params.update(best_params_optuna) 
    
    final_model = lgb.LGBMRegressor(**final_lgb_params) 
    print(f"Retraining final model with Optuna params (n_estimators={final_lgb_params.get('n_estimators')}).")
    final_model.fit(X_train_full_processed, y_full_log_target)
    print("Final model retraining complete.")

    if not X_train_full_processed.empty and not y_full_log_target.empty:
        preds_log = final_model.predict(X_train_full_processed)
        preds_actual = np.expm1(preds_log); preds_actual = np.maximum(0, preds_actual)
        y_actual = np.expm1(y_full_log_target)
        mape_final_train = calculate_mape(y_actual, preds_actual)
        print(f"MAPE of final model on FULL training data: {mape_final_train:.2f}%")

    print("\nMaking predictions on the submission test set...")
    try:
        submission_predictions_log = final_model.predict(X_submission_test_processed)
        submission_predictions_actual = np.expm1(submission_predictions_log) 
        submission_predictions_actual = np.maximum(0, submission_predictions_actual) 
        print("Submission predictions made successfully.")
    except Exception as e: print(f"Error during submission prediction: {e}"); traceback.print_exc(); exit()

    output_filename = "prediction1f.txt"
    try:
        with open(output_filename, 'w') as f:
            for val in submission_predictions_actual: f.write(f"{val}\n")
        print(f"Submission predictions saved to {output_filename}")
    except Exception as e: print(f"Error saving predictions: {e}")
    print("\nScript finished.")
