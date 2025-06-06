import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and explore both training and test data"""
    train_df = pd.read_csv('traing_data.csv')
    test_df = pd.read_csv('test_data.csv')
    
    print("=== DATA EXPLORATION ===")
    print(f"Training samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    
    print(f"\nTraining columns: {len(train_df.columns)}")
    print(f"Test columns: {len(test_df.columns)}")
    print(f"Missing in test: {set(train_df.columns) - set(test_df.columns)}")
    
    print(f"\nTarget (TimeInSeconds) distribution:")
    print(f"Min: {train_df['TimeInSeconds'].min():.1f} seconds")
    print(f"Max: {train_df['TimeInSeconds'].max():.1f} seconds")
    print(f"Mean: {train_df['TimeInSeconds'].mean():.1f} seconds")
    print(f"Median: {train_df['TimeInSeconds'].median():.1f} seconds")
    
    return train_df, test_df

def engineer_features(df):
    """Focused feature engineering on most predictive features"""
    df = df.copy()
    
    # Handle missing Cores values
    df['Cores'] = pd.to_numeric(df['Cores'], errors='coerce').fillna(3.0)
    
    # Parse SimnetVersion
    df['SimnetVersion'] = df['SimnetVersion'].astype(str)
    version_parts = df['SimnetVersion'].str.extract(r'(\d+)\.(\d+)\.(\d+)')
    df['version_major'] = pd.to_numeric(version_parts[0], errors='coerce').fillna(3)
    df['version_minor'] = pd.to_numeric(version_parts[1], errors='coerce').fillna(1)
    df['version_patch'] = pd.to_numeric(version_parts[2], errors='coerce').fillna(0)
    
    # Handle special Resolution values
    df.loc[df['Resolution'] == 'MAX', 'Resolution'] = 100
    df['Resolution'] = pd.to_numeric(df['Resolution'], errors='coerce').fillna(30)
    
    # Resolution transformations
    df['resolution_squared'] = df['Resolution'] ** 2
    df['resolution_log'] = np.log1p(df['Resolution'])
    
    # Convert numeric fields
    numeric_fields = ['SitesUsed', 'SitesNotUsed', 'SubscibersUsed', 'EvalAreaInSquareKilometers']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
    
    # Basic features
    df['total_sites'] = df['SitesUsed'] + df['SitesNotUsed']
    df['area_sqrt'] = np.sqrt(df['EvalAreaInSquareKilometers'])
    df['area_log'] = np.log1p(df['EvalAreaInSquareKilometers'])
    
    # Core utilization
    df['sites_per_core'] = df['total_sites'] / df['Cores']
    df['area_per_core'] = df['EvalAreaInSquareKilometers'] / df['Cores']
    df['resolution_per_core'] = df['Resolution'] / df['Cores']
    
    # Parse Results field
    df['Results'] = df['Results'].fillna('')
    df['result_count'] = df['Results'].str.count('\\|') + 1
    df['result_count'] = df['result_count'].where(df['Results'] != '', 0)
    
    # Key result types
    result_types = ['terrain', 'reliability', 'contour', 'delay_spread', 
                   'round_trip', 'voyager_grid', 'critical_buildings']
    
    for result_type in result_types:
        df[f'has_{result_type}'] = df['Results'].str.contains(
            result_type, case=False, na=False
        ).astype(int)
    
    # Parse Analysis field
    df['Analysis'] = df['Analysis'].fillna('')
    df['analysis_count'] = df['Analysis'].str.count('\\|') + 1
    df['analysis_count'] = df['analysis_count'].where(df['Analysis'] != '', 0)
    
    # Parse AdditionalLayers
    df['AdditionalLayers'] = df['AdditionalLayers'].fillna('')
    df['layer_count'] = df['AdditionalLayers'].str.count('\\|') + 1
    df['layer_count'] = df['layer_count'].where(df['AdditionalLayers'] != '', 0)
    
    # Boolean features
    bool_cols = ['UseNewCatpMode', 'Outbound', 'Inbound', 'Roundtrip', 
                'WorstDirection', 'VoiceTrafficUsed', 'MakeVoyagerGrid']
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(int)
    
    # Direction complexity
    df['direction_count'] = df[['Outbound', 'Inbound', 'Roundtrip', 'WorstDirection']].sum(axis=1)
    
    # Simple complexity score
    df['complexity_score'] = (
        df['Resolution'] * 
        df['total_sites'] * 
        df['area_sqrt'] * 
        (1 + df['result_count']) / 
        df['Cores']
    )
    
    # Mode encoding
    df['Mode'] = df['Mode'].fillna('Normal')
    mode_map = {'Fine': 2, 'Normal': 1, 'Draft': 0}
    df['mode_encoded'] = df['Mode'].map(mode_map).fillna(1)
    
    return df

def select_features():
    """Select proven features"""
    return [
        # Core features
        'Cores', 'Resolution', 'resolution_squared', 'resolution_log',
        'total_sites', 'SitesUsed', 'SitesNotUsed',
        'SubscibersUsed', 'EvalAreaInSquareKilometers', 'area_sqrt', 'area_log',
        
        # Version
        'version_major', 'version_minor',
        
        # Complexity
        'result_count', 'analysis_count', 'layer_count', 'direction_count',
        'complexity_score',
        
        # Utilization
        'sites_per_core', 'area_per_core', 'resolution_per_core',
        
        # Result types
        'has_terrain', 'has_reliability', 'has_contour', 
        'has_delay_spread', 'has_round_trip', 'has_voyager_grid',
        'has_critical_buildings',
        
        # Boolean flags
        'UseNewCatpMode', 'Outbound', 'Inbound', 'Roundtrip', 'WorstDirection',
        'VoiceTrafficUsed', 'MakeVoyagerGrid',
        
        # Encoded
        'mode_encoded'
    ]

def build_model(X_train, y_train, X_val=None, y_val=None):
    """Build LightGBM model with balanced parameters"""
    params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'max_depth': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_gain_to_split': 0.01,
        'verbosity': -1,
        'random_state': 42
    }
    
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_set]
    
    if X_val is not None:
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        valid_sets.append(val_set)
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    return model

def analyze_errors(y_true, y_pred):
    """Analyze prediction errors to understand MAPE issues"""
    errors = np.abs(y_true - y_pred) / y_true * 100
    
    print("\n=== ERROR ANALYSIS ===")
    print(f"MAPE: {np.mean(errors):.2f}%")
    print(f"Median APE: {np.median(errors):.2f}%")
    print(f"90th percentile APE: {np.percentile(errors, 90):.2f}%")
    
    # Find worst predictions
    worst_idx = np.argsort(errors)[-10:]
    print("\nWorst 10 predictions:")
    for idx in worst_idx:
        print(f"  True: {y_true.iloc[idx]:.1f}, Pred: {y_pred[idx]:.1f}, APE: {errors.iloc[idx]:.1f}%")
    
    # Analyze by value ranges
    print("\nMAPE by target value range:")
    ranges = [(0, 10), (10, 50), (50, 100), (100, 500), (500, 1000), (1000, np.inf)]
    for low, high in ranges:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            range_mape = np.mean(errors[mask])
            print(f"  [{low:4d}, {high:4.0f}): {range_mape:6.2f}% (n={mask.sum()})")

def apply_smart_post_processing(predictions, training_targets):
    """Smart post-processing based on error analysis"""
    # Get training distribution
    p05 = np.percentile(training_targets, 5)
    p95 = np.percentile(training_targets, 95)
    
    # Moderate adjustments for small values
    predictions = np.where(predictions < 5, predictions * 1.2, predictions)
    predictions = np.where(predictions < 10, predictions * 1.1, predictions)
    
    # Soft bounds
    predictions = np.where(predictions < p05 * 0.5, p05 * 0.7, predictions)
    predictions = np.where(predictions > p95 * 2, p95 * 1.5, predictions)
    
    # Ensure minimum
    predictions = np.maximum(predictions, 1)
    
    return predictions

def main():
    print("=== HYDRA STRATUS TIME PREDICTION - TARGET 40% MAPE ===\n")
    
    # Load data
    train_df, test_df = load_and_explore_data()
    
    # Feature engineering
    print("\n=== FEATURE ENGINEERING ===")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # Filter for version 3+ which matches test data better
    train_filtered = train_df[train_df['version_major'] >= 3].copy()
    print(f"Training samples after version filter: {len(train_filtered):,}")
    
    # Remove outliers
    time_p995 = train_filtered['TimeInSeconds'].quantile(0.995)
    train_clean = train_filtered[train_filtered['TimeInSeconds'] <= time_p995].copy()
    print(f"Training samples after outlier removal: {len(train_clean):,}")
    
    # Prepare features
    features = select_features()
    X = train_clean[features]
    y = train_clean['TimeInSeconds']
    
    # Log transform target
    y_log = np.log1p(y)
    
    # Cross-validation with error analysis
    print("\n=== CROSS-VALIDATION ===")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y.iloc[val_idx]
        
        model = build_model(X_train, y_train, X_val, np.log1p(y_val))
        models.append(model)
        
        # Predict and evaluate
        val_pred = np.expm1(model.predict(X_val))
        val_pred = np.maximum(val_pred, 1)
        
        # Apply smart post-processing
        val_pred = apply_smart_post_processing(val_pred, y)
        
        mape = mean_absolute_percentage_error(y_val, val_pred) * 100
        cv_scores.append(mape)
        print(f"Fold {fold + 1}: MAPE = {mape:.2f}%")
        
        # Analyze errors on first fold
        if fold == 0:
            analyze_errors(y_val, val_pred)
    
    print(f"\nMean CV MAPE: {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")
    
    # Use ensemble of CV models
    print("\n=== MAKING ENSEMBLE PREDICTIONS ===")
    X_test = test_df[features]
    
    # Average predictions from CV models
    test_preds = []
    for model in models:
        pred = np.expm1(model.predict(X_test))
        test_preds.append(pred)
    
    test_pred = np.mean(test_preds, axis=0)
    test_pred = np.maximum(test_pred, 1)
    
    # Apply smart post-processing
    test_pred = apply_smart_post_processing(test_pred, y)
    
    # Save predictions
    np.savetxt('prediction1g.txt', test_pred, fmt='%.6f')
    
    print("\n=== PREDICTION SUMMARY ===")
    print(f"Predictions saved: prediction1g.txt")
    print(f"Number of predictions: {len(test_pred)}")
    print(f"Min: {test_pred.min():.1f} seconds")
    print(f"Max: {test_pred.max():.1f} seconds")
    print(f"Mean: {test_pred.mean():.1f} seconds")
    print(f"Median: {np.median(test_pred):.1f} seconds")
    
    # Feature importance
    print("\n=== TOP 15 MOST IMPORTANT FEATURES ===")
    importance = models[0].feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for idx, (_, row) in enumerate(feature_imp.head(15).iterrows()):
        print(f"{idx+1}. {row['feature']}: {row['importance']:.1f}")

if __name__ == "__main__":
    main()