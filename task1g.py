import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def understand_the_problem():
    """
    Key insights about Hydra Stratus simulation time prediction:
    
    1. Network coverage simulation is computationally intensive
    2. Time varies from seconds to days - huge range!
    3. Missing factors: terrain topology (major impact but not in data)
    4. Test data only has versions >= 3.1, but training has older versions
    5. MAPE metric means relative errors matter more for small values
    """
    pass

def load_and_explore_data():
    """Initial data exploration"""
    train_df = pd.read_csv('traing_data.csv')
    test_df = pd.read_csv('test_data.csv')
    
    print("=== DATA EXPLORATION ===")
    print(f"Training samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"\nTarget distribution:")
    print(f"Min time: {train_df['TimeInSeconds'].min():.1f} seconds")
    print(f"Max time: {train_df['TimeInSeconds'].max():.1f} seconds")
    print(f"Mean time: {train_df['TimeInSeconds'].mean():.1f} seconds")
    print(f"Median time: {train_df['TimeInSeconds'].median():.1f} seconds")
    
    return train_df, test_df

def engineer_core_features(df):
    """
    Features based on computational complexity understanding
    """
    df = df.copy()
    
    # Parse SimnetVersion intelligently
    df['SimnetVersion'] = df['SimnetVersion'].astype(str)
    version_split = df['SimnetVersion'].str.split('.', expand=True)
    df['version_major'] = pd.to_numeric(version_split[0], errors='coerce').fillna(3)
    df['version_minor'] = pd.to_numeric(version_split[1], errors='coerce').fillna(1)
    
    # Convert numeric fields
    numeric_fields = ['Cores', 'Resolution', 'SitesUsed', 'SitesNotUsed', 
                     'SubscibersUsed', 'EvalAreaInSquareKilometers']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
    
    # Total computational elements
    df['total_sites'] = df['SitesUsed'] + df['SitesNotUsed']
    
    # Computational complexity indicators
    df['grid_points'] = df['Resolution'] * df['EvalAreaInSquareKilometers']
    df['site_density'] = df['total_sites'] / (df['EvalAreaInSquareKilometers'] + 0.1)
    df['compute_load'] = df['grid_points'] * df['total_sites'] / (df['Cores'] + 1)
    
    # Area-based features (simulation scales with area)
    df['area_sqrt'] = np.sqrt(df['EvalAreaInSquareKilometers'])
    df['area_log'] = np.log1p(df['EvalAreaInSquareKilometers'])
    
    # Resource utilization
    df['sites_per_core'] = df['total_sites'] / (df['Cores'] + 1)
    df['area_per_core'] = df['EvalAreaInSquareKilometers'] / (df['Cores'] + 1)
    df['resolution_factor'] = df['Resolution'] / 100  # Normalize resolution
    
    return df

def parse_complex_fields(df):
    """
    Parse string fields that encode complexity
    """
    # Results field - count complexity
    df['Results'] = df['Results'].fillna('')
    df['result_count'] = df['Results'].str.count('\\|') + 1
    df['result_count'] = df['result_count'].where(df['Results'] != '', 0)
    
    # Check for computationally expensive result types
    expensive_results = ['reliability', 'contour', 'terrain', 'interference']
    for result_type in expensive_results:
        df[f'has_{result_type}'] = df['Results'].str.contains(
            result_type, case=False, na=False
        ).astype(int)
    
    # Additional layers increase computation
    df['AdditionalLayers'] = df['AdditionalLayers'].fillna('')
    df['layer_count'] = df['AdditionalLayers'].str.count('\\|') + 1
    df['layer_count'] = df['layer_count'].where(df['AdditionalLayers'] != '', 0)
    
    # Analysis types
    df['Analysis'] = df['Analysis'].fillna('')
    df['analysis_count'] = df['Analysis'].str.count('\\|') + 1
    df['analysis_count'] = df['analysis_count'].where(df['Analysis'] != '', 0)
    
    # Direction complexity (radio propagation directions)
    direction_cols = ['Outbound', 'Inbound', 'Roundtrip', 'WorstDirection']
    for col in direction_cols:
        df[col] = df[col].fillna(False).astype(int)
    df['direction_complexity'] = df[direction_cols].sum(axis=1)
    
    # Boolean features
    bool_cols = ['UseNewCatpMode', 'VoiceTrafficUsed', 'MakeVoyagerGrid']
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(int)
    
    return df

def create_interaction_features(df):
    """
    Create features that capture interactions between variables
    """
    # Complexity score combining multiple factors
    df['complexity_score'] = (
        df['resolution_factor'] * 
        df['total_sites'] * 
        df['result_count'] * 
        (1 + df['layer_count']) * 
        (1 + df['analysis_count']) * 
        df['area_sqrt'] / 
        (df['Cores'] + 1)
    )
    
    # Version efficiency (newer versions might be optimized)
    df['version_efficiency'] = df['version_major'] + df['version_minor'] * 0.1
    df['is_modern_version'] = (df['version_major'] >= 3).astype(int)
    
    # High complexity indicators
    df['is_large_area'] = (df['EvalAreaInSquareKilometers'] > 100).astype(int)
    df['is_many_sites'] = (df['total_sites'] > 50).astype(int)
    df['is_high_resolution'] = (df['Resolution'] > 100).astype(int)
    
    # Categorical encoding
    df['Mode'] = df['Mode'].fillna('Unknown')
    mode_map = df['Mode'].value_counts().to_dict()
    df['mode_frequency'] = df['Mode'].map(mode_map)
    
    return df

def select_final_features():
    """
    Select features based on domain understanding
    """
    features = [
        # Core computational factors
        'Cores', 'Resolution', 'total_sites', 'EvalAreaInSquareKilometers',
        
        # Derived computational metrics
        'grid_points', 'site_density', 'compute_load', 'complexity_score',
        'area_sqrt', 'area_log',
        
        # Resource utilization
        'sites_per_core', 'area_per_core', 'resolution_factor',
        
        # Version info
        'version_major', 'version_minor', 'version_efficiency', 'is_modern_version',
        
        # Task complexity
        'result_count', 'layer_count', 'analysis_count', 'direction_complexity',
        
        # Expensive operations
        'has_reliability', 'has_contour', 'has_terrain', 'has_interference',
        
        # Boolean flags
        'UseNewCatpMode', 'VoiceTrafficUsed', 'MakeVoyagerGrid',
        'Outbound', 'Inbound', 'Roundtrip', 'WorstDirection',
        
        # High complexity indicators
        'is_large_area', 'is_many_sites', 'is_high_resolution',
        
        # Categorical
        'mode_frequency'
    ]
    
    return features

def build_model(X_train, y_train, X_val=None, y_val=None):
    """
    Build LightGBM model with parameters tuned for time prediction
    """
    params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'num_leaves': 63,  # 2^6 - 1 for efficiency
        'max_depth': 7,
        'learning_rate': 0.03,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 3,
        'min_child_samples': 30,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'min_gain_to_split': 0.02,
        'verbosity': -1,
        'random_state': 42,
        'force_row_wise': True
    }
    
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_set]
    
    if X_val is not None:
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        valid_sets.append(val_set)
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1500,
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )
    
    return model

def main():
    print("=== HYDRA STRATUS SIMULATION TIME PREDICTION ===\n")
    
    # Load data
    train_df, test_df = load_and_explore_data()
    
    # Feature engineering pipeline
    print("\n=== FEATURE ENGINEERING ===")
    train_df = engineer_core_features(train_df)
    train_df = parse_complex_fields(train_df)
    train_df = create_interaction_features(train_df)
    
    test_df = engineer_core_features(test_df)
    test_df = parse_complex_fields(test_df)
    test_df = create_interaction_features(test_df)
    
    # Focus on relevant data (version >= 3.0 to include more data)
    train_modern = train_df[train_df['version_major'] >= 3].copy()
    print(f"Training samples after version filter: {len(train_modern):,}")
    
    # Handle outliers conservatively (keep 99.5% of data)
    time_cap = train_modern['TimeInSeconds'].quantile(0.995)
    train_clean = train_modern[train_modern['TimeInSeconds'] <= time_cap].copy()
    print(f"Training samples after outlier removal: {len(train_clean):,}")
    
    # Prepare features
    features = select_final_features()
    features = [f for f in features if f in train_clean.columns and f in test_df.columns]
    print(f"Final feature count: {len(features)}")
    
    X = train_clean[features].fillna(0)
    y = train_clean['TimeInSeconds']
    
    # Log transform target
    y_log = np.log1p(y)
    
    # Cross-validation
    print("\n=== CROSS-VALIDATION ===")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_log.iloc[train_idx], y.iloc[val_idx]
        
        model = build_model(X_train, y_train, X_val, np.log1p(y_val))
        
        # Predict and evaluate
        val_pred = np.expm1(model.predict(X_val))
        val_pred = np.maximum(val_pred, 1)
        
        mape = mean_absolute_percentage_error(y_val, val_pred) * 100
        cv_scores.append(mape)
        print(f"Fold {fold + 1}: MAPE = {mape:.2f}%")
    
    print(f"\nMean CV MAPE: {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")
    
    # Train final model
    print("\n=== TRAINING FINAL MODEL ===")
    final_model = build_model(X, y_log)
    
    # Make predictions
    X_test = test_df[features].fillna(0)
    test_pred = np.expm1(final_model.predict(X_test))
    test_pred = np.maximum(test_pred, 1)
    
    # Save predictions
    np.savetxt('prediction_task1g.txt', test_pred, fmt='%.6f')
    
    print("\n=== PREDICTION SUMMARY ===")
    print(f"Predictions saved: prediction_task1g.txt")
    print(f"Min prediction: {test_pred.min():.1f} seconds")
    print(f"Max prediction: {test_pred.max():.1f} seconds")
    print(f"Mean prediction: {test_pred.mean():.1f} seconds")
    print(f"Median prediction: {np.median(test_pred):.1f} seconds")
    
    # Feature importance
    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    importance = final_model.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_imp.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.1f}")

if __name__ == "__main__":
    main()