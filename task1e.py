import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def extract_features(df):
    """Enhanced feature engineering with more sophisticated transformations"""
    df = df.copy()
    
    # Convert basic numeric columns
    numeric_cols = ['Cores', 'Resolution', 'SitesUsed', 'SitesNotUsed', 
                   'SubscibersUsed', 'EvalAreaInSquareKilometers', 'Version']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Handle boolean columns
    bool_cols = ['UseNewCatpMode', 'Outbound', 'Inbound', 'Roundtrip', 
                'WorstDirection', 'VoiceTrafficUsed', 'MakeVoyagerGrid']
    
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(int)
    
    # Enhanced SimnetVersion parsing
    df['SimnetVersion'] = df['SimnetVersion'].astype(str)
    version_parts = df['SimnetVersion'].str.split('.', expand=True)
    
    df['SimnetMajor'] = pd.to_numeric(version_parts[0], errors='coerce').fillna(3)
    df['SimnetMinor'] = pd.to_numeric(version_parts[1], errors='coerce').fillna(1)
    df['SimnetPatch'] = pd.to_numeric(version_parts[2], errors='coerce').fillna(0)
    
    # Parse Results string with more detail
    df['Results'] = df['Results'].fillna('')
    df['ResultCount'] = df['Results'].str.count('\\|') + 1
    df['ResultCount'] = df['ResultCount'].replace(1, 0).where(df['Results'] != '', 0)
    
    # Enhanced result type detection
    important_results = ['reliability', 'contour', 'grid', 'statistics', 'terrain', 
                        'pathloss', 'coverage', 'interference', 'capacity']
    for result in important_results:
        df[f'Has_{result}'] = df['Results'].str.contains(result, case=False, na=False).astype(int)
    
    # Parse AdditionalLayers with more sophistication
    df['AdditionalLayers'] = df['AdditionalLayers'].fillna('')
    df['LayerCount'] = df['AdditionalLayers'].str.count('\\|') + 1
    df['LayerCount'] = df['LayerCount'].replace(1, 0).where(df['AdditionalLayers'] != '', 0)
    
    # Enhanced layer type detection
    layer_types = ['building', 'clutter', 'terrain', 'population', 'road']
    for layer in layer_types:
        df[f'HasLayer_{layer}'] = df['AdditionalLayers'].str.contains(layer, case=False, na=False).astype(int)
    
    # Parse Analysis string with more detail
    df['Analysis'] = df['Analysis'].fillna('')
    df['AnalysisCount'] = df['Analysis'].str.count('\\|') + 1
    df['AnalysisCount'] = df['AnalysisCount'].replace(1, 0).where(df['Analysis'] != '', 0)
    
    # Enhanced analysis types
    analysis_types = ['Reliability', 'Contour', 'TrboReliability', 'Coverage', 'Interference']
    for analysis in analysis_types:
        df[f'Has_{analysis}'] = df['Analysis'].str.contains(analysis, case=False, na=False).astype(int)
    
    # Enhanced derived features
    df['TotalSites'] = df['SitesUsed'] + df['SitesNotUsed']
    df['SiteUsageRatio'] = df['SitesUsed'] / np.maximum(df['TotalSites'], 1)
    df['SiteDensity'] = df['TotalSites'] / np.maximum(df['EvalAreaInSquareKilometers'], 0.1)
    df['AreaLogScale'] = np.log1p(df['EvalAreaInSquareKilometers'])
    df['AreaSqrt'] = np.sqrt(df['EvalAreaInSquareKilometers'])
    
    # Enhanced core utilization features
    df['WorkPerCore'] = (df['TotalSites'] * df['Resolution'] * (1 + df['ResultCount'])) / np.maximum(df['Cores'], 1)
    df['AreaPerCore'] = df['EvalAreaInSquareKilometers'] / np.maximum(df['Cores'], 1)
    df['SubscribersPerCore'] = df['SubscibersUsed'] / np.maximum(df['Cores'], 1)
    df['ResolutionPerCore'] = df['Resolution'] / np.maximum(df['Cores'], 1)
    
    # Direction complexity
    df['DirectionCount'] = df['Outbound'] + df['Inbound'] + df['Roundtrip'] + df['WorstDirection']
    df['HasMultipleDirections'] = (df['DirectionCount'] > 1).astype(int)
    
    # Enhanced complexity scores
    df['ComputationalComplexity'] = (
        df['Resolution'] * 
        df['TotalSites'] * 
        (1 + df['ResultCount']) * 
        (1 + df['LayerCount']) * 
        (1 + df['AnalysisCount']) * 
        np.log1p(df['EvalAreaInSquareKilometers']) *
        (1 + df['DirectionCount'])
    ) / np.maximum(df['Cores'], 1)
    
    df['DataVolumeComplexity'] = (
        df['SubscibersUsed'] * 
        df['Resolution'] * 
        df['EvalAreaInSquareKilometers']
    ) / np.maximum(df['Cores'], 1)
    
    # Version-based features
    df['VersionScore'] = df['SimnetMajor'] * 100 + df['SimnetMinor'] * 10 + df['SimnetPatch']
    df['IsModernVersion'] = (df['SimnetMajor'] >= 3).astype(int)
    df['IsLatestMinor'] = ((df['SimnetMajor'] >= 3) & (df['SimnetMinor'] >= 5)).astype(int)
    
    # Interaction features
    df['SitesTimesDensity'] = df['TotalSites'] * df['SiteDensity']
    df['ResolutionTimesArea'] = df['Resolution'] * df['EvalAreaInSquareKilometers']
    df['CoreEfficiency'] = df['TotalSites'] / (df['Cores'] * df['Resolution'] + 1)
    
    # Categorical variable encoding with frequency encoding for high cardinality
    categorical_cols = ['Mode', 'TaskState']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            
            # Frequency encoding
            freq_map = df[col].value_counts().to_dict()
            df[col + '_freq'] = df[col].map(freq_map)
            
            # Label encoding
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    return df, label_encoders

def select_features():
    """Enhanced feature selection with more comprehensive feature set"""
    feature_columns = [
        # Core computational factors
        'Cores', 'Resolution', 'TotalSites', 'SitesUsed', 'SitesNotUsed', 'SubscibersUsed',
        'EvalAreaInSquareKilometers', 'AreaLogScale', 'AreaSqrt',
        
        # Site-related features
        'SiteUsageRatio', 'SiteDensity', 'SitesTimesDensity',
        
        # Version information
        'SimnetMajor', 'SimnetMinor', 'SimnetPatch', 'VersionScore', 
        'IsModernVersion', 'IsLatestMinor',
        
        # Task complexity
        'ResultCount', 'LayerCount', 'AnalysisCount', 'DirectionCount', 'HasMultipleDirections',
        
        # Enhanced complexity indicators
        'Has_reliability', 'Has_contour', 'Has_grid', 'Has_statistics', 'Has_terrain',
        'Has_pathloss', 'Has_coverage', 'Has_interference', 'Has_capacity',
        'Has_Reliability', 'Has_Contour', 'Has_TrboReliability', 'Has_Coverage', 'Has_Interference',
        
        # Layer features
        'HasLayer_building', 'HasLayer_clutter', 'HasLayer_terrain', 'HasLayer_population', 'HasLayer_road',
        
        # Boolean flags
        'UseNewCatpMode', 'VoiceTrafficUsed', 'MakeVoyagerGrid',
        'Outbound', 'Inbound', 'Roundtrip', 'WorstDirection',
        
        # Enhanced derived complexity features
        'WorkPerCore', 'AreaPerCore', 'SubscribersPerCore', 'ResolutionPerCore',
        'ComputationalComplexity', 'DataVolumeComplexity', 'CoreEfficiency',
        
        # Interaction features
        'ResolutionTimesArea',
        
        # Categorical features
        'Mode_encoded', 'Mode_freq', 'TaskState_encoded', 'TaskState_freq'
    ]
    
    return feature_columns

def create_stratified_folds(y, n_splits=10):
    """Create stratified folds based on target distribution"""
    # Create bins for stratification
    y_binned = pd.qcut(y, q=20, duplicates='drop', labels=False)
    
    # Use StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(y)), y_binned):
        folds.append((train_idx, val_idx))
    
    return folds

def train_model(X_train, y_train, X_val=None, y_val=None, enhanced_params=True):
    """Enhanced LightGBM training with better hyperparameters"""
    
    if enhanced_params:
        # More sophisticated hyperparameters for better accuracy
        params = {
            'objective': 'regression',
            'metric': 'mape',
            'boosting_type': 'gbdt',
            'num_leaves': 80,  # Increased for more complexity
            'learning_rate': 0.03,  # Lower for better convergence
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 3,
            'min_child_samples': 15,
            'min_child_weight': 0.001,
            'lambda_l1': 0.05,
            'lambda_l2': 0.05,
            'max_depth': 8,  # Controlled depth
            'min_gain_to_split': 0.01,
            'verbosity': -1,
            'random_state': 42,
            'extra_trees': True,  # Add randomness
            'path_smooth': 1.0
        }
        num_rounds = 2000  # More rounds for better training
        early_stopping_rounds = 150
    else:
        # Fallback parameters
        params = {
            'objective': 'regression',
            'metric': 'mape',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbosity': -1,
            'random_state': 42
        }
        num_rounds = 1000
        early_stopping_rounds = 100
    
    # Create datasets
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_set]
    
    if X_val is not None and y_val is not None:
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        valid_sets.append(val_set)
    
    # Train model
    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_rounds,
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
    )
    
    return model

def main():
    print("=== ENHANCED HYDRA STRATUS TIME PREDICTION MODEL ===")
    print("Loading and preprocessing data...")
    
    # Load data
    train_df = pd.read_csv('traing_data.csv')
    test_df = pd.read_csv('test_data.csv')
    
    print(f"Training data: {train_df.shape[0]:,} samples")
    print(f"Test data: {test_df.shape[0]:,} samples")
    
    # Extract features from training data
    train_processed, label_encoders = extract_features(train_df)
    
    # Apply same transformations to test data
    test_processed = test_df.copy()
    for col in ['Mode', 'TaskState']:
        if col in test_processed.columns and col in label_encoders:
            test_processed[col] = test_processed[col].fillna('Unknown')
            
            # Handle frequency encoding for unseen categories
            if col in train_processed.columns:
                freq_map = train_processed[col].value_counts().to_dict()
                test_processed[col + '_freq'] = test_processed[col].map(freq_map).fillna(1)
            
            # Handle label encoding for unseen categories
            le = label_encoders[col]
            test_processed[col + '_encoded'] = test_processed[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    test_processed, _ = extract_features(test_processed)
    
    # Filter training data more carefully
    # Keep more data but filter extreme outliers
    version_mask = (train_processed['SimnetMajor'] >= 3)  # More inclusive
    train_filtered = train_processed[version_mask].copy()
    print(f"Filtered training data: {train_filtered.shape[0]:,} samples (version >= 3.0)")
    
    # More conservative outlier removal - keep 99% of data
    time_threshold = train_filtered['TimeInSeconds'].quantile(0.99)
    train_clean = train_filtered[train_filtered['TimeInSeconds'] <= time_threshold].copy()
    print(f"After outlier removal: {train_clean.shape[0]:,} samples")
    
    # Select features
    feature_cols = select_features()
    available_features = [col for col in feature_cols if col in train_clean.columns and col in test_processed.columns]
    print(f"Using {len(available_features)} features")
    
    X = train_clean[available_features].fillna(0)
    y = train_clean['TimeInSeconds']
    
    # Enhanced target transformation with Box-Cox-like approach
    y_log = np.log1p(y)
    
    # Enhanced cross-validation with 10 folds and stratification
    print("\nPerforming enhanced 10-fold cross-validation...")
    folds = create_stratified_folds(y, n_splits=10)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"Fold {fold + 1}/10...")
        
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y_log.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]  # Original scale for MAPE calculation
        
        # Train model with enhanced parameters
        model = train_model(X_train_fold, y_train_fold, X_val_fold, np.log1p(y_val_fold), enhanced_params=True)
        
        # Predict and transform back to original scale
        val_pred_log = model.predict(X_val_fold)
        val_pred = np.expm1(val_pred_log)
        val_pred = np.maximum(val_pred, 1)  # Ensure positive predictions
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_val_fold, val_pred) * 100
        cv_scores.append(mape)
        print(f"Fold {fold + 1} MAPE: {mape:.2f}%")
    
    print(f"\nEnhanced Cross-validation MAPE: {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")
    
    # Train final model on all data with enhanced parameters
    print("\nTraining final enhanced model...")
    final_model = train_model(X, y_log, enhanced_params=True)
    
    # Make predictions on test data
    print("Making predictions...")
    X_test = test_processed[available_features].fillna(0)
    test_pred_log = final_model.predict(X_test)
    test_pred = np.expm1(test_pred_log)
    test_pred = np.maximum(test_pred, 1)  # Ensure positive predictions
    
    # Validation checks
    assert len(test_pred) == len(test_df), "Prediction count mismatch!"
    assert np.all(test_pred > 0), "Negative predictions detected!"
    assert not np.any(np.isnan(test_pred)), "NaN predictions detected!"
    
    # Save predictions
    np.savetxt('prediction1e.txt', test_pred, fmt='%.6f')
    
    print(f"\nPrediction Statistics:")
    print(f"Min: {test_pred.min():.2f} seconds")
    print(f"Max: {test_pred.max():.2f} seconds")
    print(f"Mean: {test_pred.mean():.2f} seconds")
    print(f"Median: {np.median(test_pred):.2f} seconds")
    
    print(f"\nPredictions saved to prediction1d_enhanced.txt")
    print("Enhanced model training completed successfully!")

if __name__ == "__main__":
    main()