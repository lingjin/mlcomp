import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def extract_features(df):
    """Extract and engineer features from the raw data"""
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
    
    # Parse SimnetVersion to extract major, minor, patch versions
    df['SimnetVersion'] = df['SimnetVersion'].astype(str)
    version_parts = df['SimnetVersion'].str.split('.', expand=True)
    
    df['SimnetMajor'] = pd.to_numeric(version_parts[0], errors='coerce').fillna(3)
    df['SimnetMinor'] = pd.to_numeric(version_parts[1], errors='coerce').fillna(1)
    df['SimnetPatch'] = pd.to_numeric(version_parts[2], errors='coerce').fillna(0)
    
    # Parse Results string to count complexity
    df['Results'] = df['Results'].fillna('')
    df['ResultCount'] = df['Results'].str.count('\\|') + 1
    df['ResultCount'] = df['ResultCount'].replace(1, 0).where(df['Results'] != '', 0)
    
    # Key result types that impact computation time
    important_results = ['reliability', 'contour', 'grid', 'statistics', 'terrain']
    for result in important_results:
        df[f'Has_{result}'] = df['Results'].str.contains(result, case=False, na=False).astype(int)
    
    # Parse AdditionalLayers
    df['AdditionalLayers'] = df['AdditionalLayers'].fillna('')
    df['LayerCount'] = df['AdditionalLayers'].str.count('\\|') + 1
    df['LayerCount'] = df['LayerCount'].replace(1, 0).where(df['AdditionalLayers'] != '', 0)
    
    # Parse Analysis string
    df['Analysis'] = df['Analysis'].fillna('')
    df['AnalysisCount'] = df['Analysis'].str.count('\\|') + 1
    df['AnalysisCount'] = df['AnalysisCount'].replace(1, 0).where(df['Analysis'] != '', 0)
    
    # Key analysis types
    analysis_types = ['Reliability', 'Contour', 'TrboReliability']
    for analysis in analysis_types:
        df[f'Has_{analysis}'] = df['Analysis'].str.contains(analysis, case=False, na=False).astype(int)
    
    # Derived features that capture computational complexity
    df['TotalSites'] = df['SitesUsed'] + df['SitesNotUsed']
    df['SiteDensity'] = df['TotalSites'] / np.maximum(df['EvalAreaInSquareKilometers'], 0.1)
    df['AreaLogScale'] = np.log1p(df['EvalAreaInSquareKilometers'])
    
    # Core utilization features
    df['WorkPerCore'] = (df['TotalSites'] * df['Resolution'] * df['ResultCount']) / np.maximum(df['Cores'], 1)
    df['AreaPerCore'] = df['EvalAreaInSquareKilometers'] / np.maximum(df['Cores'], 1)
    
    # Direction complexity
    df['DirectionCount'] = df['Outbound'] + df['Inbound'] + df['Roundtrip'] + df['WorstDirection']
    
    # Overall complexity score combining key factors
    df['ComplexityScore'] = (
        df['Resolution'] * 
        df['TotalSites'] * 
        (1 + df['ResultCount']) * 
        (1 + df['LayerCount']) * 
        (1 + df['AnalysisCount']) * 
        np.log1p(df['EvalAreaInSquareKilometers'])
    ) / np.maximum(df['Cores'], 1)
    
    # Version-based complexity (newer versions might be more efficient)
    df['VersionScore'] = df['SimnetMajor'] * 100 + df['SimnetMinor'] * 10 + df['SimnetPatch']
    df['IsModernVersion'] = (df['SimnetMajor'] >= 3).astype(int)
    
    # Encode categorical variables
    categorical_cols = ['Mode', 'TaskState']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    return df, label_encoders

def select_features():
    """Select the most relevant features for the model"""
    feature_columns = [
        # Core computational factors
        'Cores', 'Resolution', 'TotalSites', 'SitesUsed', 'SubscibersUsed',
        'EvalAreaInSquareKilometers', 'AreaLogScale',
        
        # Version information
        'SimnetMajor', 'SimnetMinor', 'SimnetPatch', 'VersionScore', 'IsModernVersion',
        
        # Task complexity
        'ResultCount', 'LayerCount', 'AnalysisCount', 'DirectionCount',
        
        # Complexity indicators
        'Has_reliability', 'Has_contour', 'Has_grid', 'Has_statistics', 'Has_terrain',
        'Has_Reliability', 'Has_Contour', 'Has_TrboReliability',
        
        # Boolean flags
        'UseNewCatpMode', 'VoiceTrafficUsed', 'MakeVoyagerGrid',
        'Outbound', 'Inbound', 'Roundtrip', 'WorstDirection',
        
        # Derived complexity features
        'SiteDensity', 'WorkPerCore', 'AreaPerCore', 'ComplexityScore',
        
        # Categorical encoded
        'Mode_encoded', 'TaskState_encoded'
    ]
    
    return feature_columns

def train_model(X_train, y_train, X_val=None, y_val=None):
    """Train LightGBM model optimized for this specific problem"""
    
    # LightGBM parameters optimized for time prediction
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
        num_boost_round=1000,
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    return model

def main():
    print("=== HYDRA STRATUS TIME PREDICTION MODEL ===")
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
            # Handle unseen categories
            le = label_encoders[col]
            test_processed[col + '_encoded'] = test_processed[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    test_processed, _ = extract_features(test_processed)
    
    # Filter training data to match test data (version >= 3.1)
    version_mask = (train_processed['SimnetMajor'] >= 3) & (train_processed['SimnetMinor'] >= 1)
    train_filtered = train_processed[version_mask].copy()
    print(f"Filtered training data: {train_filtered.shape[0]:,} samples (version >= 3.1)")
    
    # Remove extreme outliers (top 0.5% of computation times)
    time_threshold = train_filtered['TimeInSeconds'].quantile(0.995)
    train_clean = train_filtered[train_filtered['TimeInSeconds'] <= time_threshold].copy()
    print(f"After outlier removal: {train_clean.shape[0]:,} samples")
    
    # Select features
    feature_cols = select_features()
    
    # Ensure all features exist in both datasets
    available_features = [col for col in feature_cols if col in train_clean.columns and col in test_processed.columns]
    print(f"Using {len(available_features)} features")
    
    X = train_clean[available_features].fillna(0)
    y = train_clean['TimeInSeconds']
    
    # Log transform target to handle wide range and skewness
    y_log = np.log1p(y)
    
    # Cross-validation to estimate performance
    print("\nPerforming cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold + 1}/5...")
        
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y_log.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]  # Original scale for MAPE calculation
        
        # Train model
        model = train_model(X_train_fold, y_train_fold, X_val_fold, np.log1p(y_val_fold))
        
        # Predict and transform back to original scale
        val_pred_log = model.predict(X_val_fold)
        val_pred = np.expm1(val_pred_log)
        val_pred = np.maximum(val_pred, 1)  # Ensure positive predictions
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_val_fold, val_pred) * 100
        cv_scores.append(mape)
        print(f"Fold {fold + 1} MAPE: {mape:.2f}%")
    
    print(f"\nCross-validation MAPE: {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")
    
    # Train final model on all data
    print("\nTraining final model...")
    final_model = train_model(X, y_log)
    
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
    np.savetxt('prediction1d.txt', test_pred, fmt='%.6f')
    
    print(f"\nPrediction Statistics:")
    print(f"Min: {test_pred.min():.2f} seconds")
    print(f"Max: {test_pred.max():.2f} seconds")
    print(f"Mean: {test_pred.mean():.2f} seconds")
    print(f"Median: {np.median(test_pred):.2f} seconds")
    
    print(f"\nPredictions saved to prediction_hydra.txt")
    print("Model training completed successfully!")

if __name__ == "__main__":
    main()