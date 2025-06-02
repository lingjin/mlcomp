import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgbm
import xgboost as xgb
from sklearn.ensemble import StackingRegressor, VotingRegressor
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
   # Create a copy to avoid modifying the original
   df = df.copy()
   
   # Handle missing values in Cores - replace with median
   df['Cores'] = pd.to_numeric(df['Cores'], errors='coerce')
   cores_median = df['Cores'].median()
   df['Cores'].fillna(cores_median, inplace=True)
   
   # Handle SimnetVersion - extract version components
   df['SimnetVersion'] = df['SimnetVersion'].astype(str)
   
   # Extract up to 4 version components
   version_parts = df['SimnetVersion'].str.split('.', expand=True)
   max_parts = len(version_parts.columns)
   
   # Create version component columns with appropriate handling for missing parts
   df['SimnetMajor'] = pd.to_numeric(version_parts[0], errors='coerce').fillna(0)
   df['SimnetMinor'] = pd.to_numeric(version_parts[1], errors='coerce').fillna(0)
   
   # Handle patch version (might be at index 2 or 3 depending on version format)
   if max_parts > 2:
       df['SimnetPatch'] = pd.to_numeric(version_parts[2], errors='coerce').fillna(0)
   else:
       df['SimnetPatch'] = 0
       
   # Handle build version if available (4th component)
   if max_parts > 3:
       df['SimnetBuild'] = pd.to_numeric(version_parts[3], errors='coerce').fillna(0)
   else:
       df['SimnetBuild'] = 0
   
   # Fill NaN values in Results
   df['Results'] = df['Results'].fillna('')
   
   # Count number of results and create binary features for common result types
   df['ResultsCount'] = df['Results'].str.count('\|') + 1
   
   # Check for specific result types
   result_types = ['reliability_cn', 'gis_statistics', 'round_trip_reliability', 
                   'voyager_grid', 'site_area_reliability', 'cell_area_reliability',
                   'terrain', 'best_server', 'critical_buildings', 'catpsim']
   
   for result_type in result_types:
       df[f'Has_{result_type}'] = df['Results'].str.contains(result_type, regex=False, na=False).astype(int)
   
   # Handle AdditionalLayers
   df['AdditionalLayers'] = df['AdditionalLayers'].fillna('')
   df['AdditionalLayersCount'] = df['AdditionalLayers'].apply(
       lambda x: len(x.split('|')) if x else 0
   )
   
   # Check for specific additional layer types
   layer_types = ['terrain', 'site_area_reliability', 'cell_area_reliability', 
                  'composite', 'aggregate_ssi', 'static_composite_cinr']
   
   for layer_type in layer_types:
       df[f'Layer_{layer_type}'] = df['AdditionalLayers'].str.contains(
           layer_type, regex=False, na=False).astype(int)
   
   # Fill NaN values in Analysis
   df['Analysis'] = df['Analysis'].fillna('')
   
   # Count number of analyses and create binary features for common analysis types
   df['AnalysisCount'] = df['Analysis'].str.count('\|') + 1
   
   # Check for specific analysis types
   analysis_types = ['TdmaReliabilityDom', 'LteReliabilityDom', 'R6602ContoursDom', 
                     'BestServerDom', 'CriticalBuildingAnalysisDom', 'MotoTrboReliabilityDom']
   
   for analysis_type in analysis_types:
       df[f'Analysis_{analysis_type}'] = df['Analysis'].str.contains(
           analysis_type, regex=False, na=False).astype(int)
   
   # Handle missing values in SitesUsed and SitesNotUsed
   df['SitesUsed'] = df['SitesUsed'].fillna(0)
   df['SitesNotUsed'] = df['SitesNotUsed'].fillna(0)
   
   # Calculate sites total
   df['SitesTotal'] = df['SitesUsed'] + df['SitesNotUsed']
   
   # Handle missing values in direction features
   for col in ['Outbound', 'Inbound', 'Roundtrip', 'WorstDirection']:
       df[col] = df[col].fillna(False)
   
   # Calculate direction features
   df['DirectionCount'] = df['Outbound'].astype(int) + df['Inbound'].astype(int) + \
                          df['Roundtrip'].astype(int) + df['WorstDirection'].astype(int)
   
   # Handle missing values in EvalAreaInSquareKilometers
   df['EvalAreaInSquareKilometers'] = df['EvalAreaInSquareKilometers'].fillna(0)
   
   # Log transform area (since it has a wide range)
   df['LogArea'] = np.log1p(df['EvalAreaInSquareKilometers'])
   
   # Handle missing values in SubscibersUsed
   df['SubscibersUsed'] = df['SubscibersUsed'].fillna(1)
   
   # Handle missing values in UseNewCatpMode, VoiceTrafficUsed, and MakeVoyagerGrid
   df['UseNewCatpMode'] = df['UseNewCatpMode'].fillna(False).astype(int)
   df['VoiceTrafficUsed'] = df['VoiceTrafficUsed'].fillna(False).astype(int)
   df['MakeVoyagerGrid'] = df['MakeVoyagerGrid'].fillna(False).astype(int)
   
   # Create complexity scores
   df['ComplexityScore'] = (df['Resolution'] * df['SitesTotal'] * df['ResultsCount'] * 
                            np.log1p(df['EvalAreaInSquareKilometers'])) / np.maximum(df['Cores'], 1)
   
   df['SiteDensity'] = df['SitesTotal'] / np.maximum(df['EvalAreaInSquareKilometers'], 1)
   
   # Create interaction features
   df['AreaPerCore'] = df['EvalAreaInSquareKilometers'] / np.maximum(df['Cores'], 1)
   df['SitesPerCore'] = df['SitesTotal'] / np.maximum(df['Cores'], 1)
   df['ResolutionAreaProduct'] = df['Resolution'] * df['LogArea']
   
   # Resolution complexity features
   df['ResolutionSitesProduct'] = df['Resolution'] * df['SitesTotal']
   df['ResolutionResultsProduct'] = df['Resolution'] * df['ResultsCount']
   
   # Version-specific interactions
   df['VersionComplexity'] = df['SimnetMajor'] * 100 + df['SimnetMinor'] * 10 + df['SimnetPatch']
   
   # Task complexity score
   df['TaskComplexityScore'] = (df['Resolution'] * df['SitesTotal'] * 
                               df['ResultsCount'] * np.maximum(df['SubscibersUsed'], 1) * 
                               np.maximum(df['DirectionCount'], 1) * 
                               np.log1p(df['EvalAreaInSquareKilometers'])) / np.maximum(df['Cores'], 1)
   
   # Additional interactions
   df['CoreResolutionRatio'] = df['Resolution'] / np.maximum(df['Cores'], 1)
   df['SiteSubscriberRatio'] = df['SitesTotal'] / np.maximum(df['SubscibersUsed'], 1)
   
   # Time-related features based on Version
   df['VersionAge'] = 4 - df['SimnetMajor'] + ((10 - df['SimnetMinor']) / 10)
   
   return df

# Custom function to optimize MAPE directly for LightGBM
def custom_mape_eval(y_pred, dtrain):
   y_true = dtrain.get_label()
   # Prevent division by zero
   mask = y_true != 0
   return 'mape', np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100, False

def get_feature_lists():
   # Base features
   base_features = [
       'Cores', 'Resolution', 'SimnetMajor', 'SimnetMinor', 'SimnetPatch', 'SimnetBuild',
       'ResultsCount', 'SitesUsed', 'SitesNotUsed', 'SitesTotal', 'SubscibersUsed',
       'AdditionalLayersCount', 'UseNewCatpMode', 'DirectionCount', 'VoiceTrafficUsed',
       'MakeVoyagerGrid', 'EvalAreaInSquareKilometers', 'LogArea', 'Version',
       'AnalysisCount', 'ComplexityScore', 'SiteDensity', 'AreaPerCore', 'SitesPerCore',
       'ResolutionAreaProduct', 'ResolutionSitesProduct', 'ResolutionResultsProduct',
       'VersionComplexity', 'TaskComplexityScore', 'CoreResolutionRatio', 
       'SiteSubscriberRatio', 'VersionAge'
   ]

   # Result type features
   result_features = [f'Has_{result}' for result in [
       'reliability_cn', 'gis_statistics', 'round_trip_reliability', 
       'voyager_grid', 'site_area_reliability', 'cell_area_reliability',
       'terrain', 'best_server', 'critical_buildings', 'catpsim'
   ]]

   # Layer type features
   layer_features = [f'Layer_{layer}' for layer in [
       'terrain', 'site_area_reliability', 'cell_area_reliability', 
       'composite', 'aggregate_ssi', 'static_composite_cinr'
   ]]

   # Analysis type features
   analysis_features = [f'Analysis_{analysis}' for analysis in [
       'TdmaReliabilityDom', 'LteReliabilityDom', 'R6602ContoursDom', 
       'BestServerDom', 'CriticalBuildingAnalysisDom', 'MotoTrboReliabilityDom'
   ]]

   # Combine all numerical features
   numerical_features = base_features + result_features + layer_features + analysis_features

   # Categorical features
   categorical_features = ['Mode', 'TaskState']
   
   return numerical_features, categorical_features

def build_model():
   # Define LightGBM model with optimized hyperparameters
   lgb_params = {
       'objective': 'regression',
       'n_estimators': 2000,
       'learning_rate': 0.03,
       'num_leaves': 40,
       'max_depth': 8,
       'min_child_samples': 20,
       'subsample': 0.8,
       'colsample_bytree': 0.8,
       'reg_alpha': 0.1,
       'reg_lambda': 0.1,
       'random_state': 42
   }

   lgb_model = lgbm.LGBMRegressor(**lgb_params)

   # Define XGBoost model with optimized hyperparameters
   xgb_params = {
       'objective': 'reg:squarederror',
       'n_estimators': 1500,
       'learning_rate': 0.03,
       'max_depth': 7,
       'min_child_weight': 3,
       'subsample': 0.8,
       'colsample_bytree': 0.8,
       'gamma': 0.1,
       'reg_alpha': 0.1,
       'reg_lambda': 1.0,
       'random_state': 42
   }

   xgb_model = xgb.XGBRegressor(**xgb_params)

   # Create a voting ensemble with optimized weights
   voting_regressor = VotingRegressor(
       estimators=[
           ('lgb', lgb_model),
           ('xgb', xgb_model)
       ],
       weights=[0.6, 0.4]  # Give more weight to LightGBM as it often performs better for this type of data
   )
   
   return voting_regressor

def perform_cross_validation(model, X, y_log, y):
   # Set up cross-validation
   kf = KFold(n_splits=5, shuffle=True, random_state=42)
   
   # Get feature lists
   numerical_features, categorical_features = get_feature_lists()
   
   # Prepare preprocessing pipeline
   preprocessor = ColumnTransformer(
       transformers=[
           ('num', StandardScaler(), numerical_features),
           ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
       ])
   
   # Define pipeline
   pipeline = Pipeline(steps=[
       ('preprocessor', preprocessor),
       ('regressor', model)
   ])
   
   # Cross-validation to estimate MAPE
   print("Performing cross-validation...")
   cv_scores = []
   for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
       print(f"Training fold {fold+1}/5...")
       X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
       y_train_log, y_val = y_log.iloc[train_idx], y.iloc[val_idx]
       
       # Train the model
       pipeline.fit(X_train, y_train_log)
       
       # Make predictions and convert back from log scale
       val_preds_log = pipeline.predict(X_val)
       val_preds = np.expm1(val_preds_log)
       
       # Calculate MAPE
       mape = mean_absolute_percentage_error(y_val, val_preds) * 100
       cv_scores.append(mape)
       
       print(f"Fold {fold+1} MAPE: {mape:.2f}%")

   print(f"Average CV MAPE: {np.mean(cv_scores):.2f}%")
   
   return pipeline

def main():
   # Load data
   print("Loading data...")
   train_data = pd.read_csv('traing_data.csv')
   test_data = pd.read_csv('test_data.csv')
   print(f"Training data shape: {train_data.shape}")
   print(f"Test data shape: {test_data.shape}")

   # Apply preprocessing
   print("Preprocessing data...")
   train_data = preprocess_data(train_data)
   test_data = preprocess_data(test_data)

   # Filter to only include data from version 3.1+ to match test data
   train_data_filtered = train_data[train_data['SimnetMajor'] >= 3].copy()
   print(f"Filtered training data shape: {train_data_filtered.shape}")

   # Get feature lists
   numerical_features, categorical_features = get_feature_lists()

   # Prepare data for modeling
   X = train_data_filtered[numerical_features + categorical_features]
   y = train_data_filtered['TimeInSeconds']

   # Log transform the target for better model performance
   y_log = np.log1p(y)

   # Build model
   model = build_model()
   
   # Perform cross-validation
   pipeline = perform_cross_validation(model, X, y_log, y)
   
   # Train the final model on all data
   print("Training final model on full dataset...")
   pipeline.fit(X, y_log)
   
   # Prepare test data for prediction
   X_test = test_data[numerical_features + categorical_features]

   # Make predictions on test data
   print("Making predictions on test data...")
   test_preds_log = pipeline.predict(X_test)
   test_preds = np.expm1(test_preds_log)

   # Ensure predictions are positive
   test_preds = np.maximum(test_preds, 0)

   # Final checks
   print("Performing final checks...")
   assert len(test_preds) == len(test_data), "Prediction count doesn't match test set size"
   assert np.all(test_preds > 0), "Negative predictions detected"

   # Save predictions to file
   np.savetxt('prediction.txt', test_preds, fmt='%.6f')

   print("Predictions saved to prediction.txt")
   print("Done!")

if __name__ == "__main__":
   main()
