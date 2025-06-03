import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgbm
import xgboost as xgb
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
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
   
   # Ensure numeric types for all columns used in calculations
   df['Resolution'] = pd.to_numeric(df['Resolution'], errors='coerce').fillna(0)
   df['SitesUsed'] = pd.to_numeric(df['SitesUsed'], errors='coerce').fillna(0)
   df['SitesNotUsed'] = pd.to_numeric(df['SitesNotUsed'], errors='coerce').fillna(0)
   df['SubscibersUsed'] = pd.to_numeric(df['SubscibersUsed'], errors='coerce').fillna(1)
   df['EvalAreaInSquareKilometers'] = pd.to_numeric(df['EvalAreaInSquareKilometers'], errors='coerce').fillna(0)
   df['Version'] = pd.to_numeric(df['Version'], errors='coerce').fillna(0)
   
   # Calculate sites total
   df['SitesTotal'] = df['SitesUsed'] + df['SitesNotUsed']
   
   # Handle missing values in direction features
   for col in ['Outbound', 'Inbound', 'Roundtrip', 'WorstDirection']:
       df[col] = df[col].fillna(False).astype(int)
   
   # Calculate direction features
   df['DirectionCount'] = df['Outbound'] + df['Inbound'] + df['Roundtrip'] + df['WorstDirection']
   
   # Log transform area (since it has a wide range)
   df['LogArea'] = np.log1p(df['EvalAreaInSquareKilometers'])
   
   # Handle missing values in UseNewCatpMode, VoiceTrafficUsed, and MakeVoyagerGrid
   df['UseNewCatpMode'] = df['UseNewCatpMode'].fillna(False).astype(int)
   df['VoiceTrafficUsed'] = df['VoiceTrafficUsed'].fillna(False).astype(int)
   df['MakeVoyagerGrid'] = df['MakeVoyagerGrid'].fillna(False).astype(int)
   
   # Ensure ResultsCount is numeric
   df['ResultsCount'] = pd.to_numeric(df['ResultsCount'], errors='coerce').fillna(1)
   df['AnalysisCount'] = pd.to_numeric(df['AnalysisCount'], errors='coerce').fillna(1)
   
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
   
   # Advanced feature engineering - polynomial features
   df['SitesTotalSquared'] = df['SitesTotal'] ** 2
   df['ResolutionSquared'] = df['Resolution'] ** 2
   df['LogAreaSquared'] = df['LogArea'] ** 2
   
   # Interaction between sites and area
   df['SiteAreaInteraction'] = df['SitesTotal'] * df['LogArea']
   
   # Interaction between resolution and results
   df['ResolutionResultsInteraction'] = df['Resolution'] * df['ResultsCount']
   
   # Complex interactions
   df['ComplexityScore2'] = (df['Resolution'] ** 1.5) * (df['SitesTotal'] ** 0.8) * \
                             (df['ResultsCount'] ** 0.7) * (np.log1p(df['EvalAreaInSquareKilometers']) ** 0.9) / \
                             (np.maximum(df['Cores'], 1) ** 0.95)
   
   # Feature for multi-directional complexity
   df['MultiDirectional'] = (df['DirectionCount'] > 1).astype(int)
   
   # Feature for high resolution tasks
   df['HighResolution'] = (df['Resolution'] > 10).astype(int)
   
   # Feature for large area tasks
   df['LargeArea'] = (df['EvalAreaInSquareKilometers'] > 5000).astype(int)
   
   # Feature for complex tasks (combination of factors)
   df['ComplexTask'] = ((df['Resolution'] > 5) & (df['SitesTotal'] > 10) & 
                        (df['EvalAreaInSquareKilometers'] > 1000)).astype(int)
   
   # Normalize WorkspaceId to a hash value
   df['WorkspaceHash'] = pd.util.hash_pandas_object(df['WorkspaceId']) % 1000
   
   # Mode encoding
   if 'Mode' in df.columns:
       mode_map = {'Draft': 0, 'Normal': 1, 'Fine': 2}
       df['ModeNumeric'] = df['Mode'].map(mode_map).fillna(1)
   
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
       'SiteSubscriberRatio', 'VersionAge', 'SitesTotalSquared', 'ResolutionSquared',
       'LogAreaSquared', 'SiteAreaInteraction', 'ResolutionResultsInteraction',
       'ComplexityScore2', 'MultiDirectional', 'HighResolution', 'LargeArea',
       'ComplexTask', 'WorkspaceHash', 'ModeNumeric'
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
       'n_estimators': 3000,
       'learning_rate': 0.02,
       'num_leaves': 50,
       'max_depth': 10,
       'min_child_samples': 20,
       'subsample': 0.8,
       'colsample_bytree': 0.7,
       'reg_alpha': 0.1,
       'reg_lambda': 1.0,
       'random_state': 42,
       'importance_type': 'gain'
   }

   lgb_model = lgbm.LGBMRegressor(**lgb_params)

   # Define XGBoost model with optimized hyperparameters
   xgb_params = {
       'objective': 'reg:squarederror',
       'n_estimators': 2000,
       'learning_rate': 0.02,
       'max_depth': 9,
       'min_child_weight': 3,
       'subsample': 0.8,
       'colsample_bytree': 0.7,
       'gamma': 0.1,
       'reg_alpha': 0.2,
       'reg_lambda': 1.0,
       'random_state': 42
   }

   xgb_model = xgb.XGBRegressor(**xgb_params)
   
   # Define RandomForest model
   rf_params = {
       'n_estimators': 500,
       'max_depth': 15,
       'min_samples_split': 5,
       'min_samples_leaf': 2,
       'max_features': 'sqrt',
       'random_state': 42
   }
   
   rf_model = RandomForestRegressor(**rf_params)

   # Create a voting ensemble with optimized weights
   voting_regressor = VotingRegressor(
       estimators=[
           ('lgb', lgb_model),
           ('xgb', xgb_model),
           ('rf', rf_model)
       ],
       weights=[0.55, 0.35, 0.10]  # Weighted towards LightGBM
   )
   
   return voting_regressor

def train_and_predict(X_train, y_train_log, X_test, feature_importance=False):
   # Get feature lists
   numerical_features, categorical_features = get_feature_lists()
   
   # Prepare preprocessing pipeline
   preprocessor = ColumnTransformer(
       transformers=[
           ('num', StandardScaler(), numerical_features),
           ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
       ])
   
   # Build model
   model = build_model()
   
   # Define pipeline
   pipeline = Pipeline(steps=[
       ('preprocessor', preprocessor),
       ('regressor', model)
   ])
   
   # Train the model
   print("Training model...")
   pipeline.fit(X_train, y_train_log)
   
   # Make predictions
   print("Making predictions...")
   test_preds_log = pipeline.predict(X_test)
   test_preds = np.expm1(test_preds_log)
   
   # Ensure predictions are positive
   test_preds = np.maximum(test_preds, 0)
   
   # Print feature importances if requested and available
   if feature_importance:
       # Try to get feature importances from LightGBM component
       try:
           lgb_model = pipeline['regressor'].named_estimators_['lgb']
           importances = lgb_model.feature_importances_
           
           # Get feature names (this is approximate as we don't have exact mapping after preprocessing)
           feature_names = numerical_features.copy()
           
           # Add encoded categorical features (approximate)
           for cat in categorical_features:
               if cat == 'Mode':
                   feature_names.extend(['Mode_Draft', 'Mode_Fine', 'Mode_Normal'])
               elif cat == 'TaskState':
                   # Assuming TaskState has values 1-5
                   feature_names.extend([f'TaskState_{i}' for i in range(1, 6)])
           
           # Print top features
           if len(importances) == len(feature_names):
               feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
               feature_imp = feature_imp.sort_values('importance', ascending=False)
               print("\nTop 20 important features (approximated):")
               print(feature_imp.head(20))
           else:
               print("\nCouldn't match feature names with importances.")
       except:
           print("\nCouldn't extract feature importances.")
   
   return test_preds

def perform_cross_validation(X, y_log, y):
   # Set up cross-validation
   kf = KFold(n_splits=5, shuffle=True, random_state=42)
   
   # Cross-validation to estimate MAPE
   print("Performing cross-validation...")
   cv_scores = []
   oof_preds = np.zeros(len(X))
   
   for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
       print(f"Training fold {fold+1}/5...")
       X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
       y_train_log, y_val = y_log.iloc[train_idx], y.iloc[val_idx]
       
       # Train and predict
       val_preds = train_and_predict(X_train, y_train_log, X_val)
       
       # Store out-of-fold predictions
       oof_preds[val_idx] = val_preds
       
       # Calculate MAPE
       mape = mean_absolute_percentage_error(y_val, val_preds) * 100
       cv_scores.append(mape)
       
       print(f"Fold {fold+1} MAPE: {mape:.2f}%")

   # Calculate overall MAPE on out-of-fold predictions
   overall_mape = mean_absolute_percentage_error(y, oof_preds) * 100
   print(f"Overall CV MAPE: {overall_mape:.2f}%")
   print(f"Average CV MAPE: {np.mean(cv_scores):.2f}%")

def main():
   try:
       # Load data
       print("Loading data...")
       train_data = pd.read_csv('traing_data.csv')
       test_data = pd.read_csv('test_data.csv')
       print(f"Training data shape: {train_data.shape}")
       print(f"Test data shape: {test_data.shape}")

       # Apply preprocessing
       print("\nPreprocessing data...")
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

       # Remove outliers (tasks with extremely long or short durations)
       q_low = y.quantile(0.001)
       q_high = y.quantile(0.999)
       filtered_idx = (y >= q_low) & (y <= q_high)
       X = X[filtered_idx]
       y = y[filtered_idx]
       print(f"Data shape after outlier removal: {X.shape}")

       # Log transform the target for better model performance
       y_log = np.log1p(y)

       # Perform cross-validation to evaluate the model
       perform_cross_validation(X, y_log, y)
       
       # Prepare test data for prediction
       X_test = test_data[numerical_features + categorical_features]

       # Train on full dataset and make predictions
       print("\nTraining final model on full dataset...")
       test_preds = train_and_predict(X, y_log, X_test, feature_importance=True)

       # Final checks
       print("Performing final checks...")
       assert len(test_preds) == len(test_data), "Prediction count doesn't match test set size"
       assert np.all(test_preds > 0), "Negative predictions detected"

       # Save predictions to file
       np.savetxt('prediction1b.txt', test_preds, fmt='%.6f')

       print("Predictions saved to prediction.txt")
       print("Done!")
   except Exception as e:
       print(f"Error occurred: {str(e)}")
       import traceback
       traceback.print_exc()

if __name__ == "__main__":
   main()
