# Optimal Police Car Placement 2025
# Weighted K-Means Clustering Approach

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def solve_police_placement(input_csv_path='incidents_2025.csv', output_csv_path='police_car_locations_g25_dr.csv', num_cars=10):
    """
    Analyzes incident data and determines optimal locations for police cars
    using Weighted K-Means clustering.

    Args:
        input_csv_path (str): Path to the input CSV file with incident data.
                              Columns: latitude, longitude, priority.
        output_csv_path (str): Path to save the output CSV file with optimal locations.
                               Format: latitude,longitude (no header).
        num_cars (int): The number of police cars to place (i.e., number of clusters).
    """
    print(f"Starting Optimal Police Car Placement for {num_cars} cars.")

    # 1. Loading and Inspecting Incident Data
    print(f"\nStep 1: Loading incident data from '{input_csv_path}'...")
    try:
        incidents_df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv_path}' not found. Please check the file path.")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    print("Data loaded successfully. Basic data inspection:")
    print("DataFrame Head:")
    print(incidents_df.head())
    # Check for required columns
    required_columns = ['latitude', 'longitude', 'priority']
    if not all(col in incidents_df.columns for col in required_columns):
        print(f"Error: Input CSV must contain the columns: {', '.join(required_columns)}")
        return
    
    print("\nDataframe Info:")
    incidents_df.info()


    # 2. Data Preparation for Weighted K-Means
    print("\nStep 2: Preparing data for Weighted K-Means...")
    # Extract features (latitude, longitude)
    # These are the coordinates K-Means will use for clustering
    try:
        X = incidents_df[['latitude', 'longitude']].values
    except KeyError:
        print("Error: 'latitude' or 'longitude' columns not found in the input CSV.")
        return

    # Extract sample weights (priority)
    # These weights influence the importance of each incident in centroid calculation
    try:
        sample_weights = incidents_df['priority'].values
        # Ensure weights are numeric
        if not pd.api.types.is_numeric_dtype(sample_weights):
            print("Error: 'priority' column must be numeric.")
            return
        # Handle potential NaN or negative weights if necessary, though problem implies positive priorities
        if np.any(np.isnan(sample_weights)) or np.any(sample_weights < 0):
            print("Warning: Priorities contain NaN or negative values. This might affect results.")
            # Basic handling: replace NaN with 0 or 1, clamp negatives.
            # For competition, assume data is clean as per problem statement.
            # sample_weights = np.nan_to_num(sample_weights, nan=0.0)
            # sample_weights[sample_weights < 0] = 0.0

    except KeyError:
        print("Error: 'priority' column not found in the input CSV.")
        return

    print(f"Feature matrix X shape: {X.shape}")
    print(f"Sample weights array shape: {sample_weights.shape}")

    if X.shape == 0:
        print("Error: No data points found in the input file after processing.")
        return
    if X.shape[0] < num_cars:
        print(f"Warning: Number of incidents ({X.shape}) is less than the number of police cars ({num_cars}).")
        print("The number of clusters will be limited to the number of incidents.")
        # KMeans will handle this by setting n_clusters to n_samples if n_samples < n_clusters
        # However, it's good to be aware.

    # 3. Applying Scikit-learn's KMeans with sample_weight
    print("\nStep 3: Applying Weighted K-Means clustering...")
    
    # Configure the KMeans model
    # n_clusters: The number of police cars to place.
    # init='k-means++': A smart initialization technique for centroids.
    # n_init: Number of times the k-means algorithm will be run with different centroid seeds.
    #         The final results will be the best output of n_init consecutive runs in terms of inertia.
    #         'auto' is available in newer sklearn versions, but 10 is a common explicit choice.
    # random_state: For reproducibility of results.
    kmeans_model = KMeans(
        n_clusters=num_cars,
        init='k-means++',
        n_init=10,  # Run 10 times with different initial centroids
        random_state=42, # Ensures results are the same each time the script is run
        max_iter=300     # Maximum number of iterations for a single run
    )

    # Fit the KMeans model to the incident data, using priorities as sample weights
    # This means incidents with higher priority will have a stronger influence on centroid placement
    try:
        kmeans_model.fit(X, sample_weight=sample_weights)
    except ValueError as ve:
        print(f"Error during K-Means fitting: {ve}")
        print("This can happen if all sample weights are zero or if there are issues with input data.")
        return
        
    print("K-Means model fitting complete.")

    # 4. Retrieving Optimal Locations: The Cluster Centroids
    print("\nStep 4: Retrieving optimal police car locations (cluster centroids)...")
    # The cluster_centers_ attribute holds the coordinates of the centroids
    optimal_locations = kmeans_model.cluster_centers_
    print("Optimal Locations (Latitude, Longitude):")
    for i, loc in enumerate(optimal_locations):
        print(f"Car {i+1}: {loc[0]:.6f}, {loc[1]:.6f}")


    # 5. Formatting and Saving the Output CSV
    print(f"\nStep 5: Formatting and saving output to '{output_csv_path}'...")
    # Create a DataFrame for the optimal locations for easy CSV writing
    output_df = pd.DataFrame(optimal_locations, columns=['latitude', 'longitude'])

    # Save the DataFrame to a CSV file
    # index=False: Do not write row numbers (index) to the CSV
    # header=False: Do not write column names (header) to the CSV, as per example output
    # float_format='%.6f': Ensure coordinates are written with 6 decimal places
    try:
        output_df.to_csv(output_csv_path, index=False, header=False, float_format='%.6f')
        print(f"Successfully saved optimal locations to '{output_csv_path}'.")
    except Exception as e:
        print(f"Error saving output CSV file: {e}")

    print("\nOptimal Police Car Placement process finished.")

# --- Main execution block ---
if __name__ == '__main__':
    # These paths can be changed if your files are named differently or are in other locations
    INPUT_CSV = 'incidents_2025.csv'
    OUTPUT_CSV = 'police_car_locations_g25_dr.csv'
    NUMBER_OF_POLICE_CARS = 10 # As specified in the problem

    solve_police_placement(
        input_csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
        num_cars=NUMBER_OF_POLICE_CARS
    )