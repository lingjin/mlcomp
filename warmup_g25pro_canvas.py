import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# --- Configuration ---
NUM_POLICE_CARS = 10
INPUT_CSV_FILE = 'incidents_2025.csv'  # Make sure this file is in the same directory or provide the full path
OUTPUT_CSV_FILE = 'police_car_locations_g25.csv'
# Using a fixed random_state for KMeans ensures reproducibility of results
KMEANS_RANDOM_STATE = 42
# n_init is the number of times the k-means algorithm will be run with different centroid seeds.
# The final results will be the best output of n_init consecutive runs in terms of inertia.
KMEANS_N_INIT = 10 # Explicitly set for clarity, 'auto' is an option in newer sklearn

def load_incident_data(file_path: str) -> pd.DataFrame:
    """
    Loads incident data from a CSV file.
    Expects columns: 'latitude', 'longitude', 'priority'.
    """
    try:
        df = pd.read_csv(file_path)
        # Basic validation of expected columns
        if not {'latitude', 'longitude', 'priority'}.issubset(df.columns):
            raise ValueError(f"Input CSV must contain 'latitude', 'longitude', and 'priority' columns. Found: {df.columns.tolist()}")
        # Ensure numeric types for relevant columns
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['priority'] = pd.to_numeric(df['priority'], errors='coerce')
        
        # Drop rows with NaN in critical columns that might have resulted from coercion errors
        df.dropna(subset=['latitude', 'longitude', 'priority'], inplace=True)
        
        return df
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
        raise
    except ValueError as ve:
        print(f"Error processing CSV file: {ve}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        raise

def find_optimal_locations(df_incidents: pd.DataFrame, num_cars: int) -> np.ndarray:
    """
    Finds optimal locations for police cars using weighted K-Means clustering.
    """
    if df_incidents.empty:
        print("Warning: Incident data is empty. Cannot determine optimal locations.")
        return np.array([])
        
    if len(df_incidents) < num_cars:
        print(f"Warning: Number of incidents ({len(df_incidents)}) is less than the number of police cars ({num_cars}). "
              "KMeans will still run, but some cars might be clustered on very few or shared points.")

    # Features for clustering are latitude and longitude
    features = df_incidents[['latitude', 'longitude']]
    
    # Weights for clustering are based on incident priority
    # Higher priority means the incident has more influence on centroid placement
    weights = df_incidents['priority']

    # Initialize and fit the K-Means model
    # The `sample_weight` parameter allows us to give more importance to high-priority incidents
    kmeans = KMeans(
        n_clusters=num_cars,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
        init='k-means++' # Standard initialization method
    )
    
    try:
        kmeans.fit(features, sample_weight=weights)
    except Exception as e:
        print(f"Error during KMeans fitting: {e}")
        # If KMeans fails with weights (e.g., all weights are zero, though unlikely with priority),
        # you might consider a fallback or re-raising.
        # For now, we'll let it raise.
        raise

    # The cluster centers are the optimal locations for the police cars
    # These are returned as a NumPy array where each row is [latitude, longitude]
    optimal_locations = kmeans.cluster_centers_
    return optimal_locations

def save_locations_to_csv(locations: np.ndarray, output_file: str):
    """
    Saves the police car locations to a CSV file.
    Each row will contain 'latitude,longitude'. No header.
    """
    if locations.size == 0:
        print("No locations to save.")
        return

    # Create a DataFrame to easily save to CSV in the desired format
    # The columns in `locations` from `kmeans.cluster_centers_` will be in the
    # same order as the input features ('latitude', 'longitude').
    df_locations = pd.DataFrame(locations, columns=['latitude', 'longitude'])
    
    try:
        # Save to CSV without header and without index
        df_locations.to_csv(output_file, header=False, index=False)
        print(f"Optimal police car locations saved to '{output_file}'")
    except Exception as e:
        print(f"An error occurred while saving the locations to CSV: {e}")
        raise

# --- Main execution ---
if __name__ == '__main__':
    print("Starting Optimal Police Car Placement calculation...")
    try:
        # 1. Load incident data
        incidents_data = load_incident_data(INPUT_CSV_FILE)
        
        if not incidents_data.empty:
            # 2. Find optimal locations
            print(f"Finding optimal locations for {NUM_POLICE_CARS} police cars...")
            police_car_coordinates = find_optimal_locations(incidents_data, NUM_POLICE_CARS)
            
            # 3. Save the results
            if police_car_coordinates.size > 0:
                save_locations_to_csv(police_car_coordinates, OUTPUT_CSV_FILE)
            else:
                print("Could not determine police car locations.")
        else:
            print(f"No valid incident data loaded from '{INPUT_CSV_FILE}'. Cannot proceed.")
            
    except FileNotFoundError:
        # Handled in load_incident_data, but good to have a top-level catch
        print(f"Process aborted: Input file '{INPUT_CSV_FILE}' was not found.")
    except ValueError as ve:
        # Handled in load_incident_data for specific CSV errors
        print(f"Process aborted due to data error: {ve}")
    except Exception as e:
        # Catch-all for any other unexpected errors
        print(f"An unexpected error occurred during the process: {e}")
        print("Process aborted.")
    
    print("Calculation finished.")
