import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Load the incident data
def load_data(file_path):
   df = pd.read_csv(file_path)
   return df

# Custom weighted k-means implementation
def weighted_kmeans(X, weights, n_clusters, max_iter=300, tol=1e-4):
   # Initialize centroids randomly
   n_samples, n_features = X.shape
   centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]
   
   for _ in range(max_iter):
       # Calculate distances between each point and centroids
       distances = pairwise_distances(X, centroids)
       
       # Assign each point to the nearest centroid
       labels = np.argmin(distances, axis=1)
       
       # Update centroids based on weighted mean
       new_centroids = np.zeros((n_clusters, n_features))
       for k in range(n_clusters):
           if np.sum(labels == k) > 0:
               # Calculate weighted mean for each cluster
               weight_sum = np.sum(weights[labels == k])
               if weight_sum > 0:
                   new_centroids[k] = np.sum(X[labels == k].T * weights[labels == k], axis=1) / weight_sum
               else:
                   new_centroids[k] = centroids[k]
           else:
               # If a cluster is empty, reinitialize it
               new_centroids[k] = X[np.random.choice(n_samples)]
       
       # Check for convergence
       if np.sum((new_centroids - centroids) ** 2) < tol:
           break
           
       centroids = new_centroids
   
   return centroids, labels

# Evaluate solution
def evaluate_solution(X, weights, centroids):
   # Calculate distance from each point to the nearest centroid
   distances = pairwise_distances(X, centroids)
   min_distances = np.min(distances, axis=1)
   
   # Calculate weighted average distance
   weighted_avg_distance = np.sum(min_distances * weights) / np.sum(weights)
   
   return weighted_avg_distance

def main():
   # Properly expand file paths
   incidents_path = os.path.expanduser('incidents_2025.csv')
   output_csv_path = os.path.expanduser('police_car_locations.csv')
   output_png_path = os.path.expanduser('police_car_placement.png')
   
   # Load incident data
   incidents = load_data(incidents_path)
   
   # Extract coordinates and priorities
   X = incidents[['latitude', 'longitude']].values
   weights = incidents['priority'].values
   
   # Number of police cars
   n_clusters = 10
   
   # Run multiple times to find the best solution
   best_score = float('inf')
   best_centroids = None
   n_attempts = 100
   
   for i in range(n_attempts):
       print(f"Attempt {i+1}/{n_attempts}")
       centroids, _ = weighted_kmeans(X, weights, n_clusters)
       score = evaluate_solution(X, weights, centroids)
       
       if score < best_score:
           best_score = score
           best_centroids = centroids
           print(f"New best score: {best_score}")
   
   # Save the optimal police car locations to CSV
   pd.DataFrame(best_centroids, columns=['latitude', 'longitude']).to_csv(
       output_csv_path, index=False, header=False
   )
   
   print(f"Final best score: {best_score}")
   print(f"Optimal police car locations saved to '{output_csv_path}'")
   
   # Visualize the solution
   plt.figure(figsize=(10, 8))
   plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, s=weights*3)
   plt.scatter(best_centroids[:, 0], best_centroids[:, 1], c='red', marker='x', s=100)
   plt.title('Optimal Police Car Placement')
   plt.xlabel('Latitude')
   plt.ylabel('Longitude')
   plt.savefig(output_png_path)
   plt.show()

if __name__ == "__main__":
   main()
