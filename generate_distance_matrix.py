import pandas as pd
import openrouteservice
import numpy as np
from itertools import combinations
import time
import os

# Load the CSV file
df = pd.read_csv("gemeentehuizen.csv")

# Rename columns for clarity
df.columns = ["Municipality", "Latitude", "Longitude"]

# Initialize OpenRouteService client (Replace 'your_api_key' with a valid API key)
client = openrouteservice.Client(key='5b3ce3597851110001cf62481b7913ac96594970bd84f273b581cbd8')

# Get list of coordinates
locations = list(zip(df["Longitude"], df["Latitude"]))
n = len(locations)

# Load existing distance matrix if it exists
matrix_file = "cycling_distance_matrix.csv"
if os.path.exists(matrix_file):
    distance_df = pd.read_csv(matrix_file, index_col=0)
else:
    distance_df = pd.DataFrame(np.nan, index=df["Municipality"], columns=df["Municipality"])


def get_cycling_distance(coord1, coord2):
    """Fetch cycling distance between two coordinates using OpenRouteService."""
    try:
        route = client.directions(
            coordinates=[coord1, coord2],
            profile='cycling-regular',
            format='geojson'
        )
        return route['features'][0]['properties']['segments'][0]['distance'] / 1000  # Convert meters to km
    except Exception as e:
        print(f"Error fetching distance between {coord1} and {coord2}: {e}")
        return np.nan


# Create or update distance matrix
for i, j in combinations(range(n), 2):
    if pd.isna(distance_df.iloc[i, j]):  # Only compute if missing
        dist = get_cycling_distance(locations[i], locations[j])
        distance_df.iloc[i, j] = dist
        distance_df.iloc[j, i] = dist  # Symmetric matrix
        distance_df.to_csv(matrix_file)  # Save progress after every update
        time.sleep(1.5)  # To avoid hitting API rate limits

print("Distance matrix saved to cycling_distance_matrix.csv")
