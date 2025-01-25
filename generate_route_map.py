import pandas as pd
import folium
from folium.plugins import BeautifyIcon
import matplotlib
import matplotlib.colors as mcolors
import numpy as np

df = pd.read_csv("gemeentehuizen.csv")
df.columns = ["Municipality", "Latitude", "Longitude"]

distance_matrix = pd.read_csv("cycling_distance_matrix.csv", index_col=0)


def get_color(municipality, distance):
    """Return a color based on the ordered position of the distance from a given municipality to all others."""
    all_distances = [distance_matrix.loc[municipality, other]
                     for other in distance_matrix.columns if other != municipality]
    sorted_distances = sorted(all_distances)
    position = sorted_distances.index(distance)
    norm = position / (len(sorted_distances) - 1)
    colormap = matplotlib.colormaps["RdYlGn_r"]  # Red (long) to Green (short)
    rgba = colormap(norm)
    return mcolors.to_hex(rgba)


def get_coordinates(municipality):
    """Retrieve coordinates of a municipality from the CSV file."""
    row = df[df["Municipality"] == municipality]
    if row.empty:
        raise ValueError(f"Municipality '{municipality}' not found in dataset.")
    return row.iloc[0]["Longitude"], row.iloc[0]["Latitude"]


def show_map(municipalities):
    """Display a map with the cycling route and numbered markers using BeautifyIcon."""
    mean_lon, mean_lat = np.mean([get_coordinates(m) for m in municipalities], axis=0)
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)

    route_coords = [get_coordinates(m) for m in municipalities]

    for i, (lon, lat) in enumerate(route_coords):
        folium.Marker(
            location=[lat, lon],
            popup=f"{i+1}. {municipalities[i]}",
            icon=BeautifyIcon(number=i+1, icon_shape="marker", background_color="blue", text_color="white")
        ).add_to(m)

    for i in range(len(route_coords) - 1):
        municipality = municipalities[i]
        distance = distance_matrix.loc[municipalities[i], municipalities[i+1]]
        color = get_color(municipality, distance)

        # Create a black border (slightly thicker)
        folium.PolyLine([(route_coords[i][1], route_coords[i][0]),
                         (route_coords[i+1][1], route_coords[i+1][0])],
                        color="black", weight=8, opacity=.5).add_to(m)

        # Now add the colored polyline on top (slightly thinner)
        folium.PolyLine([(route_coords[i][1], route_coords[i][0]),
                         (route_coords[i+1][1], route_coords[i+1][0])],
                        color=color, weight=4, opacity=1).add_to(m)

    m.save("cycling_route_map.html")
    print("Map saved as cycling_route_map.html. Open this file in a browser to view the route.")


if __name__ == "__main__":
    municipalities = ["Rotterdam", "Delft", "Leiden", "Gouda", "Noordwijk"]

    # Show the route on a map
    show_map(municipalities)
