import pandas as pd
import numpy as np
import folium
from folium.plugins import Realtime
import matplotlib
import matplotlib.colors as mcolors
import threading
import webview
from flask import Flask, jsonify

# Load data
df = pd.read_csv("gemeentehuizen.csv", index_col=0)
distance_matrix = pd.read_csv("cycling_distance_matrix.csv", index_col=0)

# Global variables and lock
generation = 0
global_best_fitness = np.inf
geojson_data = {"type": "FeatureCollection", "features": []}
lock = threading.Lock()

# Flask app to serve GeoJSON
app = Flask(__name__)


@app.route("/")
def serve_map():
    m = generate_map()
    return m.get_root().render()


@app.route("/route.geojson")
def serve_geojson():
    with lock:
        resp = jsonify(geojson_data)
        resp.headers["Generation"] = generation
        resp.headers["Best-Fitness"] = f"{round(global_best_fitness)} km"
    return resp


def get_color(municipality, distance):
    """Return a color based on the ordered position of the distance from a given municipality to all others."""
    all_distances = [distance_matrix.loc[municipality, other]
                     for other in distance_matrix.columns if other != municipality]
    sorted_distances = sorted(all_distances)
    position = sorted_distances.index(distance)
    norm = position / (len(sorted_distances) - 1)
    colormap = matplotlib.colormaps["RdYlGn_r"]
    rgba = colormap(norm)
    return mcolors.to_hex(rgba)


def get_coordinates(municipality):
    """Retrieve coordinates of a municipality from the CSV file."""
    row = df.loc[municipality]
    if row.empty:
        raise ValueError(f"Municipality '{municipality}' not found in dataset.")
    return row.loc["Longitude"], row.loc["Latitude"]


def calc_fitness(start_municipality, solution):
    """Calculate the the total distance of a series of municipalities, ending at the first"""
    fitness = distance_matrix.loc[start_municipality, solution[0]]
    for i in range(len(solution) - 1):
        fitness += distance_matrix.loc[solution[i], solution[i+1]]
    fitness += distance_matrix.loc[solution[-1], start_municipality]
    return fitness


def genetic_algorihtm(start_municipality="Leiden", population_size=30, elitism_size=3, tournament_size=5, mutation_rate=0.1):
    """Implementation of a GA for finding a solution with a low fitness score"""

    remaining_municipalities = df.index.drop(start_municipality)
    solutions = [list(np.random.choice(remaining_municipalities, size=remaining_municipalities.shape[0],
                                       replace=False)) for _ in range(population_size)]
    population = [(s, calc_fitness(start_municipality, s)) for s in solutions]

    global generation
    global global_best_fitness
    global geojson_data

    while True:
        sorted_population = sorted(population, key=lambda p: p[1])
        best_solution, best_fitness = sorted_population[0]

        with lock:
            generation += 1
            if best_fitness < global_best_fitness:
                global_best_fitness = best_fitness
                geojson_data = generate_geojson([start_municipality]+best_solution)

        # Perform elitism
        new_population = sorted_population[:elitism_size]

        while len(new_population) < population_size:
            # Perform tournament selection
            def run_tournament():
                pool = sorted((population[choice] for choice in np.random.choice(
                    len(population), size=tournament_size, replace=False)), key=lambda p: p[1])
                return pool[0][0]
            parent1 = run_tournament()
            parent2 = run_tournament()

            # Perform PMX crossover
            child1, child2 = [], []
            crossover_idx1, crossover_idx2 = sorted(np.random.choice(len(parent1)+1, size=2, replace=False))
            for idx in range(len(parent1)):
                if idx >= crossover_idx1 and idx < crossover_idx2:
                    child1.append(parent2[idx])
                    child2.append(parent1[idx])
                else:
                    for parent, other_parent, child in ((parent1, parent2, child1), (parent2, parent1, child2)):
                        candidate = parent[idx]
                        while candidate in other_parent[crossover_idx1:crossover_idx2]:
                            candidate_idx = other_parent.index(candidate)
                            candidate = parent[candidate_idx]
                        child.append(candidate)
            # Perform inversion mutation
            for child in (child1, child2):
                if np.random.rand() <= mutation_rate:
                    mutation_idx1, mutation_idx2 = sorted(np.random.choice(len(parent1)+1, size=2, replace=False))
                    mutation_idx1_inverse = mutation_idx1 - 1 if mutation_idx1 > 0 else None
                    child[mutation_idx1: mutation_idx2] = child[mutation_idx2 - 1: mutation_idx1_inverse: -1]

                    new_population.append((child, calc_fitness(start_municipality, child)))

        population = new_population


def generate_geojson(solution):
    """Continuously generate a new GeoJSON route every 2 seconds."""
    route_coords = [get_coordinates(m) for m in solution]
    features = []

    for i, municipality in enumerate(solution):
        lon, lat = get_coordinates(municipality)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "id": municipality,
                "order": i + 1,
                "popup": municipality
            }
        })

    for i in range(len(route_coords) - 1):
        municipality = solution[i]
        distance = distance_matrix.loc[solution[i], solution[i+1]]
        color = get_color(municipality, distance)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [route_coords[i][0], route_coords[i][1]],
                    [route_coords[i+1][0], route_coords[i+1][1]]
                ]
            },
            "properties": {
                "id": f"line-{i}",
                "color": color,
                "weight": 4,
                "opacity": 1
            }
        })

    return {"type": "FeatureCollection", "features": features}


def generate_map():
    """Display a map with a dynamic cycling route using Folium's Realtime plugin."""
    mean_lat, mean_lon = 52.0205272081141, 4.537304129039721
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)
    title_html = '<div style="position:absolute;z-index:100000;left:6rem">\
        <h1 style="margin-bottom:0">Generation: <span id=gen></span></h1>\
        <h2>Best Fitness: <span id=fit></span></h2></div>'
    m.get_root().html.add_child(folium.Element(title_html))
    m.add_js_link("BeautifyMarkerJS",
                  "https://cdn.jsdelivr.net/gh/marslan390/BeautifyMarker/leaflet-beautify-marker-icon.min.js")
    m.add_css_link("BeautifyMarkerCSS",
                   "https://cdn.jsdelivr.net/gh/marslan390/BeautifyMarker/leaflet-beautify-marker-icon.min.css")
    url = "http://127.0.0.1:5000/route.geojson"
    realtime = Realtime(
        interval=500,
        source=folium.JsCode(
            """(responseHandler, errorHandler) => {{
                    url = '{url}';

                    fetch(url)
                    .then(function(response) {{
                        console.log(response.headers.get('Generation'));
                        document.getElementById('gen').innerText = response.headers.get('Generation');
                        document.getElementById('fit').innerText = response.headers.get('Best-Fitness');
                        return response.json();
                    }})
                    .then(responseHandler)
                    .catch(errorHandler);
                }}
            """.format(url=url)
        ),
        update_feature=folium.JsCode(
            """() => { return; }
            """
        ),
        point_to_layer=folium.JsCode(
            """(f, latlng) => {
                options = {
                    isAlphaNumericIcon: true,
                    borderColor: 'grey',
                    textColor: 'grey',
                    text: f.properties.order,
                    spin: false,
                    innerIconStyle: 'margin-top:10%;'
                };
                return L.marker(latlng, {icon: L.BeautifyIcon.icon(options)});
                }
            """
        ),
        on_each_feature=folium.JsCode(
            """(f, layer) => {
                layer.bindPopup(f.properties.popup);
                }
            """
        ),
        style=folium.JsCode(
            """(f) => {
                if (f.geometry.type === 'Point') {
                    return {color: 'black', fillColor: 'white', radius: 8};
                }
                return {color: f.properties.color, weight: f.properties.weight, opacity: f.properties.opacity};
            }"""
        )
    )
    realtime.add_to(m)

    return m


if __name__ == "__main__":
    threading.Thread(target=genetic_algorihtm, daemon=True).start()
    threading.Thread(target=app.run, kwargs={"port": 5000}, daemon=True).start()

    webview.create_window("Live Cycling Route", url="http://127.0.0.1:5000", height=700)
    webview.start()
