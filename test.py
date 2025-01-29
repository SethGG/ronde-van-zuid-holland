import pandas as pd
import numpy as np
import folium
from folium.plugins import Realtime
import matplotlib
import matplotlib.colors as mcolors
import multiprocessing
import webview
from flask import Flask, jsonify
import time
import heapq

# Load data
df = pd.read_csv("gemeentehuizen.csv", index_col=0)
distance_matrix = pd.read_csv("cycling_distance_matrix.csv", index_col=0)
# Convert distance matrix to NumPy array for fast indexing
distance_np = distance_matrix.to_numpy()
index_map = {city: i for i, city in enumerate(distance_matrix.index)}  # Map city names to row/col indices

# Manager for shared state
manager = multiprocessing.Manager()
repetition = manager.Value("i", 0)
generation = manager.Value("i", 0)
global_best_fitness = manager.Value("d", np.inf)
geojson_data = manager.dict({"type": "FeatureCollection", "features": []})
lock = manager.Lock()


def create_flask():
    # Flask app to serve GeoJSON
    app = Flask(__name__)

    @app.route("/")
    def serve_map():
        m = generate_map()
        return m.get_root().render()

    @app.route("/route.geojson")
    def serve_geojson():
        with lock:
            resp = jsonify(geojson_data.copy())  # Convert manager.dict() to regular dict
            resp.headers["Repetition"] = repetition.value
            resp.headers["Generation"] = generation.value
            resp.headers["Best-Fitness"] = f"{round(global_best_fitness.value)} km" \
                if global_best_fitness.value < np.inf else ""
        return resp

    return app


def flask_process():
    app = create_flask()
    app.run(port=5000)


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


def calc_fitness(start_municipality, solutions):
    """Calculate the total distance of a series of municipalities, ending at the first"""
    # Convert city names in solutions to integer indices
    solutions_idx = np.vectorize(index_map.get)(solutions)  # Shape: (population_size, num_municipalities)
    start_idx = index_map[start_municipality]

    # Get start distances (start → first city)
    start_distances = distance_np[start_idx, solutions_idx[:, 0]]

    # Get segment distances (city[i] → city[i+1])
    segment_distances = distance_np[solutions_idx[:, :-1], solutions_idx[:, 1:]]

    # Get end distances (last city → start)
    end_distances = distance_np[solutions_idx[:, -1], start_idx]

    # Compute total fitness (sum of all distances)
    fitness_values = start_distances + segment_distances.sum(axis=1) + end_distances

    return fitness_values


def run_repetitions(reps=10, wait=0):
    time.sleep(wait)
    for _ in range(reps):
        with lock:
            repetition.value += 1
        genetic_algorithm()


def genetic_algorithm(start_municipality="Leiden", population_size=30, elitism_size=3, tournament_size=5,
                      mutation_rate=0.3, max_generations=1000):
    """Implementation of a GA for finding a solution with a low fitness score"""

    remaining_municipalities = df.index.drop(start_municipality)
    solutions = [list(np.random.choice(remaining_municipalities, size=remaining_municipalities.shape[0],
                                       replace=False)) for _ in range(population_size)]
    fitness_scores = calc_fitness(start_municipality, solutions)
    population = list(zip(solutions, fitness_scores))

    with lock:
        generation.value = 0

    while generation.value < max_generations:
        best_solution, best_fitness = min(population, key=lambda p: p[1])

        with lock:
            generation.value += 1
            if best_fitness < global_best_fitness.value:
                global_best_fitness.value = best_fitness
                geojson_data.clear()
                geojson_data.update(generate_geojson([start_municipality] + best_solution))

        # Perform elitism
        elite_population = heapq.nsmallest(elitism_size, population, key=lambda p: p[1])

        # Generate new children
        new_children = []
        while len(new_children) < population_size - elitism_size:
            # Perform tournament selection
            def run_tournament():
                pool = sorted((population[choice] for choice in np.random.choice(
                    len(population), size=tournament_size, replace=False)), key=lambda p: p[1])
                return pool[0][0]

            parent1 = run_tournament()
            parent2 = run_tournament()

            # Perform PMX crossover
            def crossover(parent1, parent2):
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
                return child1, child2

            child1, child2 = crossover(parent1, parent2)

            # Perform inversion mutation
            def mutate(child):
                if np.random.rand() <= mutation_rate:
                    mutation_idx1, mutation_idx2 = sorted(np.random.choice(len(parent1)+1, size=2, replace=False))
                    mutation_idx1_inverse = mutation_idx1 - 1 if mutation_idx1 > 0 else None
                    child[mutation_idx1: mutation_idx2] = child[mutation_idx2 - 1: mutation_idx1_inverse: -1]

            mutate(child1)
            mutate(child2)
            new_children.extend([child1, child2])

        child_fitness_scores = calc_fitness(start_municipality, new_children)
        new_population = elite_population + list(zip(new_children, child_fitness_scores))

        population = new_population[:population_size]


def generate_geojson(solution):
    """Generate a GeoJSON route."""
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
        <h1 style="margin-bottom:0">Repetition: <span id=rep></span></h1>\
        <h1 style="margin-top:0;margin-bottom:0">Generation: <span id=gen></span></h1>\
        <h2>Best Fitness: <span id=fit></span></h2></div>'
    m.get_root().html.add_child(folium.Element(title_html))
    m.add_js_link("BeautifyMarkerJS",
                  "https://cdn.jsdelivr.net/gh/marslan390/BeautifyMarker/leaflet-beautify-marker-icon.min.js")
    m.add_css_link("BeautifyMarkerCSS",
                   "https://cdn.jsdelivr.net/gh/marslan390/BeautifyMarker/leaflet-beautify-marker-icon.min.css")
    url = "http://127.0.0.1:5000/route.geojson"
    realtime = Realtime(
        interval=250,
        source=folium.JsCode(
            """(responseHandler, errorHandler) => {{
                    url = '{url}';

                    fetch(url)
                    .then(function(response) {{
                        console.log(response.headers.get('Generation'));
                        document.getElementById('rep').innerText = response.headers.get('Repetition');
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
    flask_proc = multiprocessing.Process(target=flask_process, daemon=True)
    flask_proc.start()

    ga_proc = multiprocessing.Process(target=run_repetitions, kwargs={"wait": 2}, daemon=True)
    ga_proc.start()

    webview.create_window("Live Cycling Route", url="http://127.0.0.1:5000", height=700)
    webview.start()
