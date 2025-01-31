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
import logging

# Load data
df = pd.read_csv("gemeentehuizen.csv", index_col=0)
distance_matrix = pd.read_csv("cycling_distance_matrix.csv", index_col=0)
# Cenvert coordinate matrix to NumPy array for fast indexing
coordinates_np = df.to_numpy()
# Convert distance matrix to NumPy array for fast indexing
distance_np = distance_matrix.to_numpy()
index_map = {city: i for i, city in enumerate(distance_matrix.index)}  # Map city names to row/col indices
# Calculate normalized distance rank matrix for color assignment
sorted_indices = np.argsort(distance_np, axis=1)
norm_rank_matrix = np.zeros_like(sorted_indices, dtype=float)
for city_idx in range(distance_np.shape[0]):
    norm_rank_matrix[city_idx, sorted_indices[city_idx]] = np.linspace(0, 1, distance_np.shape[1])


def create_flask(ga: "GeneticAlgorithm"):
    # Flask app to serve GeoJSON
    app = Flask(__name__)
    log = logging.getLogger('werkzeug')
    log.disabled = True

    @app.route("/")
    def serve_map():
        m = generate_map()
        return m.get_root().render()

    @app.route("/route.geojson")
    def serve_geojson():
        track_vars = ga.get_tracking_vars()
        if track_vars and track_vars["generation"] > 0:
            resp = jsonify(generate_geojson(ga.start_idx, track_vars["best_solution"]))
            # resp.headers["Repetition"] = repetition.value
            resp.headers["Generation"] = track_vars["generation"]
            resp.headers["Best-Fitness"] = f"{round(track_vars['best_fitness'])} km" \
                if track_vars["best_fitness"] < np.inf else ""
            return resp
        else:
            return jsonify({"type": "FeatureCollection", "features": []})  # Empty response if not tracking

    return app


def flask_process(ga: "GeneticAlgorithm"):
    app = create_flask(ga)
    app.run(port=5000)


def get_color(city_idx1, city_idx2):
    """Return a color based on the normalized rank of the distance between two cities."""
    norm = norm_rank_matrix[city_idx1, city_idx2]  # Direct lookup of normalized rank
    colormap = matplotlib.colormaps["RdYlGn_r"]  # Red (long) to Green (short)
    rgba = colormap(norm)
    return mcolors.to_hex(rgba)


def calc_fitness(start_idx, solutions_idx):
    """Calculate the total distance of a series of municipalities, ending at the first"""
    # Get start distances (start → first city)
    start_distances = distance_np[start_idx, solutions_idx[:, 0]]

    # Get segment distances (city[i] → city[i+1])
    segment_distances = distance_np[solutions_idx[:, :-1], solutions_idx[:, 1:]]

    # Get end distances (last city → start)
    end_distances = distance_np[solutions_idx[:, -1], start_idx]

    # Compute total fitness (sum of all distances)
    fitness_values = start_distances + segment_distances.sum(axis=1) + end_distances

    return fitness_values


def run_repetitions(ga: "GeneticAlgorithm", reps=20, wait=0):
    time.sleep(wait)
    for _ in range(reps):
        ga.run()


class GeneticAlgorithm:
    def __init__(self, start_municipality="Leiden", population_size=2500, elitism_size=3, tournament_size=5,
                 crossover_rate=0.8, mutation_rate=0.2, max_generations=200, track_globals=False):
        """Encapsulates the Genetic Algorithm to allow optional tracking of global variables."""
        self.start_municipality = start_municipality
        self.start_idx = index_map[start_municipality]
        self.population_size = population_size
        self.elitism_size = elitism_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.track_globals = track_globals

        if self.track_globals:
            self.generation = multiprocessing.Value("i", 0)
            self.global_best_fitness = multiprocessing.Value("d", np.inf)
            # Shared array for solution (size = num_cities - 1)
            self.num_cities = len(df.index) - 1  # Excluding start city
            self.global_best_solution = multiprocessing.RawArray('i', [-1]*self.num_cities)
            self.lock = multiprocessing.Lock()

    def get_tracking_vars(self):
        """Returns current values of tracking variables."""
        if self.track_globals:
            with self.lock:
                solution_np = np.frombuffer(
                    self.global_best_solution,
                    dtype=np.int32
                ).copy()  # Copy to avoid memory view issues
                return {
                    "generation": self.generation.value,
                    "best_fitness": self.global_best_fitness.value,
                    "best_solution": solution_np
                }
        return None  # If tracking is disabled, return nothing

    def run(self):
        """Runs the genetic algorithm, tracking the best solution if a queue is provided."""
        start_idx = index_map[self.start_municipality]
        remaining_municipalities_idx = np.array([index_map[city] for city in df.index.drop(self.start_municipality)])
        solutions = np.array([np.random.permutation(remaining_municipalities_idx) for _ in range(self.population_size)])
        fitness_scores = calc_fitness(start_idx, solutions)
        population = np.array([[s, f] for s, f in zip(solutions, fitness_scores)], dtype=object)

        if self.track_globals:
            with self.lock:
                self.generation.value = 0
                self.global_best_fitness.value = np.inf
                # Reset shared array to invalid state (-1)
                solution_np = np.frombuffer(self.global_best_solution, dtype=np.int32)
                solution_np[:] = -1  # <--- This replaces the original list clearance

        for _ in range(self.max_generations):
            best_solution, best_fitness = population[np.argmin(population[:, 1])]

            if self.track_globals:
                with self.lock:
                    self.generation.value += 1
                    if best_fitness < self.global_best_fitness.value:
                        self.global_best_fitness.value = best_fitness
                        # Copy numpy array to shared memory
                        solution_np = np.frombuffer(
                            self.global_best_solution,
                            dtype=np.int32
                        )
                        np.copyto(solution_np, best_solution)

            # Perform elitism
            elite_population = population[np.argpartition(population[:, 1], self.elitism_size)[:self.elitism_size]]

            # Generate new children
            children = []
            while len(children) < self.population_size - self.elitism_size:
                # Perform tournament selection
                def run_tournament():
                    pool = population[np.random.choice(population.shape[0], size=self.tournament_size, replace=False)]
                    winner = pool[np.argmin(pool[:, 1])]
                    return winner[0]

                parent1 = run_tournament()
                parent2 = run_tournament()

                # Perform PMX crossover
                def crossover(parent1, parent2):
                    crossover_idx1, crossover_idx2 = sorted(
                        np.random.choice(parent1.shape[0] + 1, size=2, replace=False))
                    # Create crossover mapping
                    crossover_parent1 = parent1[crossover_idx1: crossover_idx2]
                    crossover_parent2 = parent2[crossover_idx1: crossover_idx2]

                    p1_to_p2_mapping = {}
                    for i in range(crossover_parent1.shape[0]):
                        p1_to_p2_mapping[crossover_parent1[i]] = crossover_parent2[i]

                    bidirec_mapping = np.arange(parent1.shape[0]+1)
                    for i in p1_to_p2_mapping:
                        if i not in p1_to_p2_mapping.values():
                            j = p1_to_p2_mapping[i]
                            while j in p1_to_p2_mapping:
                                j = p1_to_p2_mapping[j]
                            bidirec_mapping[i] = j
                            bidirec_mapping[j] = i

                    child1 = np.concatenate((
                        np.take(bidirec_mapping, parent1[:crossover_idx1]),
                        crossover_parent2,
                        np.take(bidirec_mapping, parent1[crossover_idx2:])
                    ))
                    child2 = np.concatenate((
                        np.take(bidirec_mapping, parent2[:crossover_idx1]),
                        crossover_parent1,
                        np.take(bidirec_mapping, parent2[crossover_idx2:])
                    ))

                    return child1, child2

                if np.random.rand() < self.crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Perform inversion mutation
                def mutate(child):
                    mutation_idx1, mutation_idx2 = sorted(
                        np.random.choice(parent1.shape[0]+1, size=2, replace=False))
                    mutation_idx1_inverse = mutation_idx1 - 1 if mutation_idx1 > 0 else None
                    child[mutation_idx1: mutation_idx2] = child[mutation_idx2 - 1: mutation_idx1_inverse: -1]

                if np.random.rand() < self.mutation_rate:
                    mutate(child1)
                    mutate(child2)

                children.extend([child1, child2])

            children = np.array(children)
            children_fitness_scores = calc_fitness(start_idx, children)
            children_population = np.array([[s, f] for s, f in zip(children, children_fitness_scores)], dtype=object)
            new_population = np.concatenate((elite_population, children_population))

            population = new_population[:self.population_size]


def generate_geojson(start_idx, solution):
    """Generate a GeoJSON route."""
    full_solution = np.insert(solution, 0, start_idx)  # Add start index to solution
    route_coords = coordinates_np[full_solution]  # Retrieve all coordinates at once
    features = []

    # Create markers for cities
    for i, city_idx in enumerate(full_solution):
        lat, lon = route_coords[i]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)]
            },
            "properties": {
                "id": f"point-{city_idx}",
                "order": i + 1,
                "popup": df.index[city_idx]
            }
        })

    # Create colored lines for route segments
    for i in range(full_solution.shape[0] - 1):
        city_idx1, city_idx2 = full_solution[i], full_solution[i + 1]
        lat, lon = route_coords[i]
        lat_next, lon_next = route_coords[i+1]
        color = get_color(city_idx1, city_idx2)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(lon), float(lat)],
                    [float(lon_next), float(lat_next)]
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


def generate_map(refresh_interval=100):
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
        interval=refresh_interval,
        source=folium.JsCode(
            """(responseHandler, errorHandler) => {{
                    var url = '{url}';

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
            """(f, oldLayer) => {
                if (!oldLayer) {return;}

                var type = f.geometry && f.geometry.type
                var coordinates = f.geometry && f.geometry.coordinates

                switch (type) {
                    case 'Point':
                        var options = {
                            isAlphaNumericIcon: true,
                            borderColor: 'grey',
                            textColor: 'grey',
                            text: f.properties.order,
                            spin: false,
                            innerIconStyle: 'margin-top:10%;'
                        };
                        oldLayer.setIcon(L.BeautifyIcon.icon(options))
                        break;
                    case 'LineString':
                        oldLayer.setLatLngs(L.GeoJSON.coordsToLatLngs(coordinates, type === 'LineString' ? 0 : 1));
                        break;
                    default:
                    return null;
                }

                return oldLayer;
                }
            """
        ),
        point_to_layer=folium.JsCode(
            """(f, latlng) => {
                var options = {
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
    ga = GeneticAlgorithm(track_globals=True)

    flask_proc = multiprocessing.Process(target=flask_process, kwargs={"ga": ga}, daemon=True)
    flask_proc.start()

    ga_proc = multiprocessing.Process(target=run_repetitions, kwargs={"ga": ga, "wait": 2}, daemon=True)
    ga_proc.start()

    webview.create_window("Live Cycling Route", url="http://127.0.0.1:5000", height=700)
    webview.start(gui="qt")
