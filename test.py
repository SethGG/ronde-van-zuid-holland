import pandas as pd
import numpy as np
import folium
from folium.plugins import Realtime
import matplotlib
import matplotlib.colors as mcolors
import threading
import time
import webview
from flask import Flask, jsonify
from flask_cors import CORS

# Load data
df = pd.read_csv("gemeentehuizen.csv")
df.columns = ["Municipality", "Latitude", "Longitude"]
distance_matrix = pd.read_csv("cycling_distance_matrix.csv", index_col=0)

# Flask app to serve GeoJSON
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

generation = 0
geojson_data = {"type": "FeatureCollection", "features": []}
lock = threading.Lock()


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
    row = df[df["Municipality"] == municipality]
    if row.empty:
        raise ValueError(f"Municipality '{municipality}' not found in dataset.")
    return row.iloc[0]["Longitude"], row.iloc[0]["Latitude"]


def generate_geojson():
    """Continuously generate a new GeoJSON route every 2 seconds."""
    global generation
    global geojson_data
    while True:
        municipalities = list(np.random.choice(df["Municipality"].values, 50, replace=False))
        route_coords = [get_coordinates(m) for m in municipalities]
        features = []

        for i, municipality in enumerate(municipalities):
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
            municipality = municipalities[i]
            distance = distance_matrix.loc[municipalities[i], municipalities[i+1]]
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

        with lock:
            generation += 1
            geojson_data = {"type": "FeatureCollection", "features": features}
        time.sleep(2)


@app.route("/route.geojson")
def serve_geojson():
    with lock:
        resp = jsonify(geojson_data)
        resp.headers["Access-Control-Expose-Headers"] = "Generation"
        resp.headers["Generation"] = generation
    return resp


def show_map():
    """Display a map with a dynamic cycling route using Folium's Realtime plugin."""
    html_file = "cycling_route_map.html"
    mean_lat, mean_lon = 52.0205272081141, 4.537304129039721
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)
    title_html = '<h1 style="position:absolute;z-index:100000;left:6rem" >Generation: <span id=gen></span></h1>'
    m.get_root().html.add_child(folium.Element(title_html))
    m.add_js_link("BeautifyMarkerJS",
                  "https://cdn.jsdelivr.net/gh/marslan390/BeautifyMarker/leaflet-beautify-marker-icon.min.js")
    m.add_css_link("BeautifyMarkerCSS",
                   "https://cdn.jsdelivr.net/gh/marslan390/BeautifyMarker/leaflet-beautify-marker-icon.min.css")
    url = "http://127.0.0.1:5000/route.geojson"
    realtime = Realtime(
        interval=100,
        source=folium.JsCode(
            """(responseHandler, errorHandler) => {{
                    url = '{url}';

                    fetch(url)
                    .then(function(response) {{
                        console.log(response.headers.get('Generation'));
                        document.getElementById('gen').innerText = response.headers.get('Generation');
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
    m.save(html_file)

    webview.create_window("Live Cycling Route", html_file, height=700)
    webview.start()


if __name__ == "__main__":
    threading.Thread(target=generate_geojson, daemon=True).start()
    threading.Thread(target=app.run, kwargs={"port": 5000}, daemon=True).start()
    show_map()
