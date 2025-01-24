import os
import webview
import time
import threading

# Path to the HTML map file
HTML_FILE_PATH = "cycling_route_map.html"

# File to signal when the map needs to be refreshed
SIGNAL_FILE = "refresh_signal.txt"

# Function to refresh the map (reload the HTML file)


def refresh_map(window):
    """Reload the HTML file to refresh the map."""
    print("Refreshing map...")
    window.load_url(HTML_FILE_PATH)  # Reload the map from the HTML file

# Function to monitor for changes (e.g., the creation of the signal file)


def monitor_for_refresh(window):
    """Check if the signal file exists, and if so, refresh the map."""
    while True:
        if os.path.exists(SIGNAL_FILE):  # If the signal file exists
            refresh_map(window)  # Refresh the map
            os.remove(SIGNAL_FILE)  # Remove the signal file after refreshing
        time.sleep(2)  # Check every 2 seconds

# Function to run pywebview (main thread)


def run_webview():
    """Create the webview window and run it on the main thread."""
    window = webview.create_window("Cycling Route Map", HTML_FILE_PATH, width=800, height=600)

    # Start a background thread to monitor for the refresh signal
    threading.Thread(target=monitor_for_refresh, args=(window,), daemon=True).start()

    webview.start()


if __name__ == "__main__":
    run_webview()
