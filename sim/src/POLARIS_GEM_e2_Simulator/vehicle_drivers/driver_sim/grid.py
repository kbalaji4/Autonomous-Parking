import numpy as np
import matplotlib.pyplot as plt

# Reference origin (lower-left corner of your bounding box)
origin_lat = 40.092735
origin_lon = -88.236201
R = 6378137  # Earth radius in meters

# Grid resolution in meters
grid_res = 1.0

# GPS coordinates (lat, lon)
latlon_points = [
    (40.092751, -88.236192),
    (40.092900, -88.236201),
    (40.092897, -88.235190),
    (40.092735, -88.235186)
]

# Convert GPS to local XY using equirectangular approximation
def latlon_to_xy(lat, lon, origin_lat, origin_lon):
    dx = (lon - origin_lon) * np.cos(np.radians((lat + origin_lat) / 2)) * (np.pi / 180) * R
    dy = (lat - origin_lat) * (np.pi / 180) * R
    return dx, dy

# Convert all coordinates to XY
xy_points = [latlon_to_xy(lat, lon, origin_lat, origin_lon) for lat, lon in latlon_points]
x_vals, y_vals = zip(*xy_points)

# Determine grid bounds
x_min, x_max = min(x_vals), max(x_vals)
y_min, y_max = min(y_vals), max(y_vals)
width = int(np.ceil((x_max - x_min) / grid_res)) + 1
height = int(np.ceil((y_max - y_min) / grid_res)) + 1

# Function to convert GPS to grid cell index
def gps_to_grid(lat, lon):
    x, y = latlon_to_xy(lat, lon, origin_lat, origin_lon)
    col = int((x - x_min) / grid_res)
    row = height - int((y - y_min) / grid_res) - 1  # flip y-axis for display
    return row, col

# --- Plotting ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Grid Map Covering GPS Area")
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_aspect('equal')
ax.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.5)

# Plot GPS corners as red dots
for lat, lon in latlon_points:
    row, col = gps_to_grid(lat, lon)
    ax.plot(col, row, 'ro')  # red dot
    ax.text(col + 0.5, row + 0.5, f"({lat:.5f}, {lon:.5f})", fontsize=6)

plt.xlabel("X (Grid Columns)")
plt.ylabel("Y (Grid Rows)")
plt.tight_layout()
plt.show()
