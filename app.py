# This is the overlay map of raster drone data on an openstreet map
import os
import zipfile

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import streamlit as st
from PIL import Image
from pyproj import Proj, Transformer

# zipfile path
shapefile_path = "data/chlorophyll-index-shapefile.zip"

# Extract the shapefile to a temporary directory
with zipfile.ZipFile(shapefile_path, "r") as zip_ref:
    zip_ref.extractall("/tmp/shapefile")  # Extract to /tmp/shapefile

try:
    shapefile_path = next(
        os.path.join(root, file)
        for root, _, files in os.walk("/tmp/shapefile")
        for file in files
        if file.endswith(".shp")
    )
    print("Found shapefile:", shapefile_path)
except StopIteration:
    raise FileNotFoundError("No .shp file found in the extracted ZIP archive.")

# Read the shapefile
gdf = gpd.read_file(shapefile_path)

# Step 3: Check the CRS and transform if necessary to WGS84 (EPSG:4326)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

# Define UTM zone 37S projections
utm_zone = 37
utm_proj = Proj(proj="utm", zone=utm_zone, ellps="WGS84", south=True)
wgs84_proj = Proj(proj="latlong", datum="WGS84")
transformer = Transformer.from_proj(utm_proj, wgs84_proj)

# Convert the GeoTIFF image into a PNG file and get bounds in WGS84
# Define the path to the data folder
data_folder = "data"

# Define the path to the input GeoTIFF image
input_tif_path = "data/new1.tif"

# Define the output PNG image path
output_png_path = os.path.join(data_folder, "new1.png")

# Open the GeoTIFF image using Rasterio
with rasterio.open(input_tif_path) as src:
    img_array = src.read()  # Read the image as a multi-band image

    # Reorder the bands to RGB
    img_array = np.transpose(img_array, (1, 2, 0))

    # Save the image as a PNG file
    img_rgb = Image.fromarray(img_array.astype(np.uint8))
    img_rgb.save(output_png_path)

    # Convert bounds from UTM to WGS84
    southwest_utm = (src.bounds.left, src.bounds.bottom)
    northeast_utm = (src.bounds.right, src.bounds.top)
    southwest_latlon = transformer.transform(southwest_utm[0], southwest_utm[1])
    northeast_latlon = transformer.transform(northeast_utm[0], northeast_utm[1])

# Calculate the map center point
center_lat = (southwest_latlon[1] + northeast_latlon[1]) / 2
center_lon = (southwest_latlon[0] + northeast_latlon[0]) / 2


# Create the map with Folium
m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

# Add the ImageOverlay
png_path = "data/new1.png"
folium.raster_layers.ImageOverlay(
    image=png_path,
    bounds=[
        [southwest_latlon[1], southwest_latlon[0]],
        [northeast_latlon[1], northeast_latlon[0]],
    ],
    opacity=0.6,
).add_to(m)


# Function to determine color based on chlorophyll index
def get_color(index):
    if 0.138 <= index < 0.292:
        return "#ccffcc"  # Light green
    elif 0.292 <= index < 0.332:
        return "#99ff99"
    elif 0.332 <= index < 0.363:
        return "#66ff66"
    elif 0.363 <= index < 0.407:
        return "#339933"
    else:  # index is between 0.6 and 1.0
        return "#006600"


# Step 7: Add the shapefile overlay with popups for each polygon
for _, row in gdf.iterrows():
    chlorophyll_index = row["_meanRCI"]
    popup_content = f"""
    <b>ID:</b> {row['id']}<br>
    <b>Area:</b> {row['area']}<br>
    <b>Perimeter:</b> {row['perimeter']}<br>
    <b>DN:</b> {row['DN']}<br>
    <b>Chlorophyll Index:</b> {row['_meanRCI']}

    """  # Get the chlorophyll index
    popup = folium.Popup(
        popup_content, max_width=300
    )  # Set max_width to control the width of the popup
    folium.GeoJson(
        row.geometry,
        name="Shapefile Layer",
        style_function=lambda x, index=chlorophyll_index: {
            "fillColor": get_color(index),  # Set fill color based on index
            "color": get_color(index),  # Set border color based on index
            "weight": 2,
            "fillOpacity": 0.6,
        },
        popup=popup,  # Using the _meanRCI column
    ).add_to(m)

st.title("Emali Farm  Dashboard")
map_row = st.container()
with map_row:
    # Render the map in the Streamlit app
    st.components.v1.html(m._repr_html_(), height=400)

hist_row = st.container()
col1, col2 = hist_row.columns(2)


with col1:
    # Load the data
    df = pd.read_csv("data/yield-data.csv")
    # Streamlit title

    # Define the NDRE value ranges
    range1 = (0.06404505, 0.124359014)
    range2 = (0.124359014, 0.184672978)
    range3 = (0.184672978, 0.244986941)

    # Count values in each range
    count_range1 = df[
        (df["MeanNDRE"] >= range1[0]) & (df["MeanNDRE"] < range1[1])
    ].shape[0]
    count_range2 = df[
        (df["MeanNDRE"] >= range2[0]) & (df["MeanNDRE"] < range2[1])
    ].shape[0]
    count_range3 = df[
        (df["MeanNDRE"] >= range3[0]) & (df["MeanNDRE"] <= range3[1])
    ].shape[0]

    # Total count for percentage calculation
    total_count = count_range1 + count_range2 + count_range3

    # Calculate percentages
    percentages = [
        (count_range1 / total_count) * 100,
        (count_range2 / total_count) * 100,
        (count_range3 / total_count) * 100,
    ]

    # Prepare data for histogram
    counts = [count_range1, count_range2, count_range3]

    labels = [
        "0.064 - 0.124",
        "0.124 - 0.184",
        "0.184 - 0.244",
    ]
    labels1 = [
        "Bare soil/Developing plants",
        "Unhealthy/not mature  plants",
        "Healthy plants",
    ]

    colors = sns.color_palette("viridis", len(labels))

    # Create a bar plot
    fig, ax = plt.subplots()
    sns.barplot(x=labels, y=counts, palette="viridis", ax=ax, edgecolor="black")

    # Add percentage labels on top of each bar
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        ax.text(i, count + 1, f"{pct:.2f}%", ha="center", va="bottom", fontsize=10)

    # Labeling the axes
    ax.set_xlabel("Distribution of NDRE Values with Percentages")
    ax.set_ylabel("Count")

    # Add colored rectangles and text on top of the histogram
    # Positioning variables for the top of the plot
    legend_y = 1.25  # Position just above the top of the plot
    legend_spacing = 0.1  # Space between each legend entry
    rect_width = 0.05  # Width of the rectangle

    for i, (color, label, range_text) in enumerate(
        zip(colors, labels1, [(0.064, 0.124), (0.124, 0.184), (0.184, 0.244)])
    ):
        # Draw a rectangle of the specified color
        ax.add_patch(
            plt.Rectangle(
                (0.05, legend_y - i * legend_spacing),
                0.1,
                0.05,
                color=color,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

        # Add text next to the rectangle
        ax.text(
            0.15 + 0.01,  # x (to the right of the rectangle)
            legend_y - i * legend_spacing + legend_spacing / 2 - 0.03,  # y
            f"{label} ({range_text[0]} - {range_text[1]})",
            ha="left",
            va="center",
            fontsize=10,
            color="black",
            transform=ax.transAxes,
        )

    # Display the plot with Streamlit
    st.pyplot(fig)
    # Another histogram


with col2:
    # Define the canopy coverage area ranges

    r1 = (1.0049, 2.4918)
    r2 = (2.4918, 3.9788)
    r3 = (3.9788, 5.4657)
    r4 = (5.4657, 7.000)

    count_r1 = df[(df["area"] >= r1[0]) & (df["area"] < r1[1])].shape[0]
    count_r2 = df[(df["area"] >= r2[0]) & (df["area"] < r2[1])].shape[0]
    count_r3 = df[(df["area"] >= r3[0]) & (df["area"] <= r3[1])].shape[0]
    count_r4 = df[(df["area"] >= r4[0]) & (df["area"] <= r4[1])].shape[0]
    count_total = count_r1 + count_r2 + count_r3 + count_r4
    print(count_total)
    percent = [
        (count_r1 / count_total) * 100,
        (count_r2 / count_total) * 100,
        (count_r3 / count_total) * 100,
        (count_r4 / count_total) * 100,
    ]
    counts1 = [count_r1, count_r2, count_r3, count_r4]
    labels2 = [
        "1.0049 - 2.4918",
        "2.4918 - 3.9788",
        "3.9788 - 5.4657",
        "5.4657 - 6.9526",
    ]
    colors1 = sns.color_palette("muted", len(labels2))

    fig, ax = plt.subplots()
    sns.barplot(x=labels2, y=counts1, palette="muted", ax=ax, edgecolor="black")
    # Add percentage labels on top of each bar
    for i, (count, pct) in enumerate(zip(counts1, percent)):
        ax.text(i, count + 1, f"{pct:.2f}%", ha="center", va="bottom", fontsize=10)
    # Labeling the axes
    ax.set_xlabel("Canopy Coverage")
    ax.set_ylabel("Count")
    st.pyplot(fig)
