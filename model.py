#!/usr/bin/env python
# coding: utf-8

# ====== Import the required modules
import requests
import pandas as pd
import numpy as np
import random
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster

# ====== Create the dataset for a business
overpass_url = "http://overpass-api.de/api/interpreter"
overpass_query = """
[out:json][timeout:60];
node["name"="Starbucks"]
  (24.3963,-125.0,49.3844,-66.9346);
out body;
"""
response = requests.get(overpass_url, params={"data": overpass_query})
data = response.json()

if "elements" not in data or len(data["elements"]) == 0:
    print("No data returned.")
else:
    locations = [{"name": element["tags"]["name"], "lat": element["lat"], "lon": element["lon"]} 
                 for element in data["elements"]]
    df = pd.DataFrame(locations)
    df["location_id"] = [f"SB-{str(i+1).zfill(5)}" for i in range(len(df))]
    df["microwaves"] = [random.randint(1, 3) for _ in range(len(df))]
    df["refrigerators"] = [random.randint(2, 5) for _ in range(len(df))]
    df["hvac_units"] = [random.randint(1, 4) for _ in range(len(df))]
    df["ovens"] = [random.randint(1, 3) for _ in range(len(df))]
    df["coffee_makers"] = [random.randint(3, 6) for _ in range(len(df))]
    df["ice_machines"] = [random.randint(1, 2) for _ in range(len(df))]
    df["dishwashers"] = [random.randint(1, 2) for _ in range(len(df))]
    df["workload"] = (
        df["microwaves"] * 2 + df["refrigerators"] * 4 + df["hvac_units"] * 10 +
        df["ovens"] * 3 + df["coffee_makers"] * 5 + df["ice_machines"] * 4 + df["dishwashers"] * 3
    )

# ====== Statistical analysis
unique_locations = df["location_id"].nunique()
total_workload = df["workload"].sum()
mean_workload = df["workload"].mean()
std_workload = df["workload"].std()
max_workload_per_contractor = unique_locations * 0.2     # New threshold for ~1,630 contractors or 20% of the total workload
target_workload = total_workload/unique_locations
lower_bound = max_workload_per_contractor * 0.75     # 
upper_bound = max_workload_per_contractor * 1.25     #
initial_contractors = math.ceil(total_workload / max_workload_per_contractor)

# ====== Normalize workload with MinMaxScaler for visualization or further analysis
scaler = MinMaxScaler()
df["normalized_workload"] = scaler.fit_transform(df[["workload"]])

# ====== Output results
print(f"Processed {unique_locations} Starbucks locations")
print(f"Total Workload: {total_workload}")
print(f"Mean Workload per Location: {mean_workload:.2f}, Std Dev: {std_workload:.2f}")
print(f"Max workload per contractor: {max_workload_per_contractor:.2f}")
print(f"Target Workload per Contractor: {target_workload:.2f}")
print(f"Workload Bounds: {lower_bound:.2f} to {upper_bound:.2f}")
print(f"Starting with {initial_contractors} contractors...")

# ====== Clustering based on model on lat/lon
# ====== Create clusters
X = df[["lat", "lon"]].values
kmeans = KMeans(n_clusters=initial_contractors, random_state=42).fit(X)

# Assign cluster labels to the DataFrame
df["cluster"] = kmeans.labels_  # This is the missing line that adds cluster assignments

# ====== Get cluster centroids (contractor locations)
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=["centroid_lat", "centroid_lon"])
centroids["contractor_id"] = [f"C-{str(i+1).zfill(3)}" for i in range(len(centroids))]

# ====== Define Haversine distance function with explicit numpy usage
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ====== Seed each contractor with nearest location
df["distance_to_centroid"] = np.inf
df["contractor_id"] = None
for _, centroid in centroids.iterrows():
    cid = centroid["contractor_id"]
    distances = haversine(df["lat"], df["lon"], centroid["centroid_lat"], centroid["centroid_lon"])
    df.loc[distances < df["distance_to_centroid"], "contractor_id"] = cid
    df.loc[distances < df["distance_to_centroid"], "distance_to_centroid"] = distances[distances < df["distance_to_centroid"]]

# ====== Iterative workload balancing
max_iterations = 100  # Prevent infinite loops
iteration = 0
unbalanced = True

while unbalanced and iteration < max_iterations:
    cluster_workloads = df.groupby("contractor_id")["workload"].sum()
    unbalanced = False

    # Check for contractors outside bounds
    for contractor_id in cluster_workloads.index:
        workload = cluster_workloads[contractor_id]
        contractor_df = df[df["contractor_id"] == contractor_id].copy()

        # Overloaded contractors: push farthest location to nearest neighbor
        if workload > upper_bound and len(contractor_df) > 1:
            unbalanced = True
            centroid = centroids[centroids["contractor_id"] == contractor_id].iloc[0]
            contractor_df["distance"] = haversine(
                contractor_df["lat"], contractor_df["lon"],
                centroid["centroid_lat"], centroid["centroid_lon"]
            )
            farthest = contractor_df.sort_values("distance", ascending=False).iloc[0]
            excess_workload = farthest["workload"]

            # Find nearest centroid with room (fix the indexing issue)
            # Create a mask directly using centroids and cluster_workloads
            workloads_mapped = pd.Series(
                cluster_workloads.reindex(centroids["contractor_id"]).fillna(0).values,
                index=centroids.index
            )
            mask = (workloads_mapped + excess_workload <= upper_bound) & (centroids["contractor_id"] != contractor_id)
            candidate_contractors = centroids[mask].copy()

            if not candidate_contractors.empty:
                candidate_contractors["distance"] = haversine(
                    farthest["lat"], farthest["lon"],
                    candidate_contractors["centroid_lat"], candidate_contractors["centroid_lon"]
                )
                nearest_contractor = candidate_contractors.sort_values("distance").iloc[0]["contractor_id"]
                df.loc[df["location_id"] == farthest["location_id"], "contractor_id"] = nearest_contractor
                df.loc[df["location_id"] == farthest["location_id"], "distance_to_centroid"] = haversine(
                    farthest["lat"], farthest["lon"],
                    centroids[centroids["contractor_id"] == nearest_contractor]["centroid_lat"].iloc[0],
                    centroids[centroids["contractor_id"] == nearest_contractor]["centroid_lon"].iloc[0]
                )

        # Underloaded contractors: pull nearest location from overloaded neighbor
        elif workload < lower_bound:
            unbalanced = True
            centroid = centroids[centroids["contractor_id"] == contractor_id].iloc[0]
            # Find neighbors with excess workload
            overloaded = cluster_workloads[cluster_workloads > upper_bound].index
            if not overloaded.empty:
                neighbor_dfs = df[df["contractor_id"].isin(overloaded)].copy()
                neighbor_dfs["distance"] = haversine(
                    neighbor_dfs["lat"], neighbor_dfs["lon"],
                    centroid["centroid_lat"], centroid["centroid_lon"]
                )
                closest = neighbor_dfs.sort_values("distance").iloc[0]
                if cluster_workloads[closest["contractor_id"]] - closest["workload"] >= lower_bound:
                    df.loc[df["location_id"] == closest["location_id"], "contractor_id"] = contractor_id
                    df.loc[df["location_id"] == closest["location_id"], "distance_to_centroid"] = closest["distance"]

    iteration += 1

# ====== Verify results
final_workloads = df.groupby("contractor_id")["workload"].sum()
location_counts = df.groupby("contractor_id")["location_id"].count()
within_bounds_count = ((final_workloads >= lower_bound) & (final_workloads <= upper_bound)).sum()
outside_bounds_count = len(final_workloads) - within_bounds_count

print(f"\nIterations run: {iteration}")
print(f"Contractors within bounds: {within_bounds_count}")
print(f"Contractors outside bounds: {outside_bounds_count}")
print(f"Total contractors: {len(final_workloads)}")
print(f"Percentage within bounds: {(within_bounds_count / len(final_workloads) * 100):.2f}%")
print(f"Unassigned locations: {df['contractor_id'].isna().sum()}")

# ====== Mapping of the results
# ====== Define out_of_bounds based on final_workloads
final_workloads = df.groupby("contractor_id")["workload"].sum()
out_of_bounds = final_workloads[(final_workloads < lower_bound) | (final_workloads > upper_bound)].index.tolist()
print(f"Contractors outside bounds: {len(out_of_bounds)}")

# ====== Create a base map centered on the mean of all locations
map_center = [df["lat"].mean(), df["lon"].mean()]
m = folium.Map(location=map_center, zoom_start=4, tiles="OpenStreetMap")

# ====== Assign a unique color to each contractor, ensuring all centroid IDs are included
unique_contractors = centroids["contractor_id"].unique()
colors = [
    'red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue', 'pink', 'gray', 'black',
    'lightgreen', 'darkblue', 'darkpurple', 'cadetblue', 'darkgreen', 'lightgray', 'beige'
] * (len(unique_contractors) // 17 + 1)  # Repeat colors if needed
contractor_color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(unique_contractors)}

# Diagnostic: Check for missing IDs
centroid_ids = set(centroids["contractor_id"])
df_ids = set(df["contractor_id"].dropna())  # Drop NaN if any unassigned locations exist
missing_in_df = centroid_ids - df_ids
print(f"Centroid IDs not in df: {missing_in_df}")
print(f"Number of unique contractors in centroids: {len(unique_contractors)}")
print(f"Number of unique contractors in df: {len(df_ids)}")

# ====== Add locations to the map with MarkerCluster
marker_cluster = MarkerCluster().add_to(m)
for _, row in df.iterrows():
    contractor_id = row["contractor_id"]
    if pd.isna(contractor_id):  # Skip unassigned locations
        continue
    workload = final_workloads.get(contractor_id, 0)
    color = contractor_color_map[contractor_id]
    weight = 3 if contractor_id in out_of_bounds else 1
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color=color,
        weight=weight,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=f"Contractor: {contractor_id}<br>Location: {row['location_id']}<br>Workload: {row['workload']}<br>Total Contractor Workload: {workload:.2f}"
    ).add_to(marker_cluster)

# ====== Add centroids to the map with a fallback color
for _, centroid in centroids.iterrows():
    contractor_id = centroid["contractor_id"]
    try:
        color = contractor_color_map[contractor_id]
    except KeyError:
        print(f"Warning: {contractor_id} not in color map, using fallback 'gray'")
        color = 'gray'  # Fallback color for unexpected IDs
    folium.Marker(
        location=[centroid["centroid_lat"], centroid["centroid_lon"]],
        popup=f"Contractor: {contractor_id}<br>Centroid",
        icon=folium.Icon(color=color, icon="star", prefix="fa")
    ).add_to(m)

# ====== Add a legend
legend_html = '''
     <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; height: auto; 
     border:2px solid grey; z-index:9999; font-size:14px; background-color:white; padding: 10px;">
     <b>Legend</b><br>
     <i class="fa fa-star" style="color:black"></i> Contractor Centroid<br>
     <b>Circle Colors:</b> Contractor IDs<br>
     <b>Bold Outline:</b> Contractor workload outside bounds ({lower_bound:.2f} - {upper_bound:.2f})
     </div>
     '''.format(lower_bound=lower_bound, upper_bound=upper_bound)
m.get_root().html.add_child(folium.Element(legend_html))

# ====== Save and display the map
m.save("starbucks_contractor_map.html")
print("Map saved as 'starbucks_contractor_map.html'. Open it in a browser to explore.")

# ====== Basic analysis of out-of-bounds contractors
out_of_bounds_df = df[df["contractor_id"].isin(out_of_bounds)]
out_of_bounds_workloads = final_workloads[out_of_bounds]
print("\nOut-of-Bounds Contractors Summary:")
print(f"Average workload: {out_of_bounds_workloads.mean():.2f}")
print(f"Min workload: {out_of_bounds_workloads.min():.2f}")
print(f"Max workload: {out_of_bounds_workloads.max():.2f}")
print(f"Average locations per contractor: {out_of_bounds_df.groupby(
    'contractor_id')['location_id'].count().mean():.2f}")
