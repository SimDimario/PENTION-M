import osmnx as ox
from shapely.geometry import box
import numpy as np
import logging
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm  
import os
import pandas as pd
import json
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fix_bbox_order(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Assicura che la bounding box sia nell'ordine corretto: (min_lon, min_lat, max_lon, max_lat).
    """
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError("Bounding box must be a tuple of four values: (min_lon, min_lat, max_lon, max_lat).")
    
    min_lon, min_lat, max_lon, max_lat = bbox
    min_lon, max_lon = sorted([min_lon, max_lon])
    min_lat, max_lat = sorted([min_lat, max_lat])
    return (min_lon, min_lat, max_lon, max_lat)

def generate_binary_map(
        place: str = "", 
        grid_size: int = 300,
        bbox: Optional[Tuple[float, float, float, float]] = None
        ) -> tuple[np.ndarray, dict]:
    """
    Genera una mappa binaria (0 = edificio, 1 = spazio libero) su una bounding box o area OSM.

    Args:
        place (str): Nome del luogo (se bbox è None).
        grid_size (int): Numero di celle per lato della griglia.
        bbox (tuple): Bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.

    Returns:
        np.ndarray: Mappa binaria (1 = libero, 0 = edificio).
        dict: Metadati.
    """

    logging.info(f"Generating binary map for {place}...")

    bbox=fix_bbox_order(bbox)
    logging.info(f"Using bounding box: {bbox}" if bbox else "Using place name for OSM query.")

    try:

        if bbox:
            bounds_polygon = box(*bbox)
            gdf_bounds = gpd.GeoDataFrame(geometry=[bounds_polygon], crs="EPSG:4326") #geoframe box
            gdf_bounds_proj = gdf_bounds.to_crs(epsg=32633) #Converte il GeoDataFrame da lat/lon (EPSG:4326) a coordinate UTM (EPSG:32633)
            bounds = gdf_bounds_proj.total_bounds #Estrae le coordinate minime e massime del poligono proiettato (metri)
        else:
            gdf_place = ox.geocode_to_gdf(place) #stesso di if
            gdf_place_proj = gdf_place.to_crs(epsg=32633)
            bounds = gdf_place_proj.total_bounds

        # Scarica edifici da OSM
        tags = {"building": True, "height": True}
        buildings = ox.features_from_place(place, tags=tags) if not bbox else ox.features_from_bbox(bbox, tags=tags) #type: ignore
        
        if buildings.empty:
            logging.warning(f"No building data found for {place}. Returning an empty map.")
            return np.zeros((grid_size, grid_size), dtype=np.uint8), {}
        
        logging.info(f"Found {len(buildings)} building features.")

    except Exception as e:
        logging.error(f"Error retrieving building data for {place}: {e}")
        return np.zeros((grid_size, grid_size), dtype=np.uint8), {}

    
    buildings_proj = buildings.to_crs(epsg=32633) #Proietta i dati degli edifici in coordinate UTM (EPSG:32633)
    logging.info(f"Buildings projected to EPSG:32633 CRS.")
    
    # Get full bounding box of all buildings
    x_min, y_min, x_max, y_max = bounds
    logging.info(f"Total bounds: xmin={x_min:.1f}, ymin={y_min:.1f}, xmax={x_max:.1f}, ymax={y_max:.1f}")

    # Compute grid cell size <-> resolution
    cell_width = (x_max - x_min) / grid_size #metres
    print(f"Cell width: {cell_width:.2f} metres")
    cell_height = (y_max - y_min) / grid_size #metres
    binary_grid = np.ones((grid_size, grid_size), dtype=np.uint8)

    logging.info(f"Creating {grid_size}x{grid_size} grid over entire city area.")

    # Ottimizzazione: crea spatial index per ricerche più veloci
    buildings_sindex = buildings_proj.sindex

    # Riemipimento binary grid
    for i in tqdm(range(grid_size), desc="Processing grid"):
        for j in range(grid_size):
            cell = box(
                x_min + i * cell_width,
                y_min + j * cell_height,
                x_min + (i + 1) * cell_width,
                y_min + (j + 1) * cell_height,
            )

            # Usa spatial index per ricerca più efficiente
            possible_matches_index = list(buildings_sindex.intersection(cell.bounds))
            possible_matches = buildings_proj.iloc[possible_matches_index]

            # Controlla intersezioni 
            if not possible_matches.empty and possible_matches.intersects(cell).any():
                binary_grid[j, i] = 0

    # Calcola statistiche
    total_cells = grid_size * grid_size
    building_cells = np.sum(binary_grid == 0)
    free_cells = np.sum(binary_grid == 1)

    building_density_percent = ( building_cells / total_cells) * 100

    # Metadata
    metadata = {
        'city': place,
        'grid_size': grid_size,
        'bounds': (x_min, y_min, x_max, y_max),
        'cell_size': (cell_width, cell_height),
        'crs': "epsg 32633",
        'total_buildings': len(buildings_proj),
        'building_cells': building_cells,
        'free_cells': free_cells,
        'total_cells': total_cells,
        'resolution (m)': cell_width,
        'building_density': building_density_percent,
        'mean_height': pd.to_numeric(buildings_proj['height'], errors='coerce').mean() if 'height' in buildings_proj.columns else None
    }

    logging.info("Binary map generation complete.")
    logging.info(f"Statistics: {building_cells}/{total_cells} cells with buildings ({building_cells/total_cells*100:.1f}%)")
    
    return binary_grid, metadata

def convert_np(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)): #type: ignore
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):  #type: ignore
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [convert_np(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    else:
        return obj

if __name__ == "__main__":

    target_city = "Roma, Italy"
    
    # Formato: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
    #41.128370,14.774086,41.133989,14.791138->(lat_min, lon_min, lat_max, lon_max).
    quartiere_bbox = (12.478107, 41.894985,12.495397, 41.903454) # https://bboxfinder.com/
    binary_map, metadata= generate_binary_map(place=target_city,bbox=quartiere_bbox, grid_size=500)
    metadata_clean = convert_np(metadata)

    if binary_map is not None and binary_map.size > 0:

        output_filename = os.path.join(".", "CorrectionDispersion/binary_maps_data", f"{target_city.lower().replace(', ', '_').replace(' ', '_')}{'_bbox' if quartiere_bbox is not None else ''}.npy")
        metadata_filename = os.path.join(".", "CorrectionDispersion/binary_maps_data", f"{target_city.lower().replace(', ', '_').replace(' ', '_')}_metadata{'_bbox' if quartiere_bbox is not None else ''}.json")
    
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        np.save(output_filename, binary_map)
        with open(metadata_filename, "w") as file:
            json.dump(metadata_clean, file, indent=4)

        logging.info(f"Binary map saved to '{output_filename}'. Shape: {binary_map.shape}")
        
        x_min, y_min, x_max, y_max = metadata['bounds']
        plt.imshow(binary_map, cmap='gray', extent=(x_min, x_max, y_min, y_max), origin='lower')
   
        building_cells = np.sum(binary_map == 0)
        free_cells = np.sum(binary_map == 1)

        info_text = f"""
            Informazioni Mappa:
            • Griglia: {metadata.get('grid_size', 'N/A')}×{metadata.get('grid_size', 'N/A')}
            • Edifici totali: {metadata.get('total_buildings', 'N/A')}
            • Celle edifici: {building_cells:,}
            • Celle libere: {free_cells:,}
            • CRS: {metadata.get('crs', 'N/A')}
            • Risoluzione: {metadata.get('resolution (m)', 'N/A')} m
            • Densità edifici: {metadata.get('building_density', 'N/A'):.1f}%
            • Altezza media edifici: {metadata.get('mean_height', 'N/A') if metadata.get('mean_height') is not None else 'N/A'} m
            • BBox: {metadata.get('bounds', 'N/A')}
            • Coordinate origine: {metadata.get('origin', 'N/A')}
            • Città: {metadata.get('city', 'N/A')}
            """
        
        plt.figtext(0.02, 0.02, info_text, fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.title(f"Mappa binaria di {target_city} (0 = edificio, 1 = suolo libero)")
        plt.xlabel("Coordinate X (grid)")
        plt.ylabel("Coordinate Y (grid)")
        plt.colorbar(label="Occupazione")
        plt.grid(False)
        plt.show()
    else:
        logging.error("Binary map was not generated successfully.")