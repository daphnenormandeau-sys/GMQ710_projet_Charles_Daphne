'''
------------------------------------------------------------------------
Mini-projet du cours GMQ710 (Automne 2025)
Suivi quotidien de l'évolution (vitesse, position globale, surface touchée, etc.) d'incendies de forêt 
------------------------------------------------------------------------
Écrit par: Charles Raymond & Daphné Normandeau
------------------------------------------------------------------------
'''

# Importations de librairies pertinentes
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cycler
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.plot import show
from shapely.geometry import Point, MultiPoint, LineString, Polygon, MultiPolygon
from shapely import convex_hull, buffer, intersects, union, union_all, centroid, distance, difference
from shapely.ops import transform
import pyproj
from scipy.ndimage import label
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import docx2pdf
import cartopy.crs as ccrs
import shapely.wkt
import contextily as cx


############################################################################
# Variables globales du code (certaines pouvant être modifiées)
############################################################################

#Chemin du dossier avec les images MODIS, réso. spatiale de 500 m (bandes b01 = R, b02 = NIR et b07 = SWIR)
# Les images proviennent du jeu de données Google Earth Engine "MODIS/006/MOD09GA"
refl_images_folder = 'Rasters/SUR_REFL_rasters/' #(À MODIFIER SELON LA STRUCTURE DE DOSSIERS)

# Chemin du dossier avec les images  MODIS, réso. spatiale de 1 km (bande de LST)
# Les images proviennent du jeu de données Google Earth Engine "MODIS/061/MOD21A1D"
lst_images_folder = 'Rasters/LST_rasters/' #(À MODIFIER SELON LA STRUCTURE DE DOSSIERS)

# Chemin du dossier contenant les couches SHP contenant les lieux habités de la province 
# de QC (géométrie de points, provenant du site de Données Québec)
analysis_layers_directory = 'Couches/Downloaded_layers/' #(À MODIFIER SELON LA STRUCTURE DE DOSSIERS)
shp_layer_crs = "EPSG:4326" #WGS84

#Chemin pour le template du rapport word
word_template_directory = 'C:/Users/charl/OneDrive/Bureau/Université Sherbrooke/Automne 2025/GMQ710 - Analyse et programmation en géomatique/PYTHONCODES/Mini-projet/'  #(À MODIFIER SELON LA STRUCTURE DE DOSSIERS)

# Seuil de pixels pour lequel on considère la LST comme étant suffisamment représentative de la zone étudiée
correct_LST_treshold = 20 # % # (À MODIFIER AU BESOIN)

# Date d'émission du rapport (fin de la période d'analyse)
date_of_report = '2023-06-06' # (À MODIFIER AU BESOIN)

# Date de la première image (début de la période d'analyse)
start_date = '2023-05-27'

# Valeur minimale de dNBR pour laquelle on considère qu'un pixel est considéré comme étant brûlé
dNBR_fire_treshold = 0.1

# Taille de la fenêtre d'analyse (nb de pixels) pour laquelle on juge qu'un nouveau feu débute
new_fire_window_size = 7 

# Seuil de distance pour laquelle on juge qu'une localité doit être évacuée
warning_treshold = 20000 # m (ou 20 km)

# SCR des images MODIS de réflectance (point de départ si une transformation est nécessaire)
starting_crs = "EPSG:4269" # NAD83

# SCR de prédilection lorsqu'une transformation de coordonnées degrés à métriques est nécessaire 
final_crs = "EPSG:32198" # NAD83 / Quebec Lambert

############################################################################
# Définitions de fonctions utiles à l'analyse
############################################################################

# Fonction permettant de lire et d'enregistrer le Dataset reader rasterio d'une image satellite
def read_satellite_image(file_path): 
    
    return rasterio.open(file_path)

# Fonction permettant de vérifier l'harmonie entre les images satellites (même résolution, projection, dimensions, etc.)
def check_image_harmony(raster_dataset_list, bands_info_string):
    
    reference = raster_dataset_list[0] # Premier dataset, pris comme référence
    
    for dataset in raster_dataset_list[1:]: # On fait une boucle sur les autres datasets
        
        if (dataset.res != reference.res or            # On vérifie la concordance des résolutions
            dataset.crs != reference.crs or            # Pareil pour les projections
            dataset.width != reference.width or        # Pareil pour les largeurs (taille de grille)
            dataset.height != reference.height or      # Pareil pour les hauteurs (taille de grille)
            dataset.count != reference.count or        # Pareil pour le nombre de bandes 
            dataset.bounds != reference.bounds):       # Pareil pour l'emprise et l'alignement
            
            # Dans le cas d'une différence perceptible, on retourne un message d'erreur et on affiche l'image problématique
                print('Oups, les images satellites fournies présentent des différences et doivent être harmonisées!')
                print(dataset)
                sys.exit()
    
    print(f'Bonne nouvelle, les images MODIS de {bands_info_string} sont bien harmonisées!') # Message de succès si toutes les conditions sont respectées


# Fonction permettant d'afficher les métadonnées d'une image satellite
def display_image_metadata(raster_dataset, bands_info_string):
    
    # Stocker les métadonnées 
    width = raster_dataset.width
    height = raster_dataset.height
    nb_bands = raster_dataset.count
    crs = raster_dataset.crs
    res = raster_dataset.res
    bounds = raster_dataset.bounds
    
    metadata = [width, height, nb_bands, crs, res, bounds]
    
    print(f"\nMétadonnées des images satellite de {bands_info_string}:")
    print("-------------------------------")
    print(f"Largeur (pixels): {raster_dataset.width}")
    print(f"Hauteur (pixels): {raster_dataset.height}")
    print(f"Nombre de bandes: {raster_dataset.count}")
    print(f"Système de coordonnées (CRS): {raster_dataset.crs.to_string()}")
    print(f"Résolution (taille des pixels): {raster_dataset.res}")
    print(f"Emprise (bounds): {raster_dataset.bounds}")
    
    return metadata # On retourne les métadonnées sous forme de liste


# Fonction permettant de calculer les indices spectraux NDVI et NBR à partir des images de réflectance
def calculate_spectral_indices(raster_dataset):
    
    # Enregistrement des bandes nécessaires au calcul des indices spectraux
    RED = raster_dataset.read(1)  # Bande Rouge
    NIR = raster_dataset.read(2)  # Bande NIR (Near Infrared)
    SWIR = raster_dataset.read(3)  # Bande SWIR (Shortwave Infrared)
    
    # Dans les images brutes, les pixels sans données sont représentés par des valeurs de 0, on souhaite donc 
    # leur attribuer la valeur de NaN pour éviter les erreurs de calcul lors des divisions
    RED = np.where(RED == 0, np.nan, RED)   
    NIR = np.where(NIR == 0, np.nan, NIR)   
    SWIR = np.where(SWIR == 0, np.nan, SWIR) 
    
    # Calcul des indices spectraux d'intérêt
    NDVI = (NIR - RED) / (NIR + RED) # Normalized Difference Vegetation Index
    NBR = (NIR - SWIR) / (NIR + SWIR) # Normalized Burn Ratio
    
    return NDVI, NBR # On retourne les indices spectraux calculés


# Fonction permettant de calculer les valeurs moyennes, les médianes et les écart-types d'un array numpy
def extract_data_stats(data_array):
    
    mean_list, std_list, median_list = [], [], [] # Initialisation de listes vides qui contiendront les statistiques
    
    for date_data_array in data_array:
        
        nb_nan_pixels = np.count_nonzero(np.isnan(date_data_array)) # Nombre de pixels avec NaN
        total_pixels = date_data_array.size # Nombre total de pixels
        
        if nb_nan_pixels == total_pixels: # Si tous les pixels sont des NaN,
            mean_list.append(np.nan), std_list.append(np.nan), median_list.append(np.nan)
            continue # on ajoute des NaN aux listes de statistiques et on passe à la date suivante
        
        # Calcul de la moyenne, de l'écart-type et de la médiane en ignorant les pixels avec NaN
        mean, std, median = np.nanmean(date_data_array), np.nanstd(date_data_array), np.nanmedian(date_data_array)
        
        # Ajout des valeurs de statistiques à leurs listes respectives
        mean_list.append(mean), std_list.append(std), median_list.append(median)
    
    return  mean_list, std_list, median_list # Retourne les listes de statistiques


# Fonction qui permet de trouver les journées avec plus de (treshold) % de pixels avec LST valide et qui corrige la
# grille de statistique
def correct_LST_stat_values(LST_list, LST_stat_values, treshold):
    
    good_LST_days_indexes = [] # Initialisation de la liste des indexes des journées avec bonne LST
    
    # Boucle sur les journées (nombre de grilles de LST)
    for i in range(len(LST_list)):
        total_pixels = LST_list[i].size # Nombre total de pixels
        non_valid_pixels = np.count_nonzero(np.isnan(LST_list[i])) # Nombre de pixels valides (non-NaN)
        valid_pixels = total_pixels - non_valid_pixels # Nombre de pixels valides
        valid_percentage = (valid_pixels / total_pixels) * 100 # Pourcentage de pixels valides
        
        if valid_percentage >= treshold: # Si le pourcentage de pixels valides est supérieur au seuil
            good_LST_days_indexes.append(i) # On ajoute l'index de la journée à la liste
    
    corrected_LST_stat_values = LST_stat_values.copy() # Copie de la liste des valeurs de LST pour correction
    
    for i in range(len(LST_stat_values)):
        if i not in good_LST_days_indexes: # Si l'index de la journée n'est pas dans la liste des journées avec bonne LST
            corrected_LST_stat_values[i] = np.nan # On remplace la valeur de la statistique par NaN
            
    return corrected_LST_stat_values # On retourne la liste des statistiques corrigées


# Fonction permettant de mettre des statistiques de données (et leur date correspondante) dans un Dataframe 
def generate_stats_dataframe(stat_name, stat_NDVI, stat_NBR, stat_LST, dates_list):
    
    # Création d'un dictionnaire pour stocker les données
    data = {'Date' : dates_list, f'NDVI_{stat_name}' : stat_NDVI, f'NBR_{stat_name}' : stat_NBR, f'LST_{stat_name}' : stat_LST}
    
    # Création du DataFrame pandas à partir du dictionnaire
    stats_df = pd.DataFrame(data)

    # Mettre 'Date' comme index du DataFrame
    stats_df.set_index('Date', inplace=True)
   
    return stats_df # On retourne le DataFrame créé

# Fonction qui permet de créer et sauvegarder les graphiques des données d'un DataFrame
# La fonction produit également un graphique contenant toutes les données (une courbe par colonne du DataFrame)
def plot_save_fig(df):
    
    # Boucle sur les colonnes du DataFrame 
    for column in df.columns:

        # Création du graphique
        x_ticks_locations = np.arange(len(df[column]))           # Liste des emplacement des ticks de l'axe des x
        x_ticks_labels = df.index.strftime('%Y-%m-%d').tolist()  # Nom de chaque tick (date formatée en yyyy-mm-jj)
        plt.plot(x_ticks_locations, df[column], linestyle='--', marker='o', markersize=3, color = 'black') # Écriture du graphique
        
        
        # Il y a trop de dates, et le graphique devient illisible, on va donc n'afficher qu'une date sur deux
        x_ticks_locations_display = [] # Listes vides pour stocker les ticks à afficher
        x_ticks_labels_display = [] # Listes vides pour stocker les labels à afficher
        
        index_to_delete = np.arange(min(x_ticks_locations), max(x_ticks_locations)+1, 2) # Indices des ticks à afficher (une date sur deux)
        
        for index in index_to_delete: # Boucle pour n'ajouter qu'une date sur deux
            x_ticks_locations_display.append(x_ticks_locations[index])
            x_ticks_labels_display.append(x_ticks_labels[index])
            
        plt.xticks(x_ticks_locations_display, x_ticks_labels_display, rotation=45, fontsize=8) # Affichage des ticks et labels de l'axe des x
     
        plt.ylabel(str(column)) # Mettre le nom de l'indice comme titre des ordonnées
        plt.xlabel('Date of MODIS observation')      # Mettre titre de l'axe des abscisses

        # Sauvegarder le graphique dans une image png
        plt.tight_layout() # Permet d'ajuster automatiquement les marges pour éviter que les labels soient coupés
        plt.savefig(f'{str(column)}.png')
        
        # Fermeture de la modification du graphique
        plt.close()
    
    # Produire et sauvegarder le graphique contenant toutes les données des indices spectraux
    plt.rcParams['axes.prop_cycle'] = cycler(color=['green', 'red']) # Choix des couleurs à appliquer aux courbes (vert pour NDVI, rouge pour NBR)
    df_to_plot = df.drop(columns=['LST_Mean']) # On enlève la LST pour ce graphique
    
    plt.plot(x_ticks_locations, df_to_plot, linestyle='--', marker='o', markersize=3) # Écriture du graphique
    plt.xticks(x_ticks_locations_display, x_ticks_labels_display, rotation=45, fontsize=8) # Affichage des ticks et labels de l'axe des x
    plt.xlabel('Date of MODIS observation')      # Mettre titre de l'axe des abscisse
    plt.ylabel('Mean Value')
    plt.legend(df.columns)
    plt.tight_layout() 
    plt.savefig('All_indices.png')
    plt.close()


# Fonction permettant d'identifier les zones entourées d'une fenêtre d'analyse dont les valeurs sont  soient entièrement
# au-dessus d'un seuil spécifique oiu partiellement (géré par l'input condition)
def find_windows_above_threshold(pixel_grid, window_size, threshold, condition):
    
    pixels_around_center = int((window_size-1)/2) # Nombre de pixels devant être de part et d'autre du pixel central 
    row_indexes_central = list(range(pixels_around_center, len(pixel_grid) - pixels_around_center, 1)) # Indices correspondant aux lignes de pixels centraux
    pixel_indexes_central = list(range(pixels_around_center, len(pixel_grid[0]) - pixels_around_center, 1)) # Indices (dans les lignes) correspondant aux pixels centraux
    
    identified_zones = np.full_like(pixel_grid, 0, dtype=float) # Initialisation de la liste qui contiendra les pixels centraux des zones identifiées
    
    # Boucle agissant sur chacun des pixels centraux
    for row_index in row_indexes_central:
        for pixel_index in pixel_indexes_central:
            
            # Déterminer les indexes des pixels dans la fenêtre d'analyse
            window_row_indexes = list(range(row_index - pixels_around_center, row_index + pixels_around_center + 1 )) # Indices des lignes de la fenêtre
            window_pixel_indexes = list(range(pixel_index - pixels_around_center, pixel_index  + pixels_around_center + 1)) # Indices des valeurs dans la fenêtre
            
            # Boucle pour enregistrer les valeurs de tous les pixels de la fenêtre d'analyse
            window_values_list = [] # Initialisation de la liste
            for window_row_index in window_row_indexes:
                for window_pixel_index in window_pixel_indexes:
                    window_values_list.append(dNBR_grid[window_row_index][window_pixel_index]) # On ajoute à la liste
            
            # On veut que tous les pixels de la fenêtre soient au-dessus du seuil
            if condition == 'all_pixels':
                # Si la liste de valeurs dans la fenêtre d'analyse ne contient pas de NaN, on stocke le min et la moyenne de la fenêtre
                if np.count_nonzero(np.isnan(window_values_list)) == 0:
                    window_mean = np.mean(window_values_list) # On trouve la valeur moyenne de la fenêtre
                    window_min = np.min(window_values_list) # On trouve la valeur minimale de la fenêtre
                
                    # Si le min de la fenêtre >= threshold, la zone est identifiée
                    if window_min >= threshold : 
                        identified_zones[window_row_index][window_pixel_index] = window_mean # On stocke la moyenne de la zone dans le pixel central    
            
            # On veut que au moins un pixel de la fenêtre soit au-dessus du seuil            
            if condition == 'any_pixel':
                if any(value >= threshold for value in window_values_list) == True:
                    window_mean = np.mean(window_values_list) # On trouve la valeur moyenne de la fenêtre
                    identified_zones[window_row_index][window_pixel_index] = window_mean # On stocke la moyenne de la zone dans le pixel central         
                       
    return identified_zones # on retourne les zones identifiées


# Fonction qui permet de générer un fichier SHP (avec un nom donné) à partir d'une géométrie choisie pour une date donnée
def generate_geometry_shp(dataframe, chosen_geometry, crs_of_geometry, date_prefix, date):
        
    # Si la géométrie fournit en entrée n'est pas une liste, on fait la conversion
    if type(chosen_geometry) != list:
            chosen_geometry = [chosen_geometry]
        
    # Création du geodataframe panda
    geometry_gdf = gpd.GeoDataFrame(dataframe, geometry = chosen_geometry, crs = crs_of_geometry)
        
    # Création du shapefile à partir du dataframe
    geometry_gdf.to_file(date_prefix + date + '.shp')


# Fonction qui crée des géométries vectorielles (de point et ensuite de polygones) des grilles de zones brûlées
def create_vector_layers(burnt_zones, metadata, date, wanted_output):
    
    # # Trouver et mettre dans une liste les coordonnées du point central de chaque pixel
    # Ce seront les coordonnées des points dans la couche vectorielle
    # Initialiser les listes avec la première coordonnée
    list_lon = [metadata[5][0] + 0.5 * metadata[4][0]] # Longitude du coin supérieur gauche + 1/2 de la résolution de longitude
    list_lat = [metadata[5][3] - 0.5 * metadata[4][1]] # Latitude du coin supérieur gauche - 1/2 de la résolution de latitude
    
    # Créer des listes des longitudes et latitudes de la grille
    for i in range(metadata[0] - 1):
        list_lon.append(list_lon[-1] + metadata[4][0]) # à chaque itération, on ajoute la résolution de longitude
    
    for j in range(metadata[1] - 1):
        list_lat.append(list_lat[-1] - metadata[4][1]) # à chaque itération, on soustrait la résolution de latitude

    # Créer des grilles des coordonnées de longitude et latitude
    lon_grid, lat_grid = np.meshgrid(list_lon, list_lat)
    
    # Initialiser les listes de coordonnées et de dNBR
    lat_burnt_list = []
    lon_burnt_list = []
    dNBR_list = []
    label_list = []

    # Création d'une liste des indexes des points centraux des zones brûlées
    burnt_zones_indexes = np.nonzero(np.nan_to_num(burnt_zones, nan = 0)) # indexes des lignes et colonnes des valeurs non nulles

    # Avant de créer une géométrie de polygone à partir de points, il faut séparer les points selon leur proximité
    # (dans les cas avec des points regroupés, mais séparés par une ceertaine distance) On aura donc plusieurs polygones
    #Remplacement des valeurs NAN par 0 de la grille des zones brûlées
    burnt_zones_with_zeros = np.nan_to_num(burnt_zones, nan=0)
    labeled_array, num_features = label(burnt_zones_with_zeros) #labeled_array : grille avec les labels associées à chaque 
                                                                                # groupe de points collés, num_features : nb de groupes de points
     # On regroupe les paires d'indexes
    for row, column in zip(burnt_zones_indexes[0], burnt_zones_indexes[1]):
        # Pour chaque paires d'indexe, on ajoute la valeur de dNBR et les coordonnées aux listes créées avant la boucle
        dNBR_list.append(burnt_zones[row][column])
        lon_burnt_list.append(lon_grid[row, column])
        lat_burnt_list.append(lat_grid[row, column])
        label_list.append(labeled_array[row][column])
        
    # Créer un dictionnaire 
    dict = {'Lon': lon_burnt_list,
            'Lat': lat_burnt_list,
            'dNBR': dNBR_list,
            'label' : label_list
             }
        
    # On crée un DataFrame à partir du dictionnaire
    point_df = pd.DataFrame(dict)
    
    # Création de la géométrie Point à partir des colonnes "Lat" et "Lon"
    point_geometry = [Point(lon, lat) for lon, lat in zip(point_df['Lon'], point_df['Lat'])]
    
    # Création des GeoDataFrame en spécifiant les géométrie (necessaire pour identifier les polygones)
    point_gdf = gpd.GeoDataFrame(point_df, geometry=point_geometry, crs = metadata[3])
    
    # Boucle pour créer les géométries de polygones
    #Initialisation d'une liste de polygones
    polygons_list = []
    linestring_list = []
    point_list = []
    nb_labels = 0 # compteur pour le nombre de polygones
    for k in range(1, num_features + 1):
        # Entrées du gdf avec les points du groupe ayant le label == k + 1
        point_gdf_groups = point_gdf.loc[point_gdf['label'] == k, ['geometry']]

        # Création de la liste des points du polygone
        polygon_points = [geom for geom in point_gdf_groups['geometry']]
        
        # Création de la géométrie Polygon convexe à partir de la géométrie de points
        multi_geom = convex_hull(MultiPoint(polygon_points))

        # On ajoute la géométrie créée à la liste correspondante
        if isinstance(multi_geom, Point):
            point_list.append(multi_geom)

        if isinstance(multi_geom, LineString):
            linestring_list.append(multi_geom)

        if isinstance(multi_geom, Polygon):
            polygons_list.append(multi_geom)
            nb_labels = nb_labels + 1
    
    # Créer un dictionnaire et un dataframe pour les polygones
    polygon_dict = {'label' : range(1, nb_labels + 1)} 
    polygon_df = pd.DataFrame(polygon_dict)
    
    
    # On écrit les géométries dans des couches shapefile, si désiré
    # (sera uniquement utilisé pour la première date avec un feu)
    if wanted_output == 'create_shp':
        generate_geometry_shp(point_df, point_geometry, metadata[3], 'first_burnt_zones_point_', date)
        generate_geometry_shp(polygon_df, polygons_list, metadata[3], 'first_burnt_zones_polygon_', date)
    
    return polygons_list # On retourne la géométrie de polygone


# Fonction qui permet de combiner les polygones d'une liste qui intersectent un polygone original selon une resolution donnée
def polygon_update(original_polygon, polygons_list, buffer_size):
    
    iterated_polygon = original_polygon # Initialisation du polygone qui sera itéré
    
    # On boucle sur les polygones de la liste
    for polygon in polygons_list:
        
    # Création de buffers autour des polygones candidats et du polygone étudié
        buffered_polygon = buffer(polygon, buffer_size)
        buffered_iterated_polygon = buffer(iterated_polygon, buffer_size)
        
    # Si les polygones avec buffer s'intersectent, un nouveau polygone les combinant est créé
        if intersects(buffered_polygon, buffered_iterated_polygon) == True:
            new_polygon = union(polygon, iterated_polygon) # Union des polygones
            
            # Avant le repassage dans la boucle, le polygone itéré est mis à jour
            iterated_polygon = new_polygon 
        
    return iterated_polygon # On retourne la nouvelle géométrie finale


# Fonction permettant de reprojeter une liste de géométries SHAPELY dans un autre sytème de coordonnées 
def change_of_crs(list_of_geometries, original_crs, goal_crs):
    
    # Définition des systèmes (avant et après) en objets pyproj
    original_crs = pyproj.CRS(original_crs)
    goal_crs = pyproj.CRS(goal_crs)
    
    # On crée le transformeur pyproj (sera appliqué sur une géométrie)
    transformer = pyproj.Transformer.from_crs(original_crs, goal_crs, always_xy = True).transform
    
    # Initialisation de la liste vide qui accueillera les géométries transformées
    transformed_list_of_geometries = []
    
    # Boucle pour transformer les géométries et les ajouter à la liste
    for geometry in list_of_geometries:
        
        # Si l'élément ne contient aucune géométrie,
        if geometry == None:
            transformed_geometry = None # La géométrie demeure inchangée
            transformed_list_of_geometries.append(transformed_geometry) # On ajoute à la liste
            
            continue # On passe à la prochaine itération
        
        # On transforme la géométrie de l'itération actuelle
        transformed_geometry = transform(transformer, geometry)
        
        # On ajoute la géométrie transformée à la liste
        transformed_list_of_geometries.append(transformed_geometry)
        
    return transformed_list_of_geometries # On retourne la liste de géométries transformées

  
# Fonction qui permet de trouver les centroides d'une liste de polygones, 
# le centroide des differences et les distances entre chacuns
def find_difference_centroids_and_distances(polygons_list):
    
    # Initialisation de la list qui contiendra les centroides pour chaque date
    polygons_centroid = []
    
    # Boucle pour remplir la liste des centroides de chaque date
    for polygons in polygons_list:
        current_centroid = centroid(polygons)
        polygons_centroid.append(current_centroid)
    
    # Initialisation des listes de differences de polygones, des centroides de ces differences et de distances
    polygons_difference = [None] # Initialisation des differences de polygones
    polygons_difference_centroid = [None] # Initialisation de la liste des centroides
    centroid_distances = [np.nan] # Initialisation de la liste des distances
    
    # Boucle pour remplir les differentes listes
    for i in range(len(polygons_list) - 1):
        
        # Difference de polygones entre le polygone suivant et le polygone 'i'
        current_difference = difference(polygons_list[i+1], polygons_list[i])
        polygons_difference.append(current_difference)
        
        # Centroide de la difference des polygones (et donc l'expansion du feu)
        current_difference_centroid = centroid(current_difference)
        polygons_difference_centroid.append(current_difference_centroid)
        
        # Distance separant le centroide du polygone 'i' et le centroide de la difference de polygone
        current_distance = distance(polygons_centroid[i], polygons_difference_centroid[i+1])
        centroid_distances.append(current_distance)
        
    return polygons_centroid, polygons_difference_centroid,  centroid_distances


# Fonction permettant de générer un graphique d'évolution de la position du feu (centroides)
def generate_fire_evolution_plot(burnt_polygons_centroid, burnt_polygons_difference_centroid, fire_origin_index, fire_origin_date, dates_list):
    
    # Changement de système de coordonnées pour être en latitude et longitude
    burnt_polygons_centroid = change_of_crs(burnt_polygons_centroid, final_crs, starting_crs)
    burnt_polygons_difference_centroid = change_of_crs(burnt_polygons_difference_centroid, final_crs, starting_crs)
    
    # Initialisation des listes qui contiendront les points (x,y) de l'évolution des positions extremes du feu
    list_of_x = [burnt_polygons_centroid[fire_origin_index].x]
    list_of_y = [burnt_polygons_centroid[fire_origin_index].y]

    # Initialisation de la liste qui contiendra les dates de chacun des points du graphique
    list_of_dates = [fire_origin_date]

    # Boucle sur les indices des listes de polygones brulés
    for i in range (len(burnt_polygons_difference_centroid)):
    
    # On associe l'element a une variable point
        point = burnt_polygons_difference_centroid[i]
    
    # Si aucune géométrie de point (date antérieure au début du feu)
        if point == None:
            continue # On passe à  la prochaine itération
    
    # Si le point est vide (pas de variation entre 2 centroides successifs)
        if point.is_empty == True:
            continue # On passe a la prochaine iteration
    
    # Si le point est non-nul, on ajoute les coordonnées aux listes respectives
        x_coord = point.x
        y_coord = point.y
    
        list_of_x.append(x_coord)
        list_of_y.append(y_coord)
        list_of_dates.append(dates_list[i])

    # Création, personnalisation et sauvegarde du graphique
    plt.plot(list_of_x, list_of_y, linestyle='--', marker='o', markersize=3, color = 'red')
    
    # Plot des points temporel final et d origine
    plt.plot(list_of_x[0], list_of_y[0], marker='*',  markersize = 10, color = 'black')
    plt.plot(list_of_x[-1], list_of_y[-1], marker='X',  markersize = 10, color = 'black')
    
    # Annotation de chaque point avec la date adequate
    for i, date in enumerate(list_of_dates):
        plt.annotate(date, (list_of_x[i], list_of_y[i]), textcoords = "offset points", xytext = (0, 10), ha = 'left', fontsize = 7, fontweight='bold')
    
    # Autres parametres du graphique pour l'affichage
    plt.xlim(min(list_of_x)-0.05, max(list_of_x)+0.05)
    plt.ylim(min(list_of_y)-0.025, max(list_of_y)+0.025)
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.grid()
    
    # Sauvegarde du graphique
    plt.savefig('fire_temporal_evolution.png')
    plt.close()
    

# Fonction calculant la direction générale d'un vecteur reliant 2 points de géométrie shapely
def general_vector_direction(point1, point2):
    
    # On enregistre les valeurs de coordonnées x et y
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    
    # On calcule les cotés du triangle rectangle
    delta_x, delta_y = x2-x1, y2-y1
    
    # On calcule l'angle avec la fonction trigonométrique arctangente
    angle = np.arctan2(delta_y, delta_x) # Angle en radians
    
    # Selon la valeur et le quadrant, on attribue la bonne direction générale
    if angle > np.pi/2:
        direction = "Nord-Ouest"
    elif angle > 0:
        direction = "Nord-Est"
    elif angle > -np.pi/2:
        direction = "Sud-Est"
    else:
        direction = "Sud-Ouest"
        
    return direction


# Fonction qui calcule les distances séparant le polygone de feu aux lieux habités voisins et qui retourne des infos sur la localité la plus proche 
# et les localités situées sous un seuil de danger
def find_closest_inhabited_places(analysis_layers_directory, fire_polygon, crs_of_shp, warning_treshold):
    
    # Lecture de la couche de points des lieux habités dans un geodataframe
    geodataframe = gpd.read_file(f"zip://{analysis_layers_directory}/Lieux_habit_QC.zip")

    # Création d'un array contenant les différentes géométries de la couche
    geometry_array = geodataframe.geometry.array
    
    # Conversion dans le meme systeme metrique que les polygones de feu
    geometry_array = change_of_crs(geometry_array, crs_of_shp, final_crs)

    # Array contenant le type de lieu habité
    area_type_array = geodataframe['typentitto'].to_numpy()

    # Array contenant le nom du lieu habité
    area_name_array = geodataframe['nomcartrou'].to_numpy()
    
    # Arrays contenant les latitudes et longitudes des lieux habités
    longitude_array = geodataframe['longitude'].to_numpy()
    latitude_array = geodataframe['latitude'].to_numpy()
    
    # Initialisation de la plus petite distance entre le feu et une ville
    smallest_distance = 500000
    
    # Initialisation d'une chaîne de caractères sur les localités en danger
    danger_info_string = ''
    
    # Boucle sur les geometries du shapefile
    for i in range(len(geometry_array)):
        
        # Geometrie de point du lieu habite
        point = geometry_array[i]
        
        # Calcul de la distance minimale entre les points et le polygone de feu
        current_distance = distance(point, fire_polygon)
        
        # On verifie si on a un nouveau minimum et si le point designe une Ville
        if current_distance < smallest_distance and area_type_array[i] == 'Ville':
            smallest_distance = current_distance # On met à jour la plus petite distance
            closest_city = area_name_array[i] # La ville la plus près est sauvegardée
            
        # On verifie si le point est en danger, et on sauvegarde les infos pertinentes
        if current_distance < warning_treshold:
            current_string = f'{area_type_array[i]} de {area_name_array[i]} ({round(longitude_array[i],2)}, {round(latitude_array[i],2)}) est à {round(current_distance/1000,2)} km du feu. !! SVP VEUILLEZ ÉVACUER !!\n\n'
            danger_info_string += current_string # On ajoute le string aux infos
    
    # Si aucune localité est sous le seuil d'avertissement, on enregistre un message que tout est ok       
    if len(danger_info_string) == 0:
        danger_info_string = "Il n'y a présentement aucun lieu habité se trouvant en danger imminent. Restez cependant à l'affût de l'évolution du feu dans les jours à suivre!"
    
    return closest_city, danger_info_string
    
            
# Fonction permettant de générer deux images des zones brûlées
def generateImage_firezones(transformed_burnt_polygons_list, metadata_refl, starting_crs, final_crs):
    # Créer un polygone de l'étendue de la zone du feu
    transformer = pyproj.Transformer.from_crs(starting_crs, final_crs, always_xy = True).transform
    # initialiser les bounds
    x0,y0,x1,y1 = [metadata_refl[-1][0], metadata_refl[-1][1], metadata_refl[-1][2], metadata_refl[-1][3]]
    # convertir les bounds en un polygone
    polygon_bounds = shapely.wkt.loads(f'POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))')
    # convertir le polygone dans le crs final
    poly_bounds = transform(transformer, polygon_bounds)

    # on utilise Matplotlib et cartopy
    ax = plt.axes(projection=ccrs.epsg(final_crs[-5:]))
    # on place une image en fond
    cx.add_basemap(ax, crs=ccrs.epsg(final_crs[-5:]), source=cx.providers.OpenStreetMap.Mapnik, zoom = 19)
    # on ajoute une grille
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    # on ajoute la grille de la région du feu
    ax.add_geometries([poly_bounds], crs = ccrs.epsg(final_crs[-5:]), facecolor='none', edgecolor='red', linewidth=1.0)
    # on ajoute les polygones des zones brûlées
    for poly in transformed_burnt_polygons_list:
        if poly is not None:
            if poly.geom_type == 'MultiPolygon': # si c'est un multipolygone
                for geom in poly.geoms:
                    xs, ys = geom.exterior.xy
                    ax.fill(xs, ys, alpha=0.5, fc='orange', ec='none') # Fill the polygon
            elif poly.geom_type == 'Polygon': # si c'est un polygone
                xs, ys = poly.exterior.xy
                plot = ax.fill(xs, ys, alpha=0.5, fc='orange', ec='none')

    # réglage des axes et de la légende 
    p1 = mpatches.Rectangle((0, 0), 1, 1, edgecolor= 'red', facecolor='white',alpha=0.5) #symbol pour le contour de la zone étudiée
    p2 = mpatches.Rectangle((0, 0), 1, 1, color = 'orange',alpha=0.5) #symbol des zones brûlées
    ax.legend(handles = [p1, p2], labels =["Fenêtre\nd'analyse",'Zone brûlée'], loc='center left', bbox_to_anchor=(1.1, 0.9))                       
    ax.set_xlim(left=poly_bounds.bounds[0]-2, right=poly_bounds.bounds[2]+2)
    ax.set_ylim(bottom=poly_bounds.bounds[1]-2, top=poly_bounds.bounds[3]+2)
    # Informations générales de la carte
    ax.text(0.5, -0.2,
        "Système de coordonnées : NAD83/Québec Lambert EPSG : 32198\n" \
        "Source des données : MODIS\n" \
        "Auteurs : Charles Raymond et Daphné Normandeau",
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5)) # Optional text box
    #Agrandir l'étendue de l'image pour que le texte ne soit pas coupé
    plt.subplots_adjust(bottom=0.3, left = 0.001) 
    # sauvegarder et fermer le graphique
    plt.savefig('fire_zones.png')
    plt.close()    


# Fonction permettant de créer une carte GeoTIFF pour les indices spectraux calculés (NDVI et NBR)
# et le graphique associé
def create_geotiff_map(NDVI, NBR, dNBR, dNDVI, date_of_report, metadata_refl):
    # On convertit les grilles en type de données float32 pour l'écriture GeoTIFF
    NDVI = NDVI.astype(np.float32)
    NBR = NBR.astype(np.float32)
    dNDVI = dNDVI.astype(np.float32)
    dNBR = dNBR.astype(np.float32)

    # géoréférencement des images
    west, south, east, north = metadata_refl[5]
    width, height = metadata_refl[0], metadata_refl[1]
    transform = from_bounds(west, south, east, north, width, height)
    #TODO boucle
    # Définir les métadonnées
    profile = {
        'driver': 'GTiff',
        'dtype': np.float32,
        'count': 1,  # Nombre de bandes
        'width': NDVI.shape[1],
        'height': NDVI.shape[0],
        'crs': metadata_refl[3], 
        'transform': transform 
    }

    # Création et écriture des fichiers Géotiff
    with rasterio.open('NDVI_' + str(date_of_report) + '.tif', 'w', **profile) as dst:
        dst.write(NDVI, 1) 
    #générer un graphique du raster
    src = rasterio.open('NDVI_' + str(date_of_report) + '.tif')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(src.read(1), cmap='RdYlGn')
    plt.colorbar(fraction=0.025)
    plt.savefig('NDVI_' + str(date_of_report))
    plt.close()    

    with rasterio.open('NBR_' + str(date_of_report) + '.tif', 'w', **profile) as dst:
        dst.write(NBR, 1) 
    #générer un graphique du raster
    src = rasterio.open('NBR_' + str(date_of_report) + '.tif')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(src.read(1), cmap='PuOr')
    plt.colorbar(fraction=0.025)
    plt.savefig('NBR_' + str(date_of_report))
    plt.close() 
    
    with rasterio.open('dNDVI_' + str(date_of_report) + '.tif', 'w', **profile) as dst:
       dst.write(dNDVI, 1) 
    #générer un graphique du raster
    src = rasterio.open('dNDVI_' + str(date_of_report) + '.tif')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(src.read(1), cmap='RdYlGn_r')
    plt.colorbar(fraction=0.025)
    plt.savefig('dNDVI_' + str(date_of_report))
    plt.close()

    with rasterio.open('dNBR_' + str(date_of_report) + '.tif', 'w', **profile) as dst:
        dst.write(dNBR, 1)
    #générer un graphique du raster
    src = rasterio.open('dNBR_' + str(date_of_report) + '.tif')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(src.read(1), cmap='Spectral_r')
    plt.colorbar(fraction=0.025)
    plt.savefig('dNBR_' + str(date_of_report))
    plt.close() 
    

# Fonction qui génère un rapport docx selon des variables définies
def generate_report(word_template_directory, start_fire_date, date, city, dir_fire, fire_speed, warning_message, burnt_area, burnt_area_total, first_image_date, LST_treshold):
        # on lit le modèle avec les balises
        doc = DocxTemplate(word_template_directory + 'template_word_proj.docx')
        # on envoie les infos
        context = {
            'start_fire_date': start_fire_date, 'date': date, 'city': city, 
            'dir_fire': dir_fire, 'fire_speed': round(fire_speed,2),
            'warning_message' : warning_message, 'burnt_area' : burnt_area,
            'burnt_area_total' : burnt_area_total, 'first_image_date' : first_image_date,
            'LST_treshold': LST_treshold
        }
        
        # on ajoute l'image des polygones de feux
        context['fire_zones_image'] = InlineImage(doc, 'fire_zones.png', width=Mm(80))
        # on ajoute le graphique qui montre l'évolution de la localisation du feu
        context['fire_evolution_image'] = InlineImage(doc, 'fire_temporal_evolution.png', width=Mm(80))
        # on ajoute l'image du NDVI
        context['NDVI_today'] = InlineImage(doc, 'NDVI_' + str(date) + '.png', width=Mm(80))  
        # on ajoute l'image du dNDVI
        context['dNDVI'] = InlineImage(doc, 'dNDVI_' + str(date) + '.png', width=Mm(80)) 
        # on ajoute l'image du NBR
        context['NBR_today'] = InlineImage(doc, 'NBR_' + str(date) + '.png', width=Mm(80)) 
        # on ajoute l'image du dNBR
        context['dNBR'] = InlineImage(doc, 'dNBR_' + str(date) + '.png', width=Mm(80))  
        # on ajoute le graphique d'évolution des moyennes des indices
        context['graph_mean_indices'] = InlineImage(doc,'All_indices.png', width=Mm(80))
        # on ajoute le graphique d'évolution de la moyenne de température
        context['graph_mean_LST'] = InlineImage(doc,'LST_Mean.png', width=Mm(80))

        doc.render(context)
        # on génère le fichier pour une donnée
        doc.save('report_' + str(date) + '.docx')


############################################################################
# EXÉCUTIONS PRINCIPALES DU SCRIPT
############################################################################

############################################################################
# PARTIE 1: LECTURE ET VÉRIFICATION DES IMAGES SATELLITES
############################################################################

# Génération d'une liste des dates des images MODIS à analyser
# On a des images datant du 27 mai au 30 juin 2023
start_date = np.datetime64(start_date) # Conversion de la première date en format datetime64
date_of_report = np.datetime64(date_of_report)  # Conversion de l'image de la date finale en format datetime64
dates_list = np.arange(start_date, date_of_report + np.timedelta64(1, 'D'))

# Changement du format des dates pour correspondre aux noms des fichiers
dates_list_for_search = [str(x).replace("-","_") for x in dates_list]

# Initialisation de listes vides pour stocker les objets Dataset reader rasterio
refl_raster_dataset_list = []
lst_raster_dataset_list = []

# Enregistrement des objets Dataset reader rasterio pour chaque image satellite dans les listes respectives
for date in dates_list_for_search:
    refl_raster_dataset_list.append(read_satellite_image(f'{refl_images_folder}MODIS_B1_B2_B7_AOI_{date}.tif'))
    lst_raster_dataset_list.append(read_satellite_image(f'{lst_images_folder}MODIS_LST_AOI_{date}.tif'))

# Vérification de l'harmonie entre les images satellites (résolution, projection, dimensions, etc.)
print('\nVérification de l\'harmonie entre les images satellites...\n')
check_image_harmony(refl_raster_dataset_list, "RÉFLECTANCE")
check_image_harmony(lst_raster_dataset_list, "TEMPÉRATURE DE SURFACE")

# Affichage des métadonnées des images satellites (réflectance et LST)
metadata_refl = display_image_metadata(refl_raster_dataset_list[0], "RÉFLECTANCE") # Affichage des métadonnées des images de réflectance")
metadata_lst = display_image_metadata(lst_raster_dataset_list[0], "TEMPÉRATURE DE SURFACE") # Affichage des métadonnées des images de LST")


############################################################################
# PARTIE 2: ENREGISTREMENT DES DONNÉES (INDICES SPECTRAUX ET VALEURS DE LST)
############################################################################

# Listes vides pour stocker les indices spectraux calculés et données de LST
NDVI_list, NBR_list, LST_list = [], [], []

# Boucle sur chaque dataset de réflectance pour calculer les indices spectraux
for refl_raster_dataset in refl_raster_dataset_list:
    NDVI, NBR = calculate_spectral_indices(refl_raster_dataset) # Calcul des indices spectraux
    NDVI_list.append(NDVI), NBR_list.append(NBR) # Ajout des indices aux listes correspondantes

# Boucle sur chaque dataset de LST pour enregistrer la LST
for lst_raster_dataset in lst_raster_dataset_list:
    LST = lst_raster_dataset.read(1)  # Lecture de la bande de LST
    LST = LST - 273.15 # Conversion de la LST de Kelvin à Celsius
    LST_list.append(LST) # Ajout des données de LST à la liste correspondante


############################################################################
# PARTIE 3: STATISTIQUES D'INDICES SPECTRAUX ET LST DANS LA ZONE D'INTÉRÊT
############################################################################

# Extraction des valeurs moyennes des indices spectraux pour chaque date
NDVI_mean_values, _, _ = extract_data_stats(NDVI_list)
NBR_mean_values, _, _ = extract_data_stats(NBR_list)
LST_mean_values, _, _ = extract_data_stats(LST_list)

# Puisque la LST moyenne n'est pas toujours représentative de la région (LST du produit MODIS non corrigée pour la couverture nuageuse),
# il faut choisir les journées où la LST moyenne est pertinente (+ de 20% des pixels ont une valeur valide)
corrected_LST_mean_values = correct_LST_stat_values(LST_list, LST_mean_values, correct_LST_treshold)

# Stockage des valeurs moyennes dans un DataFrame pandas
mean_df = generate_stats_dataframe('Mean', NDVI_mean_values, NBR_mean_values, corrected_LST_mean_values, dates_list)
#print(mean_df.head()) # Affichage des premières lignes du DataFrame des moyennes

# Créer et sauvegarder les graphiques d'évolution temporelle des indices et de la LST
plot_save_fig(mean_df)


############################################################################
# PARTIE 4: DÉTECTION ET CARTOGRAPHIE DES ZONES BRÛLÉES
############################################################################
# Dans cette partie du projet, on cherche à identifier les zones considérées comme étant brûlées en calculant la différence de l'indice NBR
# entre une image avant feu (ici, on prendra la première image disponible) et l'image d'une date ultérieure

# On initialise une liste qui contiendra les polygones de zones brûlées (il y a déjà une valeur vide car on commence les calculs de dNBR à la seconde date)
# (on ne peut pas calculer de différence sans avoir une date antérieure)
burnt_polygons_list = [None]

# On cherche à trouver le point d'origine du feu en analysant les grilles de dNBR entre la première date et les dates suivantes
for i in range(len(NBR_list) - 1):
    dNBR_grid = NBR_list[0] - NBR_list[i+1] # Calcul de la grille dNBR entre la première date et la date étudiée
    
    # Nous considérerons qu'une variation de dNBR supérieure à 'dNBR_fire_treshold' constitue une zone brûlée
    # Or, puisque nous cherchons un point de départ du feu et que nous souhaitons éviter les fausses détections,
    # nous chercherons une fenêtre d'analyse carrée assez grande de côté 'new_fire_window_size' pixels 
    # dans laquelle tous les pixels sont considérées brûlés (valeurs supérieures au seuil)
    fire_origin_grid = find_windows_above_threshold(dNBR_grid, new_fire_window_size, dNBR_fire_treshold, 'all_pixels') # on calcule avec le dNBR, tous les pixels doivent être au-dessus du seuil
    
    if np.sum(fire_origin_grid) != 0: # et donc si on a trouvé des fenêtres totalement brûlées
        
        fire_origin_date = dates_list_for_search[i + 1] # Date du début de feu
        fire_origin_index = i+1                         # Index du début de feu
        
        # On crée les couches vectorielles de l'origine du feu et on enregistre sa géométrie
        fire_origin_polygon = create_vector_layers(fire_origin_grid, metadata_refl, fire_origin_date, 'create_shp') 
                                                                                                                    
        print(f'\nIl semble que le feu de forêt ait débuté le : {fire_origin_date}!') # On affiche la date de l'origine du feu
        
        # On ajoute le polygone à la liste de polygones brûlés
        burnt_polygons_list.append(*fire_origin_polygon)
        
        break # On arrête la boucle après avoir trouvé le premier point d'origine du feu
    
    # Si la somme est vide et donc qu'on a pas de feu, on ajoute une géométrie vide pour la date en cours
    else:
        burnt_polygons_list.append(None)

# Une fois le polygone du tout début du feu trouvé, on cartographie les zones brûlées pour toutes les dates suivantes

# On initialise les zones brûlées avec le polygone d'origine du feu. On le transforme tout de suite en objet MultiPolygon pour pouvoir
# ajouter des nouveaux polygones facilement au cours des itérations
burnt_polygons = MultiPolygon(fire_origin_polygon)

# On fait une boucle sur toutes les dates suivant la première apparition d'un feu
for i in np.arange(fire_origin_index, len(NBR_list) - 1):
    
    # Message de suivi dans la console
    print(f"\nAnalyse de l'image du {dates_list_for_search[i + 1]} en cours ...")
    
    dNBR_grid = NBR_list[0] - NBR_list[i+1] # Calcul de la grille dNBR entre la première date et la date étudiée
    dNDVI_grid = NDVI_list[0] - NDVI_list[i+1] # Calcul de la grille dNDVI entre la première date et la date étudiée
    
    # On vérifie si des nouveaux polygones pourraient être des nouveaux points d'origine
    possible_new_fire_origin = find_windows_above_threshold(dNBR_grid, new_fire_window_size, dNBR_fire_treshold, 'all_pixels')
    
    # Si on trouve un nouveau point d'origine de feu
    if np.sum(possible_new_fire_origin) != 0:
        # On crée ces nouveaux polygones de début de feu
        new_fire_origin_polygons = create_vector_layers(possible_new_fire_origin, metadata_refl, dates_list_for_search[i + 1], 'only_polygons')
        burnt_polygons = union_all([burnt_polygons, *new_fire_origin_polygons]) # On ajoute ces polygones à la zone brûlée en cours
    
    
    # Par la suite, on étudie des plus petites zones étant des continuations de feux..
    # On veut que le dNBR soit suffisamment grand et touche la zone identifiée comme deja brulee
    possible_burnt_zones = find_windows_above_threshold(dNBR_grid, 3, dNBR_fire_treshold, 'all_pixels') # On calcule avec le dNBR les candidats de zone brulee
    
    # On polygonise les zones potentiellement brûlées.
    possible_polygons_list = create_vector_layers(possible_burnt_zones, metadata_refl, dates_list_for_search[i + 1], 'only_polygons') 
    
    # On doit maintenant vérifier si ces polygones intersectent les zones brûlées et ne sont pas du bruit
    
    updated_polygon_list = [] # On initie une liste vide qui accueillera des nouveaux polygones mis à jour
    
    # On fait une boucle sur les polygones considérés comme étant déjà brûlés
    for burnt_polygon in burnt_polygons.geoms:
        # On met à jour les polygones déjà brûlés avec les nouvelles zones brûlées adjacentes
        updated_polygon = polygon_update(burnt_polygon, possible_polygons_list, metadata_refl[4][0])
        updated_polygon_list.append(updated_polygon)
    
    # On unifie les polygones qui étaient déjà brûlés avec les nouveaux. Ces polygones serviront de point de départ
    # pour la prochaine itération
    burnt_polygons = union_all([burnt_polygons, *updated_polygon_list])
    
    # On crée une couche shapefile des polygones pour la date en cours
    generate_geometry_shp(None, burnt_polygons, starting_crs, 'updated_burnt_zones_', dates_list_for_search[i + 1])

    # On ajoute les polygones finaux à la liste temporelle de géométries
    burnt_polygons_list.append(burnt_polygons)

print('\nTOUTES LES IMAGES ONT ÉTÉ ANALYSÉES AVEC SUCCÈS!')
print('================================================')

############################################################################
# PARTIE 5: CALCULS ET ANALYSES SUR LES ZONES BRÛLÉES
############################################################################ 
    
# On a calculé la nouvelle zone brûlée pour toutes les dates
# Nous allons maintenant calculer la direction de déplacement du feu et sa vitesse
# (en comparant deux itérations subséquentes)

# On transforme les points dans une projection métrique adaptée à la zone d'intérêt
transformed_burnt_polygons_list = change_of_crs(burnt_polygons_list, starting_crs, final_crs)

# Calcul des centroides de l'ensemble des polygones brulés pour chaque date, des centroides des nouvelles zones brulées pour l'itération en cours, et de la distance entre ces 2 ensembles
burnt_polygons_centroid, burnt_polygons_difference_centroid, centroid_distances = find_difference_centroids_and_distances(transformed_burnt_polygons_list)

# Calcul de la vitesse générale du feu 
fire_general_speed_ms = np.array(centroid_distances)/86400   # en m/s
fire_general_speed_kmh = fire_general_speed_ms*3600/1000     # en km/h

# Création du graphique d'évolution de la position globale du feu
generate_fire_evolution_plot(burnt_polygons_centroid, burnt_polygons_difference_centroid, fire_origin_index, fire_origin_date, dates_list_for_search)

# On calcule la direction générale du feu à la dernière itération
fire_direction = general_vector_direction(burnt_polygons_centroid[-1], burnt_polygons_difference_centroid[-1])

# calcul de la superficie des nouvelles zones brûlées
burnt_area_today = transformed_burnt_polygons_list[-1].area - transformed_burnt_polygons_list[-2].area #superficie brûlée dans la journée du rapport
burnt_area_today = round(burnt_area_today / 1000000, 2) # conversion en km^2
total_burnt_area = transformed_burnt_polygons_list[-1].area # superficie totale brûlée depuis le début du feu
total_burnt_area = round(total_burnt_area / 1000000, 2) # conversion en km^2

# On sauvegarde la ville la plus proche ainsi que les infos des localités à risque!
closest_city, danger_info_string = find_closest_inhabited_places(analysis_layers_directory, transformed_burnt_polygons_list[-1], shp_layer_crs, warning_treshold)

# On génère une image des polygones des zones brûlées géoréférencées
# Cette image sera appelée directement dans la fonction qui remplit le template
generateImage_firezones(transformed_burnt_polygons_list, metadata_refl, starting_crs, final_crs)

# On génère des images GeoTIFF des indices NDVI et NBR pour la date en cours, + les dNDVI et dNBR entre la première et dernière date
# Ces images seront appelée directement dans la fonction qui remplit le template
create_geotiff_map(NDVI, NBR, dNBR_grid, dNDVI_grid, date_of_report, metadata_refl)


############################################################################
# PARTIE 6: GÉNÉRATION DU RAPPORT QUOTIDIEN POUR LA DATE 'date_of_report'
############################################################################
# Dans cette section, des variables sont définies et sont insérées dans leurs balises respectives du template word 
# 'template_word_proj.docx' dont le chemin est spécifié dans les variables globales en début de script

# Ces nouvelles définitions ne font que rassembler au même endroit toutes les balises du template à remplir
start_fire_date = fire_origin_date      # Date à laquelle le feu a officiellement commencé
date = date_of_report                   # Date à laquelle le rapport est émis
city = closest_city                     # Ville la plus rapprochée du feu de forêt
dir_fire = fire_direction               # Direction générale du feu
fire_speed = fire_general_speed_kmh[-1] # Vitesse du feu à la dernière itération
warning_message = danger_info_string    # Message d'avertissement en cas de danger
first_image_date = start_date           # Date de la première image étudiée (pré-feu, ayant servi aux calculs)
burnt_area = burnt_area_today           # Superficie brûlée dans les dernières 24 heures
burnt_area_total = total_burnt_area     # Superficie totale brûlée depuis le début du feu
LST_treshold = correct_LST_treshold     # Seuil de pixels pour lesquel la LST est valide

# Génération du rapport quotidien sous format docx présentant l'état du feu
generate_report(word_template_directory, start_fire_date, date, city, dir_fire, fire_speed, warning_message, burnt_area, burnt_area_total, first_image_date, LST_treshold)

# Conversion du rapport généré en format PDF pour un rendu plus professionnel
docx2pdf.convert(f"report_{date}.docx", f"report_{date}.pdf")

# Impression d'un message de fin
print("\nRapport généré avec succès!\nFin de l'exécution du programme ...")
