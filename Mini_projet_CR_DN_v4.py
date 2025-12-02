'''
------------------------------------------------------------------------
Mini-projet du cours GMQ710 (Automne 2025)
Suivi d'évolution d'incendies de forêt et de leur impact environnemental
------------------------------------------------------------------------
Écrit par: Charles Raymond & Daphné Normandeau
------------------------------------------------------------------------
'''

# Importations de librairies pertinentes
import sys
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from matplotlib import cycler


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
        
        # Calcul de la moyenne, de l'écart-type et de la médiane en ignorant les pixels avec NaN
        mean, std, median = np.nanmean(date_data_array), np.nanstd(date_data_array), np.nanmedian(date_data_array)
        
        # Ajout des valeurs de statistiques à leurs listes respectives
        mean_list.append(mean), std_list.append(std), median_list.append(median)
    
    return  mean_list, std_list, median_list # Retourne les listes de statistiques


# Fonction permettant de mettre des statistiques de données (et leur date correspondante) dans un Dataframe 
def generate_stats_dataframe(stat_name, stat_NDVI, stat_NBR, dates_list):
    
    # Création d'un dictionnaire pour stocker les données
    data = {'Date' : dates_list, f'NDVI_{stat_name}' : stat_NDVI, f'NBR_{stat_name}' : stat_NBR}
    
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
    
    # Produire et sauvegarder le graphique contenant toutes les données
    
    plt.rcParams['axes.prop_cycle'] = cycler(color=['green', 'red']) # Choix des couleurs à appliquer aux courbes (vert pour NDVI, rouge pour NBR)
    
    plt.plot(x_ticks_locations, df, linestyle='--', marker='o', markersize=3) # Écriture du graphique
    plt.xticks(x_ticks_locations_display, x_ticks_labels_display, rotation=45, fontsize=8) # Affichage des ticks et labels de l'axe des x
    plt.xlabel('Date of MODIS observation')      # Mettre titre de l'axe des abscisse
    plt.ylabel('Mean Value')
    plt.legend(df.columns)
    plt.tight_layout() 
    plt.savefig('All_indices.png')


#TODO fonction pour trouver le pixel central/trouver l'origine du feu et à quel image
#commencer par gros cadran et raffiner
#point origine feu et point avec la plus grande diff : flèche direction

# Fonction permettant d'identifier les zones brûlées dans la grille de pixels selon une fenêtre d'analyse et un seuil spécifique
def find_burnt_zone(dNBR_grid, window_size, threshold):
    
    pixels_around_center = int((window_size-1)/2) # Nombre de pixels devant être de part et d'autre du pixel central (carré de côté 'pixels_around_around_center' autour du pixel central)
    row_indexes_central = list(range(pixels_around_center, len(dNBR_grid) - pixels_around_center, 1)) # Indices correspondant aux lignes de pixels centraux
    pixel_indexes_central = list(range(pixels_around_center, len(dNBR_grid[0]) - pixels_around_center, 1)) # Indices (dans les lignes) correspondant aux pixels centraux
    
    burnt_zones = np.full_like(dNBR_grid, 0, dtype=float) # Initialisation de la liste qui contiendra les dNBR des zones brûlées
                            
    
    max_dNBR = 0 # Initialisation du maximum du dNBR
    index_max_dNBR = [0 , 0] # Initialisation de la position du pixel central du maximum du dNBR
    
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
                    if not m.isnan(dNBR_grid[window_row_index][window_pixel_index]): #si le pixel est non-nul,
                        window_values_list.append(dNBR_grid[window_row_index][window_pixel_index]) # On ajoute à la liste
            
            # Si la fenêtre d'analyse a des valeurs non-nulles, on stocke le min et la moyenne du dNBR
            if window_values_list:
                window_mean = np.mean(window_values_list) # On trouve la valeur moyenne de la fenêtre
                window_min = np.min(window_values_list) # On trouve la valeur minimale de la fenêtre
            
                # Si le dNBR_min de la fenêtre >= threshold, la zone est brûlée
                if window_min >= threshold : 
                    burnt_zones[window_row_index][window_pixel_index] = window_mean # si le min dNBR dépasse le seuil,
                                                                                     # la zone est brûlée et on enregistre la valeur moyenne
                                                                                     # pour le pixel central
                    if window_mean > max_dNBR:
                        max_dNBR = window_mean
                        index_max_dNBR[0] = row_index
                        index_max_dNBR[1] = pixel_index
                    
    return burnt_zones, [max_dNBR, index_max_dNBR] # on retourne les zones brûlées et le point central de la zone la plus intense du feu   

# Fonction qui crée une grille dans laquelle chaque élément
# correspond à un pixel et contient les coordonnées du pixel
# et sa température
def create_point_layer(burnt_zones, metadata, date):
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
    
    # Création d'une liste des indexes des points centraux des zones brûlées
    burnt_zones_indexes = np.nonzero(burnt_zones) # indexes des lignes et colonnes des valeurs non nulles

     # On regroupe les paires d'indexes
    for row, column in zip(burnt_zones_indexes[0], burnt_zones_indexes[1]):
        # Pour chaque paires d'indexe, on ajoute l'année de changement et les coordonnées aux listes créées avant la boucle
        dNBR_list.append(burnt_zones[row][column])
        lon_burnt_list.append(lon_grid[row, column])
        lat_burnt_list.append(lat_grid[row, column])
        

    # Créer un dictionnaire 
    dict = {'Lon': lon_burnt_list,
            'Lat': lat_burnt_list,
            'dNBR': dNBR_list,
             }
        
    # On crée un DataFrame à partir du dictionnaire
    df = pd.DataFrame(dict)
    
    # Création de la géométrie Point à partir des colonnes "Lat" et "Lon"
    geometry = [Point(lon, lat) for lon, lat in zip(df['Lon'], df['Lat'])]

    # Création du GeoDataFrame en spécifiant la géométrie
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs = metadata[3])
    
    # Création d'un shapefile à partir du Dataframe
    gdf.to_file('burnt_zones_' + date + '.shp')
        
# Fonction qui permet de trouver la direction générale d'un feu 
# à l'aide de données de température du sol
def fire_direction(lst_raster_dataset_list):
    # Déterminer la différence de température entre la dernière et l'avant-dernière journée
    diff_temp_raster = lst_raster_dataset_list[-1] - lst_raster_dataset_list[-2]

    # Création de listes de différence de température
    # selon les 4 cadrans de base (N,S,E,O)


############################################################################
# EXÉCUTIONS PRINCIPALES DU SCRIPT
############################################################################

############################################################################
# PARTIE 1: LECTURE ET VÉRIFICATION DES IMAGES SATELLITES
############################################################################

# Chemin du dossier avec les images MODIS, réso. spatiale de 500 m (bandes b01 = R, b02 = NIR et b07 = SWIR)
# Les images proviennent du jeu de données Google Eart Engine "MODIS/006/MOD09GA"
refl_images_folder = 'Rasters/SUR_REFL_rasters/' #(À MODIFIER SELON LA STRUCTURE DE DOSSIERS)

# Chemin du dossier avec les images  MODIS, réso. spatiale de 1 km (bande de LST)
# Les images proviennent du jeu de données Google Eart Engine "MODIS/061/MOD21A1D"
lst_images_folder = 'Rasters/LST_rasters/' #(À MODIFIER SELON LA STRUCTURE DE DOSSIERS)

# Génération d'une liste des dates des images MODIS à analyser
dates_list = np.arange('2023-05-27', '2023-06-30', dtype='datetime64[D]')

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
    LST_list.append(LST) # Ajout des données de LST à la liste correspondante

############################################################################
# PARTIE 3: STATISTIQUES D'INDICES SPECTRAUX ET LST DANS LA ZONE D'INTÉRÊT
############################################################################

# Extraction des valeurs moyennes des indices spectraux pour chaque date
NDVI_mean_values, _, _ = extract_data_stats(NDVI_list)
NBR_mean_values, _, _ = extract_data_stats(NBR_list)

#TODO Il faut trouver les journées pour lesquelles la moyenne est pas représentative...
LST_mean_values, _, _ = extract_data_stats(LST_list)


# Stockage des valeurs moyennes dans un DataFrame pandas
mean_df = generate_stats_dataframe('Mean', NDVI_mean_values, NBR_mean_values, dates_list)
print(mean_df.head()) # Affichage des premières lignes du DataFrame des moyennes

# Créer et sauvegarder les graphiques d'évolution temporelle des indices
plot_save_fig(mean_df)

# Calcul et stockage des zones brûlées 
#for i in range(len(NBR_list) - 1):
#    burnt_zones, fire_zone = find_burnt_zone(NBR_list[i + 1] - NBR_list[i], 3, 0.1) # on calcule avec le dNBR
#    create_point_layer(burnt_zones, metadata, dates_list_for_search[i + 1])


