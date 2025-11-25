# GMQ710 - Suivi global temporel de feux de forêts
## Objectifs
Le but principal est d’être en mesure de développer un outil capable de produire automatiquement un rapport qui résume des résultats obtenus à partir de l’analyse de données de télédétection concernant les feux de forêts. Le rapport émis devrait aussi être en mesure d’émettre des recommandations quant à la sécurité des gens habitant à proximité des incendies. Plus précisément, on cherchera tout d’abord à acquérir des images MODIS provenant de plusieurs dates subséquentes et survolant la même région atteinte d’un feu (semblable au TD02). Ce satellite a été choisi pour sa résolution temporelle se rapprochant le plus possible d’un suivi en temps réel pour des images accessibles gratuitement. On cherchera ensuite à écrire un script Python étant en mesure de charger ces images satellites et qui effectuera les analyses voulues (semblable au TD02).  Parmi les analyses, on retrouvera la détermination du point spatio-temporelle correspondant à l’origine du feu sélectionné. Puis, on cherchera à déterminer la direction générale et la vitesse de propagation du feu entre chaque itération. Cette évolution servira à estimer la position ultérieure du feu et à émettre des recommandations d’évacuation si un danger est imminent. Le danger sera mesuré en estimant la position actuelle de l’incendie puis en calculant les distances aux routes et lieux habités les plus près. Un graphique de propagation du feu devra aussi être émis pour l’itération actuelle. On y retrouvera aussi le calcul d’indices spectraux comme le NDBI et le NBR (voir annexe) ainsi que de leurs statistiques. Ces indices combinés à un suivi de la température de surface moyenne (accessible par une bande de MODIS) serviront à caractériser l’intensité actuelle du feu et à estimer la superficie qui a été brûlée. Les résultats du point d’origine et de la superficie atteinte seront comparés à des données rendues publiques par le Gouvernement du Québec dans le cas du feu no.344 en 2023 près de la ville de Lebel-sur-Quévillon, soit le feu québécois le plus important de 2023 selon le bilan de la SOPFEU.
## données utilisées
| source                                  | Type        | Format            | Utilité               |
|-----------------------------------------|-------------|-------------------|-----------------------|
| Images Modis (Google Earth Engine) | raster | TIFF | Bonne résolution temporelle (1 jour), images à analyser |
| Données sur les feux de forêt (polygones approx et points d’origine) (Données Québec) | Vecteur | shp | Définir une étendue pour la recherche d’images et validation post-résultats |
| Lieux habités (Données Québec) | Vecteur | Shapefile (géométrie de points) | Calculer les distances aux entités les plus proches pour évaluer le danger |
| Réseau routier du Québec (Données ouvertes Canada) | Vecteur | Shapefile (géométrie de lignes) | Évaluer le danger causé par un accès réduit aux routes d’évacuations |
## Approche / Méthodologie envisagée
Afin de produire un programme permettant de produire un rapport PDF énonçant l'état des lieux concernant des feux actifs au Qc, les étapes suivantes seront réalisées :
1)	Initialisation du script Python et des bibliothèques désirées
2)	Chargement des images à partir de la bibliothèque rasterio (analogue au TD02)
3)	Calculs des indices spectraux et statistiques pour chaque itération temporelle accessible
4)	Graphiques d’évolution temporelles des indices (NDVI/NBR) et de la température (à partir d’un dataframe ‘pandas’)
5)	Classification des zones saines, brûlées et en combustion (à partir des indices et de la température à chaque itération pour un suivi des zones touchées)
6)	Suivi temporel des zones touchées 
7)	Détermination de la direction générale du feu entre 2 itérations/images
8)	Création d’une couche de géométrie point pour la limite du feu dans la direction de propagation, qui sera mise à jour à chaque image
9)	Calcul de la vitesse de déplacement du feu entre chaque itération
10)	Création et mise à jour d’un graphique de propagation du feu (flèche liant les 2 géométries de points)
11)	Animation ? (Background des images + évolution du feu par-dessus, à voir)
12)	Établir des règles concernant les distances au feu nécessitant une évacuation en utilisant geopandas et en calculant les distances séparant le feu des municipalités avoisinantes.
13)	Avertissement lorsque le feu est près des routes d’évacuation
14)	Création du rapport automatisé avec la librairie appropriée 

Préalablement à ces étapes, des images satellites de MODIS d'un feu de forêt documenté seront récupérées et seront les images de référence pour construire le scripte du programme. Le feu ayant eu lieu a Lebel-sur-Quévillon en 2023 (# 344 de 2023 de la SOPFEU) a été choisi pour ce faire.

## Outils et langages prévus :
-	Langage(s) : Python
-	Bibliothèques ou logiciels : rasterio, numpy, matplotlib, pandas, geopandas, fpdf2, fpdf, ReportLab

## Répartition des tâches dans l’équipe
- Daphné Normandeau : Écriture de la logique du code, production de graphiques, rédaction de la documentation
- Charles Raymond : Recherche de données, généralisation et révision du code, peaufinage des graphiques et de la présentation

## Questions à résoudre
- Certaines images de température LST ont plusieurs pixels manquants pour la zone du feu de Lebel-sur-Quévillon. Devrais-t-on quand même utiliser ces données? Si oui, devrait-on garder uniquement les images avec un seuil de pixels existants pour calculer une moyenne indicative ?
- Il y a un décalage géographique entre la couche shapefile d'enveloppe et les rasters téléchargés (projection MODIS sinusoidale), pourquoi?
- Est-ce qu'il est OK de de reprojeter la couche de points suivant l'évolution du feu après leur détermination ? En d'autres mots, pouvons-nous faire les traitements avant d'effectuer la reprojection ?
  

