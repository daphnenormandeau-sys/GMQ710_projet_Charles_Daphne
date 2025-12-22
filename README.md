# GMQ710 - Suivi global temporel de feux de forêts
## Objectifs
Le but principal est d’être en mesure de développer un outil capable de produire automatiquement un rapport qui résume des résultats obtenus à partir de l’analyse de données de télédétection concernant les feux de forêts. Le rapport émis devrait aussi être en mesure d’émettre des recommandations quant à la sécurité des gens habitant à proximité des incendies. Plus précisément, on cherchera tout d’abord à acquérir des images MODIS provenant de plusieurs dates subséquentes et survolant la même région atteinte d’un feu. Ce satellite a été choisi pour sa résolution temporelle se rapprochant le plus possible d’un suivi en temps réel pour des images accessibles gratuitement. On cherchera ensuite à écrire un script Python étant en mesure de charger ces images satellites et qui effectuera les analyses voulues.  Parmi les analyses, on retrouvera la détermination la zone spatiale correspondant à l’origine du feu sélectionné. Puis, on cherchera à déterminer la direction générale et la vitesse de propagation du feu entre chaque itération. Cette évolution servira à estimer la position ultérieure du feu et à émettre des recommandations d’évacuation si un danger est imminent. Le danger sera mesuré en estimant la position actuelle de l’incendie puis en calculant les distances aux lieux habités les plus près. Un graphique de propagation du feu devra aussi être émis pour l’itération actuelle. On y retrouvera aussi le calcul d’indices spectraux comme le NDVI et le NBR (voir annexe) ainsi que de leurs statistiques. Ces indices combinés à un suivi de la température de surface moyenne (accessible par une bande de MODIS) serviront à caractériser l’intensité actuelle du feu et à estimer la superficie qui a été brûlée. Les résultats du point d’origine et de la superficie atteinte seront comparés à des données rendues publiques par le Gouvernement du Québec dans le cas du feu no.344 en 2023 près de la ville de Lebel-sur-Quévillon, soit le feu québécois le plus important de 2023 selon le bilan de la SOPFEU.
## Données utilisées
| source                                  | Type        | Format            | Utilité               |
|-----------------------------------------|-------------|-------------------|-----------------------|
| Images Modis (Google Earth Engine) | raster | TIFF | Bonne résolution temporelle (1 jour), images à analyser |
| Données sur les feux de forêt (polygones approx et points d’origine) (Données Québec) | Vecteur | shp | Définir une étendue pour la recherche d’images et validation post-résultats |
| Lieux habités (Données Québec) | Vecteur | Shapefile (géométrie de points) | Calculer les distances aux entités les plus proches pour évaluer le danger |

Lien Google Drive afin de télécharger ces différentes données: https://drive.google.com/drive/folders/1PEE9J_BTLDXJDvl5_iXUxcJSjebqVfPQ?usp=sharing

## Approche / Méthodologie envisagée
Afin de produire un programme permettant de produire un rapport PDF énonçant l'état des lieux concernant des feux actifs au Qc, les étapes suivantes seront réalisées :
1)	Initialisation du script Python et des bibliothèques désirées
2)	Chargement des images à partir de la bibliothèque rasterio (analogue au TD02)
3)	Calculs des indices spectraux et statistiques pour chaque itération temporelle accessible
4)	Graphiques d’évolution temporelles des indices (NDVI/NBR) et de la température (à partir d’un dataframe ‘pandas’)
5)	Classification des zones brûlées (à partir du dNBR à chaque itération pour un suivi des zones touchées)
6)	Suivi temporel des zones touchées (fenêtre d'analyse à chaque itération, polygonisation itérative de la zone brûlée)
7)	Détermination de la direction générale du feu entre 2 itérations/images à partir des poygones (distances entre centroides)
8)	Création d’un graphique de points pour la limite du feu dans la direction de propagation, qui sera mise à jour à chaque image
9)	Calcul de la vitesse de déplacement du feu entre chaque itération
10)	Création et mise à jour d’un graphique de propagation du feu 
11)	Établir des règles concernant les distances au feu nécessitant une évacuation en utilisant geopandas et en calculant les distances séparant le feu des municipalités avoisinantes.
12)	Création du rapport automatisé avec la librairie appropriée (docxtpl et docx2pdf)

Préalablement à ces étapes, des images satellites de MODIS d'un feu de forêt documenté seront récupérées et seront les images de référence pour construire le scripte du programme. Le feu ayant eu lieu a Lebel-sur-Quévillon en 2023 (# 344 de 2023 de la SOPFEU) a été choisi pour ce faire.

## Outils et langages prévus :
-	Langage(s) : Python
-	Bibliothèques ou logiciels : rasterio, numpy, shapely,  matplotlib, pandas, geopandas, docxtpl, docx2pdf

## Répartition des tâches dans l’équipe
- Daphné Normandeau : Écriture de la logique du code, production de graphiques, rédaction de la documentation
- Charles Raymond : Recherche de données, généralisation et révision du code, peaufinage des graphiques et de la présentation

## Pour rouler le script Python:
Afin de rouler adéqautement le script Python, il est essentiel de télécharger la dernière version à jour du code (soit Mini_projet_CR_DN_v11.py), les données du Google Drive présent dans le lien plus haut ainsi que le template 'template_word_proj.docx'. Une fois les données (rasters et couches) décompressées, il faudra ajuster le chemin de ces différents éléments et du template dans les variables globales du code. La date pour laquelle on souhaite générer le rapport PDF est régie par la variable globale 'date_of_report'. Il est possible que vous devrez installer des librairies en entête du code si votre environnement Python personnel ne les possède pas encore. ATTENTION: Dû à des limitations du code dans l'algorithme de polygonisation des zones brûlées, il est futile d'émettre un rapport pour des dates trop avancées (10 juin et plus). En effet, les variations entre les images seront minimes, puisque l'algorithme considérera la zone entière comme étant brûlée...
Pour voir un aperçu du fonctionnement du programme et des résultats, une vidéo de présentation du projet est disponible sur Youtube : 



  
  
  

