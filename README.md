# Hough Transform - mif17

Ce projet à été réalisé dans le cadre de l'unité d'enseignement "Analyse d'image - mif17" de première année de master informatique à l'Université Claude Bernard Lyon 1.

## Structure du projet 

```bash
├── ressources # images utilisées pour les démos
|   ├── cathedrale_lyon.jpg
|   ├── droites_simples.png
|   ├── exemple_simple.jpg
|   └── image_simple.jpg
├── src # fichiers c++
|   ├── applications.cpp 
|   ├── gradient.hpp
|   ├── hough.hpp
|   ├── kernel.hpp
|   ├── main.cpp # programme principale 
|   ├── multithreading.hpp
|   ├── ui.hpp
|   └── utils.hpp
├── CMakeLists.txt
├── rapport.pdf
└── README.md
```

## Prérequis 

L'installation de la bibliothèque OpenCV c++ est necéssaire pour que le projet compile. Voir ici: https://opencv.org/get-started/

## Compilation du projet (Sous linux / Wsl)

1. Ce placer à la racine du projet: 
```bash
    cd ~/path/to/directory/
```
2. Créer un répertoire `build` et se placer dedans: 
```bash
    mkdir build && cd build
```
3. Générer le projet avec `cmake` en mode `Release`: 
```bash
    cmake -DCMAKE_BUILD_TYPE=Release ..
```
4. Compiler le projet avec `make` et exécuter le programme: 
```bash
    make && ./hough [lines|circles] <filepath> 
```

## Explication des arguments de commande

L'exécutable `./hough` prend deux arguments :
1. Le **mode** 
   - `lines` -> détection de ligne 
   - `circles` -> détection de cercles 
2. Le **chemin du fichier** testé  
   - rien -> `../ressources/Droites_simples.png`
   - `../ressources/<image_name>`
  
## Application 

Il suffit d'appuyer sur la touche '**R**' pour exécuter l'algorithme de **Hough Transform** avec les paramètres définit dans le panneau de contrôle. 

### Panneau de configuration

Au démarrage de l'application, un panneau de contrôle, avec des sliders sur différents paramètres, s'affiche. Plusieurs types de paramètres peuvent être modifier pour influer sur le résultat de l'algorithme de Hough Transform:
- **[Input]** : Paramètres pour le filtre bilatéral appliqué à l'image lors de la phase de pré-traitement.  
- **[Binary]** : Paramètres à modifier lorsqsu'une image binaire est utilisée en entrée ou si l'on ne souhaite pas utiliser de gradient.  
- **[Gradient]** : Paramètres à modifier pour le calcul du gradient. 
- **[Hough]** : Paramètres correspondant généralement aux seuils utilisés dans l'algorithme de la transformée de Hough. 
- **[Hough + Gradient]** : Paramètres à modifier si le gradient est utilisé pour la détection de contours. 


### Démo Hough Line

La démonstration de détection de lignes affiche cinq images : 
- Input image : l'image d'entrée 
- Filtered image : l'image filtrée 
- Edges : les contours détecté (soit par le gradient, soit par la fonction '*cv::canny()*' d'OpenCV)
- Accumulator : l'accumulateur représentant le nombre d'intersections pour chaque droite définit en fonction de $\boldsymbol\theta$ et $\boldsymbol\rho$
- Final result : L'image résultat qui combine l'image source avec les lignes détectées

### Démo Hough Circle

La démonstration de détection de cercles affiche les mêmes images que celle des lignes excepté l'accumulateur qui, ayant trois dimensions, ne peut être affiché simplement avec la fonction '*cv::imshow()*' d'OpenCV. 