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
1. Le mode 
   - `lines` -> détection de ligne 
   - `circles` -> détection de cercles 
2. Le fichier 
   - rien -> `../ressources/Droites_simples.png`
   - `../ressources/<image_name>`