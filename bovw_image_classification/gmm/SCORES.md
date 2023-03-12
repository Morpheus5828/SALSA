# Scores Obtenus

## Présentation

### Méthode: Bag of words
La méthode se découpe en trois grandes étapes:

- On décrit les images à l'aide d'un algorithme. Chaque image contient alors plusieurs descripteurs.
  
  
- On regroupe tout les descripteurs ensemble afin d'identifier des motifs. Pour cela, on applique un algorithme de clustering.
  
  
- Pour chaque image, on crée un histogramme des fréquences de chaque motif. Notre modèle s'entraînera alors sur ces histogrammes.

### Variation du score
Le score final peut varier selon plusieurs critères. Afin d'obtenir le meilleur score possible, il est nécessaire d'effectuer des tests sur différents ensembles de données. Il est possible de faire varier les points suivants:

- **Extraction des motifs:** Afin de produire les descripteurs, ont peut choisir de prendre *toutes* les données, ou seulement celles du dossier *Mer*.


- **Taille des images:** Est-ce que mettre toutes les images à la *même taille* apporte une différence ?


- **Descripteur:** Comme il existe plusieurs algorithmes permettant de décrire une image, on peut se demander si certains peuvent produire un meilleur score que d'autres.


- **Clustering:** Le choix de l'algorithme permettant de trouver les clusters joue un rôle important. De plus, chaque algorithme de clustering possède ses propres hypers paramètres que l'on peut modifier.


- **Normalisation:** Avant d'entraîner le modèle sur les données, on peut les normaliser.


- **Modèle:** Enfin, le modèle choisi influe grandement sur le score final. De plus, il est aussi important de faire varier ses hypers paramètres.

## Algorithme de clustering

On sépare les recherches selon le choix de l'algorithme de clustering. Une fois ce dernier choisie, nous ferons varier tous les autres paramètres cités précédemment.

### Gaussian Mixture

C'est un algorithme de clustering similaire à KMeans. La plus grande différence réside dans le fait que les clusters peuvent avoir une forme différente de la sphère. Toutefois, nous devons toujours choisir *le nombre de cluster.*

## Sauvegarde

Les valeurs intermédiaires sont sauvegardées dans le dossier `save`. Les fichiers sont répartis dans des sous répertoires.

### Le dossier `features`
Il contient les sauvegardes des features *(matrice de tous les descripteurs)* et des descripteurs de chaque image. Chaque couple *features/descriptors* est situé dans le dossier associé à son algorithme de description, comme `sift`.
À l'intérieur, les dossiers sont séparés deux catégories, selon les données d'apprentissages choisis : 
- `all`: toutes les images
- `sea`: dossier Mer

Dans chacun de ces deux dossiers, les données sont évidemment séparées en `descriptors` et `features`.

Enfin, les fichiers sont nommés selon la règle suivante: `[resized]-<algorithme>-<données>-<nom>.txt`
- **resized:** est optionnel. Il est écrit si les images ont été redimensionnées. Selon la méthode de redimensionnage utilisée, sa valeur est différente :
    - *resized_w:* pour une redimension selon la largeur.
    - *resized_h:* pour une redimension selon la hauteur.
    - *resized_a:* pour une redimension selon la largeur et la hauteur. Dans ce cas, l'image est déformée.
- **algorithme:** est remplacé par le nom de l'algorithme de description utilisé.
- **données:** est remplacé par l'ensemble de données choisi pour extraire les motifs, c'est-à dire *all* ou *sea*.
- **nom:** correspond à l'élément sauvegardé, c'est-à dire *features* ou *descriptors*.

### Le dossier `clustering`
Il contient les sauvegardes des models des algorithmes de clustering. Les fichiers sont d'abord situés dans le dossier correspond à l'algorithme de clustering utilisé, puis ils sont ensuite répartis selon l'algorithme de description utilisé.
Les fichiers sont nommés selon la règle suivante: `[resized]-<algorithme_clustering>-<algorithme_description>-<données>-<nb_clusters>.joblib`
- **resized:** est optionnel. Il est écrit si les images ont été redimensionnées. Selon la méthode de redimensionnage utilisée, sa valeur est différente :
    - *resized_w:* pour une redimension selon la largeur.
    - *resized_h:* pour une redimension selon la hauteur.
    - *resized_a:* pour une redimension selon la largeur et la hauteur. Dans ce cas, l'image est déformée.
- **algorithme_clustering:** est remplacé par le nom de l'algorithme de clustering utilisé.
- - **algorithme_description:** est remplacé par le nom de l'algorithme de description utilisé.
- **données:** est remplacé par l'ensemble de données choisi pour extraire les motifs, c'est-à dire *all* ou *sea*.
- **nb_clusters:** correspond aux nombres de clusters utilisés par l'algorithme de clustering.


# Résultats

Chaque score obtenu est en réalité une moyenne. En effet, chaque modèle est crée et entraîné 50 fois.

## Test 1 : Nombre de clusters
L'algorithme de clustering met énormément de temps à être calculé sur un nombre de cluster trop élevé. Ainsi, nous allons regarder les différents scores obtenus
sur deux valeurs différentes : *log(descriptors)* et *log(features)*. Le premier correspond donc au log du *nombre d'images utilisé* pour extraire les motifs et le second
correspond au log de *tous* les motifs obtenus.

### Taille : log(descriptors)

On a les paramètres suivants :
- **Extraction des motifs:** Mer.
- **Taille des images:** Non redimensionnées.
- **Descripteur:** SIFT
- **Clustering:** Gaussian Mixture (*clusters=5 car log(descriptors))*

Essayons différents modèles avec ces paramètres.
- **Gaussian NB:** 0.5713253012048194
- **Linear SVC:** 0.5703614457831325
- **Perceptron linéaire :** 0.6421686746987953
- **Perceptron Multicouche** *(solver=adam)***:** 0.6412048192771083
- **Régression Logistique:** 0.6956626506024094
- **K Neighbors:** 0.6881927710843374
- **Bagging** *(estimator=Arbre)***:** 0.6650602409638552
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.6985542168674698
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.6893975903614458
- **Random Forest:** 0.6968674698795181

Si je normalise les valeurs d'entraînements avant :
- **Gaussian NB:** 0.5792771084337349
- **Linear SVC:** 0.6802409638554218
- **Perceptron linéaire :** 0.603132530120482
- **Perceptron Multicouche** *(solver=adam)***:** 0.70289156626506
- **Régression Logistique:** 0.6848192771084336
- **K Neighbors:** 0.6956626506024094
- **Bagging** *(estimator=Arbre)***:** 0.6616867469879518
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.6518072289156627
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.7142168674698794
- **Random Forest:** 0.7038554216867469

### Taille : log(features)

On a les paramètres suivants :
- **Extraction des motifs:** Mer.
- **Taille des images:** Non redimensionnées.
- **Descripteur:** SIFT
- **Clustering:** Gaussian Mixture (*clusters=12 car log(features))*

Sans normaliser :
- **Gaussian NB:** 0.5877108433734942
- **Linear SVC:** 0.5751807228915662
- **Perceptron linéaire :** 0.626987951807229
- **Perceptron Multicouche** *(solver=adam)***:** 0.6467469879518073
- **Régression Logistique:** 0.6722891566265059
- **K Neighbors:** 0.6956626506024094
- **Bagging** *(estimator=Arbre)***:** 0.6922891566265059
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.6840963855421688
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.6915662650602408
- **Random Forest:** 0.7248192771084336

Puis en normalisant :
- **Gaussian NB:** 0.5746987951807231
- **Linear SVC:** 0.6725301204819278
- **Perceptron linéaire :** 0.6120481927710845
- **Perceptron Multicouche** *(solver=adam)***:** 0.6980722891566267
- **Régression Logistique:** 0.6848192771084336
- **K Neighbors:** 0.6783132530120481
- **Bagging** *(estimator=Arbre)***:** 0.6838554216867468
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.6539759036144578
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.7012048192771081
- **Random Forest:** 0.7089156626506024

## Test 2 : Données d'extraction de motifs

On change les données d'extraction de motifs et on se base sur toutes les données.

### Taille : log(descriptors)
On a les paramètres suivants :
- **Extraction des motifs:** Toutes les données.
- **Taille des images:** Non redimensionnées.
- **Descripteur:** SIFT
- **Clustering:** Gaussian Mixture (*clusters=6 log(descriptors))*

Sans normaliser :
- **Gaussian NB:** 0.5739759036144578
- **Linear SVC:** 0.5836144578313253
- **Perceptron linéaire :** 0.6334939759036143
- **Perceptron Multicouche** *(solver=adam)***:** 0.6306024096385542
- **Régression Logistique:** 0.6759036144578314
- **K Neighbors:** 0.6853012048192771
- **Bagging** *(estimator=Arbre)***:** 0.6804819277108435
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.6780722891566264
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.7012048192771083
- **Random Forest:** 0.7142168674698793

Puis en normalisant :
- **Gaussian NB:** 0.56
- **Linear SVC:** 0.6809638554216866
- **Perceptron linéaire :** 0.5913253012048191
- **Perceptron Multicouche** *(solver=adam)***:** 0.7120481927710841
- **Régression Logistique:** 0.67855421686747
- **K Neighbors:** 0.680722891566265
- **Bagging** *(estimator=Arbre)***:** 0.6785542168674698
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.6481927710843374
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.7146987951807227
- **Random Forest:** 0.7014457831325303

### Taille : log(features)
Trop long à calculer !!!

## Test 3 : Redimensionner les images

On retaille toutes les images de trois manières différentes : 
- Selon la largeur.
- Selon la hauteur.
- Selon la largeur **et** la hauteur. Dans ce cas, l'image sera déformée.

Dans tous les cas, les nouvelles valeurs de largeur (respectivement hauteur) sont obtenues en faisant la moyenne des largeurs (respectivement hauteurs) de toutes les
images d'entraînements.

### Redimension : largeur
On a les paramètres suivants :
- **Extraction des motifs:** Mer.
- **Taille des images:** Redimensionnées selon la largeur.
- **Descripteur:** SIFT
- **Clustering:** Gaussian Mixture (*clusters=5 log(descriptors))*

Sans normaliser :
- **Gaussian NB:** 0.7036144578313253
- **Linear SVC:** 0.5812048192771084
- **Perceptron linéaire :** 0.6224096385542169
- **Perceptron Multicouche** *(solver=adam)***:** 0.6120481927710845
- **Régression Logistique:** 0.7065060240963855
- **K Neighbors:** 0.651566265060241
- **Bagging** *(estimator=Arbre)***:** 0.6749397590361446
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.6640963855421689
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.6645783132530121
- **Random Forest:** 0.6930120481927707

Puis en normalisant :
- **Gaussian NB:** 0.705301204819277
- **Linear SVC:** 0.7159036144578311
- **Perceptron linéaire :** 0.6228915662650603
- **Perceptron Multicouche** *(solver=adam)***:** 0.7115662650602408
- **Régression Logistique:** 0.7219277108433735
- **K Neighbors:** 0.6893975903614457
- **Bagging** *(estimator=Arbre)***:** 0.6624096385542169
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.7009638554216869
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.7207228915662648
- **Random Forest:** 0.6869879518072292

On change maintenant la taille des clusters. On a alors les paramètres suivants :
- **Extraction des motifs:** Mer.
- **Taille des images:** Redimensionnées selon la largeur.
- **Descripteur:** SIFT
- **Clustering:** Gaussian Mixture (*clusters=12 log(features))*

Sans normaliser :
- **Gaussian NB:** 0.6995180722891565
- **Linear SVC:** 0.5614457831325302
- **Perceptron linéaire :** 0.6115662650602409
- **Perceptron Multicouche** *(solver=adam)***:** 0.6426506024096387
- **Régression Logistique:** 0.713734939759036
- **K Neighbors:** 0.6474698795180721
- **Bagging** *(estimator=Arbre)***:** 0.6775903614457832
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.6462650602409639
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.669397590361446
- **Random Forest:** 0.6857831325301205

Puis en normalisant :
- **Gaussian NB:** 0.706746987951807
- **Linear SVC:** 0.723132530120482
- **Perceptron linéaire :** 0.6612048192771084
- **Perceptron Multicouche** *(solver=adam)***:** 0.7113253012048191
- **Régression Logistique:** 0.7067469879518073
- **K Neighbors:** 0.6843373493975905
- **Bagging** *(estimator=Arbre)***:** 0.6843373493975904
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.7021686746987953
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.7122891566265058
- **Random Forest:** 0.6956626506024098

### Redimension : hauteur
On a les paramètres suivants :
- **Extraction des motifs:** Mer.
- **Taille des images:** Redimensionnées selon la hauteur.
- **Descripteur:** SIFT
- **Clustering:** Gaussian Mixture (*clusters=5 log(descriptors))*

Sans normaliser :
- **Gaussian NB:** 0.7067469879518071
- **Linear SVC:** 0.5660240963855422
- **Perceptron linéaire :** 0.6322891566265061
- **Perceptron Multicouche** *(solver=adam)***:** 0.6443373493975902
- **Régression Logistique:** 0.7202409638554215
- **K Neighbors:** 0.6824096385542167
- **Bagging** *(estimator=Arbre)***:** 0.6848192771084336
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.6816867469879518
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.6679518072289157
- **Random Forest:** 0.7028915662650601

Puis en normalisant :
- **Gaussian NB:** 0.6975903614457831
- **Linear SVC:** 0.714457831325301
- **Perceptron linéaire :** 0.6544578313253012
- **Perceptron Multicouche** *(solver=adam)***:** 0.720722891566265
- **Régression Logistique:** 0.7175903614457829
- **K Neighbors:** 0.6898795180722891
- **Bagging** *(estimator=Arbre)***:** 0.6720481927710844
- **Bagging** *(estimator=Perceptron_linéaire)***:** 0.7045783132530121
- **Bagging** *(estimator=Perceptron_multicouche(solver=adam))***:** 0.7303614457831324
- **Random Forest:** 0.697831325301205