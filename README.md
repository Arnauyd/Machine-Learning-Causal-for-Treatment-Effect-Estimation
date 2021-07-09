# Projet Fil Rouge - Air Liquide 
## Machine Learning Causal pour l’estimation des effets d’un traitement

[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com)  [![forthebadge](http://forthebadge.com/images/badges/powered-by-electricity.svg)](http://forthebadge.com)

## Pour commencer

Entrez ici les instructions pour bien débuter avec votre projet...

### Pré-requis

Ce qu'il est requis pour commencer avec votre projet...

- Programme 1
- Programme 2
- etc...

### Installation

Les étapes pour installer votre programme....

Dites ce qu'il faut faire...

_exemple_: Executez la commande ``telnet mapscii.me`` pour commencer ensuite [...]


Ensuite vous pouvez montrer ce que vous obtenez au final...

## Description du projet

Durant notre projet, on fait des études sur des données simulées dans le cadre d’un traitement binaire. Les données sont séparées en deux parties. Le premier groupe représente le groupe témoin et le deuxième correspond au groupe ayant reçu le traitement. 
Dans cette partie, on explique les différentes étapes lors de la construction du pipeline d’apprentissage automatique causal i.e. causal machine learning. Ce dernier est un outil utilisé pour mener une étude de simulation approfondie comparant les performances des différents estimateurs afin de mieux appréhender leurs applications sur des données réelles.
On explore également l’estimation de l’effet d’un traitement pour des sous-populations d’intérêts. L’ambition est de trouver des profils types à l’aide d’une approche basée sur le clustering pour recommander le type d’intervention le plus adapté à chaque individu ou à chaque groupe d’individus.

### Pipeline du machine learning causal

<p align="center">
    <img src='img/pipeline.JPG'>
</p> 

La figure ci-dessus montre le schéma récapitulatif du notre pipeline. Tout d’abord, on commence par générer des données simulées. Pour cela, on produit une matrice de données X avec d colonnes et N lignes. Cette matrice représente les caractéristiques (features) des sujets étudiés, d correspond au nombre de caractéristiques de chaque individu et N est le nombre d’observations autrement dit le nombre de personnes participant à l’étude. Chaque observation i correspondant à une ligne Xi suit la loi normale.

Ensuite, on simule l’affectation du traitement selon un vecteur W de longueur N. Le i-ème élément de ce vecteur correspond à une valeur dans {0, 1} indiquant si le sujet i reçoit le traitement. Lors de ce projet, on procède de deux façons pour calculer le vecteur W. Dans le premier cas, on attribue le traitement de manière aléatoire c’est-à-dire sans prendre compte des caractéristiques de l’individu. On fixe un paramètre p appelé score de propension dont la valeur est entre 0 et 1 et on génère N valeurs avec à chaque fois une probabilité p d’avoir 1 comme le montre la formule. Dans le deuxième cas, les caractéristiques des sujets étudiés sont considérées lors de l’attribution du traitement. Le score de propension devient une fonction de X. Par conséquent, chaque individu i a une probabilité différente d’avoir le traitement.

Finalement, on simule les réponses potentielles de chaque individu au traitement. Pour cela on spécifie des fonctions mu0 et mu1 appelée les fonctions réponses, et on calcule les sorties Y selon la formule.

## Auteurs
Membres de l'équipe : 
* **Amal Benali* _alias_ [@outout14](https://github.com/)
* **Fatima-Ezzahra Jait* _alias_ [@outout14](https://github.com/)
* **Jean-Philippe Quach* _alias_ [@outout14](https://github.com/)
* **Mohammed El yaagoubi** _alias_ [@moelyaagoubi](https://github.com/moelyaagoubi)
* **Arnaud Sangouard* _alias_ [@outout14](https://github.com/)

Liste des encadrants :
* Encadrants académiques : Florence D'alche-Buc - Nathan Noiry - Yannick Guyonvarch
* Encadrants industriels : Habiboulaye Amadou Boubacar - Mehdi Rahim


