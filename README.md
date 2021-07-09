# Projet Fil Rouge - Air Liquide 
## Machine Learning Causal pour l’estimation des effets d’un traitement

[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com)  [![forthebadge](http://forthebadge.com/images/badges/powered-by-electricity.svg)](http://forthebadge.com)

 ##### Encadrants industriels : Habiboulaye Amadou Boubacar - Mehdi Rahim
 ##### Encadrants académiques : Florence D'alche-Buc - Nathan Noiry - Yannick Guyonvarch
 ##### Membres de l'équipe : Amal Benali - Fatima-Ezzahra Jait - Jean-Philippe Quach - Mohammed El yaagoubi -  Arnaud Sangouard

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

La figure ci-dessus montre le schéma récapitulatif du notre pipeline. Tout d’abord, on commence par générer des données simulées. Pour cela, on produit une matrice de données X avec d colonnes et N lignes. Cette matrice représente les caractéristiques (features) des sujets étudiés, d correspond au nombre de caractéristiques de chaque individu et N est le nombre d’observations autrement dit le nombre de personnes participant à l’étude. Chaque observation i correspondant à une ligne $X_{i}$ suit la loi normale.


## Fabriqué avec

Entrez les programmes/logiciels/ressources que vous avez utilisé pour développer votre projet

_exemples :_
* [Materialize.css](http://materializecss.com) - Framework CSS (front-end)
* [Atom](https://atom.io/) - Editeur de textes

## Contributing

Si vous souhaitez contribuer, lisez le fichier [CONTRIBUTING.md](https://example.org) pour savoir comment le faire.

## Versions
Listez les versions ici 
_exemple :_
**Dernière version stable :** 5.0
**Dernière version :** 5.1
Liste des versions : [Cliquer pour afficher](https://github.com/your/project-name/tags)
_(pour le lien mettez simplement l'URL de votre projets suivi de ``/tags``)_

## Auteurs
Listez le(s) auteur(s) du projet ici !
* **Jhon doe** _alias_ [@outout14](https://github.com/outout14)

Lisez la liste des [contributeurs](https://github.com/your/project/contributors) pour voir qui à aidé au projet !

_(pour le lien mettez simplement l'URL de votre projet suivi de ``/contirubors``)_

## License

Ce projet est sous licence ``exemple: WTFTPL`` - voir le fichier [LICENSE.md](LICENSE.md) pour plus d'informations

