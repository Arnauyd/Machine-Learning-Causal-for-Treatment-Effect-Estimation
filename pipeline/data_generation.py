# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 07:35:25 2021

@author: MELYAAGOUBI
"""

import random
import numpy as np
from scipy.stats import bernoulli



def treatment_assign(Nobs, dim, X, p=None):
    '''
    Input: 
    
    - Nobs : Nombre de lignes da la matrice X i.e. nombre de personnes.
    - dim : Nombre de colonnes de la matrice X i.e. nombres de caractéristiques (features).
    - X : La matrice X des features de dimension (Nobs, dim).
    - p : Score de propension. 

    Output:
    
    - W : Vecteur de dimension (1, Nobs) contenant des 0 ou 1 pour désigner 
          l'affectation du traitement.
    '''
    sigmoid = lambda x: 1/(1+np.exp(-x))
    omega = np.random.uniform(0, 1, (Nobs, dim))

    if type(p) == float :
        W = bernoulli.rvs(p, size = Nobs) 
        
    else :
      p = np.zeros(Nobs)
      for i in range(Nobs):
        p[i] = sigmoid(omega[i] @ X[i])
        
      W = bernoulli.rvs(p, size = Nobs) 

    return W


def causal_generation(Nobs, dim, beta, bias, f, g, p):
    '''
    Input :
    
    - Nobs : Nombre de lignes da la matrice X i.e. nombre de personnes.
    - dim : Nombre de colonnes de la matrice X i.e. nombres de caractéristiques (features).
    - beta : Vecteur de dimension (2, dim).
    - bias : Vecteur de dimension (1, 2).
    - f et g : Fonctions pour calculer les résultats Y (outputs).
    - p : Argument de la fonction treatment_assign().
    
    Output:
    
    (X, Y, W) : Triplet contenant la matrice X des features de dimension (Nobs, dim), 
                Y le vecteur des résultats potentiels (Nobs, 1) et W le vecteur de 
                longueur (1, Nobs) contenant des 0 ou 1 pour désigner l'affectation 
                du traitement.
    '''
    moy = np.zeros(dim)
    var = np.eye(dim)
    X = np.random.multivariate_normal(moy, var, Nobs)
    Y = np.zeros(Nobs)

    W = treatment_assign(Nobs, dim, X, p)

    for i in range(Nobs):
        bruit = np.random.normal(0, 1)
        if W[i] == 0:
            Y[i] = f(beta[0] @ X[i] + bias[0]) + bruit
        if W[i] == 1:
            Y[i] = g(beta[1] @ X[i] + bias[1]) + bruit
            
    return (X, W, Y)


def random_select_bootstrap(Nobs, sample_size, Nsamples):
    '''
    Choix d'indices à partir de l'ensemble des données (X, W, Y) à l'aide de 
    la méthode boostrap.
   
    Input :
        
    - Nobs : Nombre de lignes da la matrice X i.e. nombre de personnes.
    - sample_size : Le nombre de personnes selectionnés à chaque tirage avec remise.
    - Nsamples : Le nombre de fois que l'opération de tirage est répétée.
    
    Output :
        
    - index_samples : Une liste contenant tous les échantillons d'indices générés 
                      à l'aide des tirages avec remise.

    '''
    index_samples = []
    for i in range(Nsamples):
        samples = random.sample(range(1, Nobs), sample_size)
        index_samples.append(samples)
        
    return index_samples


def causal_generation_bootstrap(Nobs, dim, beta, bias, f, g, p, B):
    '''
    Création d'une liste à l'aide de la méthode boostrap
    
    Input :
        
    - Nobs : Nombre de lignes da la matrice X i.e. nombre de personnes.
    - dim : Nombre de colonnes de la matrice X i.e. nombres de caractéristiques (features).
    - beta : Vecteur de dimension (2, dim).
    - bias : Vecteur de dimension (1, 2).
    - f et g : Fonctions pour calculer les résultats Y (outputs).
    - p : Argument de la fonction treatment_assign().
    - B : Nombre d'échantillons Boostrap, 999 est une valeur par défaut pertinente
  
    Output:
        
    - [(X, W, Y)] : liste de B Triplets contenant la matrice X des features, W le vecteur 
                    de l'affectation du traitement et Y le vecteur des résultats potentiels. 
                
    '''
    Bootstraps = []

    for b in range(B):
        Bootstraps.append(causal_generation(Nobs, dim, beta, bias, f, g, p))
      
    return Bootstraps

