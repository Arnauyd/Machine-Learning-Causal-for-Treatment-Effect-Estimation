# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 07:35:25 2021

@author: MELYAAGOUBI
"""

import numpy as np
from scipy.stats import bernoulli

def treatment_assign(Nobs, d, X, p):
    '''
    Input: 
    
    p : score de propension.
    Nobs : Nombre de lignes da la matrice X i.e. nombre de personnes.
    
    Output:
    
    W : Vecteur de taille Nobs contenant des 0 ou 1 pour désigner l'affectation du traitement.
    '''
    sigmoid = lambda x: 1/(1+np.exp(-x))
    
    omega = np.random.uniform(0, 1, (Nobs, d))
    psi = np.random.uniform(0, 1, (Nobs, 1))

    if p == None:
      p = np.zeros(Nobs)
      for i in range(Nobs):
        p[i] = sigmoid(omega[i] @ X[i])
      W = bernoulli.rvs(p, size = Nobs) 
    else:
      W = bernoulli.rvs(p, size = Nobs) 
    
    return W


def causal_generation(Nobs, dim, beta, bias, f, g, p):
    '''
    Input :
    
    Nobs : Nombre de lignes da la matrice X i.e. nombre de personnes.
    dim : Nombre de colonnes de la matrice X i.e. nombres de caractéristiques (features).
    beta : Vecteur de dimension (2, dim).
    bias : Vecteur de dimension (1, 2).
    W : Vecteur de dimension (1, Nobs) contenant des 0 ou 1 pour désigner 
    l'affectation du traitement.
    f et g sont des fonctions.
    
    Output:
    
    (X, Y, W) : Triplet contenant la matrice X des features, Y le vecteur des 
                résultats potentiels et W le vecteur de l'affectation du traitement.
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


def causal_generation_bootstrap(beta, bias, B, Nobs, dim, f, g, p):
    '''
    Create list of bootstrap elements
    Input :
    
    B : Nombre d'échantillons Boostrap, 999 est une valeur par défaut pertinente
    Nobs : Nombre de lignes da la matrice X i.e. nombre de personnes, 1000 par défaut
    dim : Nombre de colonnes de la matrice X i.e. nombres de caractéristiques 
    (features), 2 par défaut
    beta : Vecteur de dimension (2, dim).
    bias : Vecteur de dimension (1, 2).
    W : Vecteur de dimension (1, Nobs) contenant des 0 ou 1 pour désigner 
    l'affectation du traitement.
    f et g sont des fonctions, identité par défaut
    
    Output:
    
    [(X, W, Y)] : liste de B Triplets contenant la matrice X des features, W 
    le vecteur de l'affectation du traitement et Y le vecteur des résultats 
    potentiels. 
                
    '''
    Bootstraps=[]

    for b in range(B):
        Bootstraps.append(causal_generation(Nobs, dim, beta, bias, f, g, p))
      
    return Bootstraps