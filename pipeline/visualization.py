# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 09:24:10 2021

@author: MELYAAGOUBI
"""

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.5})
plt.rcParams['figure.figsize'] = 10, 8

from data_generation import causal_generation
from data_generation import causal_generation_bootstrap
from confidence_interval import IC
from slearner import SLearner

       
def visualization(beta, bias, B, N, d, f, g, p, base_metalerner=SLearner()):
    ates = []
    ates_inf = []
    ates_sup = []
    B = 999
    
    for i in range(100,1000,100):
        Bootstraps = causal_generation_bootstrap(beta, bias, B, N, d, f, g, p)
        mu, inf, sup = IC(Bootstraps, base_metalerner, alpha=0.05)
        ates.append(mu)
        ates_inf.append(inf)
        ates_sup.append(sup)

    plt.figure(figsize=(10,8))
    x=np.arange(100,1000,100)
    plt.plot(x, ates, color='k',label='ATE')
    plt.fill_between(x, ates_inf,ates_sup)
    plt.ylim(min(ates_inf)-1,min(ates_inf)+1)
    plt.xlabel("N observations")
    plt.ylabel("ATE")
    plt.title("Evaluation de l'ATE en fonction du nombre d'observations")
    plt.legend(loc=1,numpoints=1)
    
    plt.savefig('visu.pdf')
    plt.show();
    
    
    
def graphic_comparison(nb_obs, d, p, beta, bias, f, g, B, base_learner_homemade, base_learner_causalml):

  '''
  Create a graphic to compare our bases learners vs base learners from causalml 

  Input :
    
  B : Nombre d'échantillons Boostrap, 999 est une valeur par défaut pertinente
  nb_obs : Nombre de lignes da la matrice X, les listes sont acceptables i.e. nombre de personnes, 1000 par défaut
  dim : Nombre de colonnes de la matrice X i.e. nombres de caractéristiques (features), 2 par défaut
  beta : Vecteur de dimension (2, dim).
  bias : Vecteur de dimension (1, 2).
  W : Vecteur de dimension (1, Nobs) contenant des 0 ou 1 pour désigner l'affectation du traitement.
  f et g sont des fonctions, identité par défaut
    
  Output:
  Graphique comparant nos base_learner avec ceux du causal_ml
  '''

  ate_causal_ml = []
  lb_causal_ml = []
  ub_causal_ml = []

  ates = []
  ates_inf = []
  ates_sup = []

  for n in nb_obs :

    # Génération des données
    Bootstraps = causal_generation_bootstrap(beta, bias, B, n, d, f, g, p)
    mu, inf, sup = IC(Bootstraps, base_metalerner=base_learner_homemade, alpha=0.05)
    ates.append(mu)
    ates_inf.append(inf)
    ates_sup.append(sup)

    # S learner causal ML
    X, W, Y = causal_generation(n, d, beta, bias, f, g, p)
    lr = base_learner_causalml
    te, lb, ub = lr.estimate_ate(X, W, Y)
    ate_causal_ml.append(te[0])
    lb_causal_ml.append(lb[0])
    ub_causal_ml.append(ub[0])


  plt.figure(figsize=(12,7))
  plt.plot(nb_obs,ates, color='blue',label = 'ATE_homemade')
  plt.fill_between(nb_obs, ates_inf,ates_sup,alpha = 0.5, color='blue', label = 'IC_ATE_homemade')
  plt.plot(nb_obs, ate_causal_ml,color='orange',label='ATE_causal_ml')
  plt.fill_between(nb_obs, lb_causal_ml, ub_causal_ml,alpha = 0.5, color='orange',label='IC_ATE_causalml')
  plt.xlabel("Nombre d'observations")
  plt.ylabel("Valeurs de l'ATE")
  plt.title("Evaluations de l'ATE en fonction du nombre d'observations")
  plt.legend()
  plt.savefig('visu.pdf')
  plt.show()
  
  