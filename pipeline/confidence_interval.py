# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 08:25:10 2021

@author: MELYAAGOUBI
"""

import numpy as np 
from scipy.stats import norm

from slearner import SLearner

def IC(X, W, Y, samples, base_metalearner, alpha=0.05):
    
    '''
    Calculate l'intervalle de confiance d'un métalearner
    
    Input :
    
    Bootstraps : Liste d'échantillons Boostraps comprenant B triplets (X,Y,W)
    base_metalerner : metalearner à évaluer
    
    Output:
    
    IC : (IC inf , IC sup) du métalearner 
    '''
    
    #Calcul de l'ATE les B estimateurs du Bootstrap
    N = len(samples)
    ATEs = np.zeros((N))
    #base_metalearner = SLearner()
    
    for i in range(N):
        Xb, Wb, Yb = X[samples[i]], W[samples[i]], Y[samples[i]]
        base_metalearner.fit(Xb, Wb, Yb)
        base_metalearner.predict_CATE(Xb)
        ATEs[i] = base_metalearner.predict_ATE()
    
    #Calcul des intervalles de confiance
    Mu_ATEs = ATEs.mean()
    #std_ATEs = ATEs.std()
    #ATEs_tilt = (ATEs-Mu_ATEs)/std_ATEs
    #ATEs_tilt.sort()
    ATEs.sort()
    
    
    IC_inf, IC_sup = ATEs[int(N*(alpha/(2)))], ATEs[int(N*(1-alpha/(2)))]
    
    return Mu_ATEs, IC_inf, IC_sup