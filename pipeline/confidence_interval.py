# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 08:25:10 2021

@author: MELYAAGOUBI
"""

import numpy as np 
from scipy.stats import norm

from slearner import SLearner

def IC(Bootstraps, base_metalerner=SLearner(), alpha=0.05):
    
    '''
    Calculate l'intervalle de confiance d'un métalearner
    
    Input :
    
    Bootstraps : Liste d'échantillons Boostraps comprenant B triplets (X,Y,W)
    base_metalerner : metalearner à évaluer
    
    Output:
    
    IC : (IC inf , IC sup) du métalearner 
    '''
    
    #Calcul de l'ATE les B estimateurs du Bootstrap
    ATEs = np.zeros((len(Bootstraps)))
    base_metalearner = SLearner()
    
    for i in range(len(Bootstraps)):
        base_metalearner.fit(Bootstraps[i][0], Bootstraps[i][1], Bootstraps[i][2])
        base_metalearner.predict_CATE(Bootstraps[i][0])
        ATEs[i] = base_metalearner.predict_ATE()
    
    #Calcul des intervalles de confiance
    Mu_ATEs = ATEs.mean()
    std_ATEs = ATEs.std()
    ATEs_tilt = (ATEs-Mu_ATEs)/std_ATEs
    ATEs_tilt.sort()
    ATEs.sort()
    
    B = len(Bootstraps)
    IC_inf, IC_sup = ATEs[int(B*(alpha/(2)))], ATEs[int(B*(1-alpha/(2)))]
    
    return Mu_ATEs, IC_inf, IC_sup