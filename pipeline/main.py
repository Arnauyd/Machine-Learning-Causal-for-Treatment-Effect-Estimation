# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 07:37:52 2021

@author: MELYAAGOUBI
"""

import numpy as np
import scipy as sp
from scipy import stats
from scipy import integrate
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from data_generation import causal_generation, causal_generation_bootstrap
from slearner import SLearner
from tlearner import TLearner
from realvalues_estimation import monte_carlo
from confidence_interval import IC
from visualization import visualization, graphic_comparison

from save_data import save_data

#from causalml.inference.meta import LRSRegressor


if __name__ == '__main__':
    # paramètres pour faciliter le calcul de l'ATE par intégration
    N = 1000
    d = 2                                     # d = 2, afin de pouvoir être calculé par intégration et avec Monte Carlo
    p = 0.7
    beta = np.random.uniform(1, 30, (1, d))
    beta = np.vstack((beta,beta))               # beta0 = beta1           
    bias = np.array([10,0])                   # Gamma0 = Gamma1 
    f = lambda x:x
    g = lambda x:x
    
    save_data("simu_data.txt", N, d, p, f, g, "Linear regression", "S-Learner et T-Learner")
    
    print("------------ Estimation de l'ATE/CATE sur des données synthétiques --------------")
    print()
    # Génération des données
    X, W, Y = causal_generation(N, d, beta, bias, f, g, p = 0.8)
    #print(X.shape, W.shape, Y.shape)
    
    # S-Learner
    slearner = SLearner(base_estimator = LinearRegression())
    slearner.fit(X, W, Y)
    cate_hat_S = slearner.predict_CATE(X)
    ate_hat_S = slearner.predict_ATE()
    print("- Calcul de la valeur de l'ATE avec S-Learner = {}.".format(ate_hat_S))
    
    # T-Learner 
    tlearner = TLearner(base_estimator0 = LinearRegression(), base_estimator1 = LinearRegression())
    tlearner.fit(X,W,Y)
    cate_hat_T = tlearner.predict_CATE(X)
    ate_hat_T = tlearner.predict_ATE()
    print("- Calcul de la valeur de l'ATE avec T-Learner = {}.".format(ate_hat_T))


    # Intervalle de confiance
    B = 999
    Bootstraps = causal_generation_bootstrap(beta, bias, B, N, d, f, g, p)
    _, IC_inf, IC_sup = IC(Bootstraps, base_metalerner = SLearner(), alpha=0.05)
    print("- Intervalle de confiance pour l'ATE : [{}, {}].".format(IC_inf, IC_sup))

 
    # Estimation avec Monté-Carlo
    print('- ATE estimée par la méthode de monte carlo: {}'.format(monte_carlo(10**6, d, beta, bias, f, g)))
    
    
    # Visualization
    B = 999
    visualization(beta, bias, B, N, d, f, g, p, base_metalerner = SLearner())
    


    
    
    