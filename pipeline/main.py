# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 07:37:52 2021

@author: MELYAAGOUBI
"""

import numpy as np

from sklearn.linear_model import LinearRegression

from data_generation import causal_generation, causal_generation_bootstrap, random_select_bootstrap
from slearner import SLearner
from tlearner import TLearner
from realvalues_estimation import monte_carlo
from confidence_interval import IC
from visualization import visualization, graphic_comparison

from save_data import save_data
from tab_generation import tab_gen
#from causalml.inference.meta import LRSRegressor


if __name__ == '__main__':
    # paramètres pour faciliter le calcul de l'ATE par intégration
    N = 1000
    d = 2                                     # d = 2, afin de pouvoir être calculé par intégration et avec Monte Carlo
    p = 0.7
    beta = np.random.uniform(1, 30, (1, d))
    beta = np.vstack((beta,beta))               # beta0 = beta1           
    bias = np.array([10,0])                   # Gamma0 = Gamma1 
    f = lambda x:np.sin(x)
    g = lambda x:x
    
    #save_data("simu_data.txt", N, d, p, f, g, "Linear regression", "S-Learner et T-Learner")
    
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
    
    """
    # T-Learner 
    tlearner = TLearner(base_estimator0 = LinearRegression(), base_estimator1 = LinearRegression())
    tlearner.fit(X,W,Y)
    cate_hat_T = tlearner.predict_CATE(X)
    ate_hat_T = tlearner.predict_ATE()
    print("- Calcul de la valeur de l'ATE avec T-Learner = {}.".format(ate_hat_T))
    """

    # Intervalle de confiance
    Nsamples = 250
    sample_size = int(0.8*N)
    samples = random_select_bootstrap(N, 250, 100)
    _, IC_inf, IC_sup = IC(X, W, Y, samples, base_metalearner = SLearner(), alpha=0.05)
    print("- Intervalle de confiance pour l'ATE : [{}, {}].".format(IC_inf, IC_sup))

 
    # Estimation avec Monté-Carlo
    print('- ATE estimée par la méthode de monte carlo: {}'.format(monte_carlo(10**6, d, beta, bias, f, g)))
    
    
    # Visualization
    #visualization(beta, bias, d, f, g, p, base_metalearner = SLearner())
    """
    nb_obs = [i for i in range(100,2000,100)]
    
    graphic_comparison(nb_obs, d, p, beta, bias, f, g, B, 
                       base_learner_homemade=SLearner(), base_learner_causalml=LRSRegressor())
    """
    
    # Tab generation
    tab = tab_gen(N, beta, bias, f, g)
    
    