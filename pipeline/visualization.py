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
    
    plt.figure(figsize=(9,5))
    x = np.arange(100,1000,100)
    plt.plot(x, ates, color='k',label='ATE')
    plt.fill_between(x, ates_inf,ates_sup)
    plt.ylim(min(ates_inf)-1,min(ates_inf)+1)
    plt.xlabel("N observations")
    plt.ylabel("ATE")
    plt.title("Evaluation de l'ATE en fonction du nombre d'observations")
    plt.legend(loc=1,numpoints=1)
    
    plt.savefig('visu.pdf')
    
    #plt.show();