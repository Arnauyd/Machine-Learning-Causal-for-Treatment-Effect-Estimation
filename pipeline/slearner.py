# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 07:06:47 2021

@author: MELYAAGOUBI
"""

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin



class SLearner(BaseEstimator, ClassifierMixin):
    """ Homemade SLearner class """
    
    def __init__(self, base_estimator=LinearRegression()):
        # init
        self.estimator = base_estimator
        
    def fit(self, X, W, Y):
        # Initiation des variables
        self.X = X
        self.W = W
        self.Y = Y
        self.features = np.hstack((self.X, self.W[:,np.newaxis]))
        self.clf = self.estimator.fit(self.features, self.Y)

    def predict_CATE(self, x):
        # Complete the method      
        self.mu0_hat = self.clf.predict(np.c_[x, np.zeros(len(x))])
        self.mu1_hat = self.clf.predict(np.c_[x, np.ones(len(x))])
        return self.mu1_hat - self.mu0_hat

    def predict_ATE(self):
        return (self.mu1_hat - self.mu0_hat).mean()
    
    


def run_slearner(X, W, Y, baselearner):
  slearner = SLearner(base_estimator = baselearner)
  slearner.fit(X,W,Y)
  
  cate_hat_S = slearner.predict_CATE(X)
  #print("- Les dimensions du CATE = {}.".format(cate_hat_S.shape))
  ate_hat_S = slearner.predict_ATE()
  #print("- L'estimation de la valeur de l'ATE = {}.".format(ate_hat_S))
  return ate_hat_S