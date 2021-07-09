# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 07:22:21 2021

@author: MELYAAGOUBI
"""


import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class TLearner(BaseEstimator, ClassifierMixin):
    """ Homemade SLearner class """
    
    def __init__(self, base_estimator0 = LinearRegression(), base_estimator1 = LinearRegression() ):
        # init
        self.estimator0 = base_estimator0
        self.estimator1 = base_estimator1

    def fit(self, X, W, Y):
        # Initiation des variables
        self.X = X
        self.W = W
        self.Y = Y
        self.mu_0 = self.estimator0.fit(X[self.W==0,:], self.Y[self.W==0])
        self.mu_1 = self.estimator1.fit(X[self.W==1,:], self.Y[self.W==1])

    def predict_CATE(self, x):
        # Complete the method         
        self.Y_0_hat = self.mu_0.predict(x)
        self.Y_1_hat = self.mu_1.predict(x)
        return self.Y_1_hat - self.Y_0_hat

    def predict_ATE(self):
        return (self.Y_1_hat - self.Y_0_hat).mean()
    
    
    
def run_tlearner(X, W, Y, baselearner0, baselearner1):
  tlearner = TLearner(base_estimator0 = baselearner0, 
                      base_estimator1 = baselearner1)
  tlearner.fit(X,W,Y)

  cate_hat_T = tlearner.predict_CATE(X)
  #print("- Les dimensions du CATE = {}.".format(cate_hat_S.shape))
  ate_hat_T = tlearner.predict_ATE()
  #print("- L'estimation de la valeur de l'ATE = {}.".format(ate_hat_T))
  return ate_hat_T