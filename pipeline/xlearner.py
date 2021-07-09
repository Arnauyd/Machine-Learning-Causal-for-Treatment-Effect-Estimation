# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 04:02:15 2021

@author: MELYAAGOUBI
"""


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin


class XLearner(BaseEstimator, ClassifierMixin):
    """ Homemade XLearner class """
    
    def __init__(self, outcome_learner0, outcome_learner1, effect_learner0, effect_learner1):
        # init
        self.outcome_learner0 = outcome_learner0
        self.outcome_learner1 = outcome_learner1
        self.effect_learner0 = effect_learner0
        self.effect_learner1 = effect_learner1

    def fit(self, X, W, Y):
        # Initiation des variables
        self.X = X
        self.W = W
        self.Y = Y 
        
        #Stage 1 : Estimate the average outcomes μ0(x) and  μ1(x)
        self.mu_0 = self.outcome_learner0.fit(X[self.W==0,:], self.Y[self.W==0])
        self.mu_1 = self.outcome_learner1.fit(X[self.W==1,:], self.Y[self.W==1])
        
        #Stage 2 : Impute the user level treatment effects
        self.D0 = self.mu_1.predict(X[self.W==0,:]) - self.Y[self.W==0] 
        self.D1 = self.Y[self.W==1] - self.mu_0.predict(X[self.W==1,:])    
        
        #estimate τ1(x) = E[D1|X=x], and τ0(x) = E[D0|X=x] using machine learning models:
        self.tau_0 = self.effect_learner0 .fit(X[self.W==0,:], self.D0)
        self.tau_1 = self.effect_learner1 .fit(X[self.W==1,:], self.D1)
        

    def predict_CATE(self, x, p):
        # Complete the method         
        self.CATE_hat = p*self.tau_0.predict(x) + (1-p)*self.tau_1.predict(x)
        return self.CATE_hat

    def predict_ATE(self):
        return (self.CATE_hat).mean()
    
    


def run_xlearner(X, W, Y, outcome_learner0, outcome_learner1, effect_learner0, effect_learner1):
  xlearner = XLearner(outcome_learner0, outcome_learner1, 
                      effect_learner0, effect_learner1)
  xlearner.fit(X,W,Y)
  cate_hat_X = xlearner.predict_CATE(X, W)
  ate_hat_X = xlearner.predict_ATE()
  return ate_hat_X