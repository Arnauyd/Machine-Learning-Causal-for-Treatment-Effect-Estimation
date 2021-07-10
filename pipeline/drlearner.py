# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 04:03:04 2021

@author: MELYAAGOUBI
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

class DRLearner(BaseEstimator, ClassifierMixin):
    """ Homemade DRLearner class """
    
    def __init__(self, model_regression, model_final, model_propensity=LogisticRegression()):
        # init
        self.model_regression = model_regression
        self.model_propensity = model_propensity
        self.model_final = model_final
        

    def fit(self, X, W, Y):
        # Initiation des variables
        self.X = X
        self.W = W
        self.Y = Y 
        
        #Stage 1 : Regression of the outcomes μ(X,T) = E[Y|X,W,T]
        self.features = np.hstack((self.X, self.W[:,np.newaxis]))
        self.mu = self.model_regression.fit(self.features, self.Y)
        
        #Stage 1 : Model to estimate the propensity_score
        self.model_propensity = CalibratedClassifierCV(self.model_propensity)
        self.model_propensity.fit(self.X, self.W)
        self.propensity = self.model_propensity.predict_proba(X)

        #Stage 1 : predict Y_pred
        self.Y_pred_0 = self.mu.predict(np.hstack((self.X, np.zeros((self.X.shape[0],1)))))
        self.Y_pred_0 += (Y - self.Y_pred_0) * (1 - self.W) / self.propensity[:,0]
        self.Y_pred_1 = self.mu.predict(np.hstack((self.X, np.ones((self.X.shape[0],1)))))
        self.Y_pred_1 += (Y - self.Y_pred_1) * (self.W) / self.propensity[:,1]
        
        #Stage 2 : fit model final
        self.model_final.fit(self.X, self.Y_pred_1 - self.Y_pred_0)
        
    def predict_CATE(self, x):
        # Complete the method         
        self.CATE_hat = self.model_final.predict(x)
        return self.CATE_hat

    def predict_ATE(self):
        return (self.CATE_hat).mean()
    
    
    

def run_drlearner(X, W, Y, model_regression, model_final):
  drlearner = DRLearner(model_regression, model_final)
  drlearner.fit(X,W,Y)

  cate_hat_dr = drlearner.predict_CATE(X)
  #print("- Les dimensions du CATE = {}.".format(cate_hat_dr.shape))
  ate_hat_dr = drlearner.predict_ATE()
  #print("- L'estimation de la valeur de l'ATE = {}.".format(ate_hat_dr))
  return ate_hat_dr