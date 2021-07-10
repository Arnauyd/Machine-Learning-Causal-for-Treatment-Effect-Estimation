# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 23:25:40 2021

@author: MELYAAGOUBI
"""

import sys
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import inspect
from termcolor import colored

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from data_generation import causal_generation

from realvalues_estimation import monte_carlo
from slearner import run_slearner
from tlearner import run_tlearner
from xlearner import run_xlearner
from drlearner import run_drlearner

def printfunc(f):
  fstring = str(inspect.getsourcelines(f)[0])
  fstring = fstring.strip("['\\n']").split(" = ")[1].split("x:")[1].strip("np.")
  return fstring

def tab_gen(N, beta, bias, f, g):

    
    dim = []
    real_ate = []
    bases = []
    score_prop = []

    base_learners = {"Linear Regression" : LinearRegression(),
                     "Random Forest" : RandomForestRegressor(),
                     "XGboost" : GradientBoostingRegressor()}

    base_learners1 = {"Linear Regression" : LinearRegression(),
                      "Random Forest" : RandomForestRegressor(),
                      "XGboost" : GradientBoostingRegressor()}

    res = {"Propension score": score_prop, 
           "Base Learner" : bases,
           "Dimension" : dim,
           "Monte Carlo ATE" : real_ate,
           "S-Learner": [],  "T-Learner": [],
           "X-Learner": [], "Doubly Robust Learning": []}

    for b in list(base_learners.keys()):
      bl = base_learners[b]
      b2 = base_learners1[b]
      
      for d in [5]:
        beta0 = np.random.uniform(1, 30, (1, d))
        beta = np.vstack((beta0, beta0))                      
        
        # Real Value ATE
        mc_ate = round(monte_carlo(10**6, d, beta, bias, f, g), 3)

        for p in [0.1, 0.5, 0.9, None]:
          dim.append(d)
          real_ate.append(mc_ate)
          if p == None:
            score_prop.append("confounding")
          else:
            score_prop.append(p)

          bases.append(b)

          slearner = []
          tlearner = []
          xlearner = []
          drlearner = []

          for _ in range(10):
            
            X, W, Y = causal_generation(N, d, beta, bias, f, g, p)

            # S-Learner
            ate_S = run_slearner(X, W, Y, bl)
            slearner.append(round(ate_S, 3))

            # T-Learner
            ate_T = run_tlearner(X, W, Y, bl, b2)
            tlearner.append(round(ate_T, 3))

            # X-Learner
            ate_hat_X = run_xlearner(X, W, Y, bl, b2)
            xlearner.append(round(ate_hat_X, 3))

            # Doubly Robust Learning
            ate_dr = run_drlearner(X, W, Y, bl,  b2)
            drlearner.append(round(ate_dr, 3))

    
    
          # Results
          s_mean_value = round(np.mean(slearner), 3)
          s_std_value = round(np.std(slearner), 3)
          res["S-Learner"].append(str(s_mean_value) + " ± " + str(s_std_value))
    
          t_mean_value = round(np.mean(tlearner), 3)
          t_std_value = round(np.std(tlearner), 3)
          res["T-Learner"].append(str(t_mean_value) + " ± " + str(t_std_value))
    
          x_mean_value = round(np.mean(xlearner), 3)
          x_std_value = round(np.std(xlearner), 3)
          res["X-Learner"].append(str(x_mean_value) + " ± " + str(x_std_value))
    
          dr_mean_value = round(np.mean(drlearner), 3)
          dr_std_value = round(np.std(drlearner), 3)
          res["Doubly Robust Learning"].append(str(dr_mean_value) + " ± " + str(dr_std_value))
    

    res["Dimension"] = dim
    res["Base Learner"] =  bases
    res["Monte Carlo ATE"] = real_ate
    res["Propension score"] = score_prop

    df = pd.DataFrame(res, columns = list(res.keys()))
    df = df.set_index(["Base Learner", "Dimension", "Monte Carlo ATE", "Propension score"])
    
    # Saving the reference of the standard output
    original_stdout = sys.stdout    
 
    with open('tab.txt', 'w') as file:
        sys.stdout = file
        print()
        print("              Results of ATE estimation based on a simulation model:")
        print()
    
        print("- Nombre de d'observations [N] = {}.".format(N))
        print("- La fonction f(x) = {}.".format(printfunc(f)))
        print("- La fonction g(x) = {}.".format(printfunc(g)))

        print(df)
        # Reset the standard output
        sys.stdout = original_stdout
    
    