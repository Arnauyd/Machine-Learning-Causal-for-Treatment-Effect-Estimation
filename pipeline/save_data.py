# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:34:27 2021

@author: MELYAAGOUBI
"""

def save_data(filename, N, d, p, f, g, base_learner, meta_learner):
    
    file = open(filename,"w")
    
    L = ["Nombre d'observations =" + str(N) + "\n",
         "Dimension =" + str(d) + "\n",
         "Score de propension =" + str(p) + "\n", 
         "fonction f =" + str(f) + "\n",
         "fonction g =" + str(g) + "\n",
         "Base learner =" + str(base_learner) + "\n",
         "Meta learner =" + str(meta_learner) + "\n"]  
    
    file.writelines(L)
    file.close() 
    return 0