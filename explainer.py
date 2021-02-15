# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:16:06 2021

@author: Karthik
"""
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  lime import lime_tabular
from sklearn.model_selection import train_test_split

def interpret_model(dataframe,feature_set,model):
    

  """ dataframe - Specify the Name of the dataframe
      feature_set - The set of features you want to use(list)
      models- Should Be in a dictionary form where model should be a function passed as a value with the name of model as the key of dict
      wrong_predictions=True (Change to `false` if u want to only see the correct classification results for the model)
  """
  X=dataframe[feature_set]
  y=dataframe['Default_Status']
  train_X, test_X, train_y, test_y = train_test_split(X,y, random_state=20)

  

# =============================================================================
#   model.fit(train_X,train_y)
#   model_preds=model.predict(test_X)
# =============================================================================




  from lime.lime_tabular import LimeTabularExplainer
  class_names =['Wont Default','Will Default']
  #instantiate the explanations for the data set
  limeexplainer = LimeTabularExplainer(train_X.values, class_names=class_names, feature_names=feature_set,kernel_width=3,verbose=False, mode='classification')
  return limeexplainer
  
  




  

  