# -*- coding: utf-8 -*-
"""
Created on Mon May 09 21:09:19 2016

@author: Administrator
"""

import pandas as pd 
import numpy as np
import matplotlib.pylab as plt
#from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.seasonal import seasonal_decompose
#matplot inline
from matplotlib.pylab import rcParams
from sklearn import linear_model
import unit_function

rcParams['figure.figsize']=15, 6
unit_function.test_all(artist_listen['listen_sum'])
(preds,results1,results2)=unit_function.get_expregress_All_results_offline(artist_listen['listen_sum'])
F1=unit_function.eval(preds,results1)
F2=unit_function.eval(preds,results2)
print '1预测F值为'
print F1
print '2预测F值为'
print F2
