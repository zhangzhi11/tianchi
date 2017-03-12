# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:10:04 2016

@author: Administrator
"""
import pandas as pd
import numpy as np
df = pd.DataFrame({'key1':['a','a','b','b','a'],
                   'key2':['one','two','one','two','one'],
                   'data1':np.random.randn(5),
                   'data2':np.random.randn(5)})
grouped = df.groupby(df['key1'])                   
print grouped.mean()