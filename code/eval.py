# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:09:02 2016

@author: Administrator
"""

import pandas as pd
import numpy as np

def e(x,y):
    e=((x-y)/y)**2
    return e
    

def eval(predictPath,truthPath):
    pred=pd.read_csv(predictPath,header=None,names=['artist_id','pPlays','Ds'])
    truth=pd.read_csv(truthPath,header=None,names=['artist_id','tPlays','Ds'])
    if not pred.irow==50:
        print 'The size of predict csv is wrong!'
        return 0
    pred.g    
    artist_names=pd.unique(truth['artist_id'])
    ass=[]
    bss=[]
    for artist in artist_names:
        p_artist_table=pred[pred['artist_id']==artist]
        t_artist_table=truth[truth['artist_id']==artist]
        pt_artist=pd.merge(p_artist_table,t_artist_table,on='Ds')
        es=pt_artist['pPlays,tPalys'].map(e)
        a=np.sqrt(es.sum(axis=0)/es.irow)
        b=np.sqrt(truth['tPlays'].sum())
        ass.append(a)
        bss.append(b)
    ass=np.array(ass)
    bss=np.array(bss)
    f=np.sum((1-ass)*bss)
    return f
        
        
        
        