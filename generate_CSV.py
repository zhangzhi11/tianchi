# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:02:24 2016

@author: Administrator
"""
import time
import numpy as np
import pandas as pd
from unit_function import get_expregress_All_results_online
DS_TIMEFORMAT='%Y%m%d'
TIMEFORMAT='%Y%m%d %X'
#result为预测天数*50的list
#artist namelist
#outdir为输出的路径
#将结果转化为csv文件
def generate_csv(result,namelist,outdir='submit/'):
#    outlist=[]
    localtime=time.asctime()
    a=localtime.split(' ')
    localtime_str=a[1]+'_'+a[2]+'_'+a[3].split(':')[0]+a[3].split(':')[1]
    outname='submit_'+localtime_str+'.csv'
    #生成从2015-09-01到2015-10-30的时间
    #daylist=range(20150901,20150931)+range(20151001,20151032)
    #daylist=[20150901:20150931]
    daylist=pd.date_range('20150901','20151030')
    for i1,series in enumerate(result):
#        for i,s in enumerate(series):
        series=np.floor(series).astype(int)
        df2=pd.DataFrame({'artist_id':namelist[i1],
                          'plays':series,
                          'Ds':daylist})            
        if i1==0:
            result=df2
        else:
            result=pd.concat([result,df2])
    result.to_csv(outdir+outname,columns=['artist_id','plays','Ds'],date_format=DS_TIMEFORMAT,index=False,header=False)

results1=get_expregress_All_results_online(artist_listen['listen_sum'])
generate_csv(results1,artist_listen['artist_list'])       
