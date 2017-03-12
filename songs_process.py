# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 19:32:04 2016

@author: Administrator
"""

import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import scipy  
from statsmodels.tsa.seasonal import seasonal_decompose
DS_TIMEFORMAT='%Y%m%d'
import cPickle as pickle



def save_artist_csv(user_actions,songs):
    outdir='data/artists/'
    song_artist=songs.loc[:,['song_id','artist_id']]
    song_artist_user_action=pd.merge(user_actions,song_artist,on='song_id')
    song_artist_user_action.dropna(axis=0)

    artist_names=pd.unique(songs['artist_id'])
    
    for artist_name in artist_names:
        print artist_name
        artist_table = song_artist_user_action[song_artist_user_action['artist_id'] == artist_name]
        #del artist_table['artist_id']        
        artist_table.to_csv(outdir+artist_name+'.csv',columns=['user_id','song_id','gmt_create','action_type','Ds'],date_format=DS_TIMEFORMAT,index=False,header=False)
    return 0

def read_artist_csv(artist_name):
    inputdir='data/artists/'
    header=['user_id','song_id','gmt_create','action_type','Ds']
    artist_table=pd.read_csv(inputdir+artist_name+'.csv',header=None,names=header)
    artist_table['Ds']=pd.to_datetime(artist_table['Ds'],format='%Y%m%d')
    return artist_table
#save_artist_csv(user_actions,songs)

    
def fill_zero_days(dta,a=1):
#填补没有信息的天数
    days= pd.date_range('3/1/2015',periods=183,freq='D')
    #这里为没有数值的设置为局部平均值  
    #ts = pd.Series(1,index = days)
    ts=pd.Series(a,index=days)   
    for day in dta.index:
        ts[day]=dta[day]      
    return ts 
    
#对于每个歌手区别显著歌曲和非显著歌曲
#得到每个歌曲的长度为183序列
def get_songs(artist_table,artist_name,save_mask=False):
    outdir='data/artist_songs_Ds/'
    song_names = pd.unique(artist_table['song_id'])
    song_listen_sum=[]
    for song_name in song_names:
        song_table=artist_table[artist_table['song_id']==song_name]
        song_listen_table = song_table[song_table['action_type']==1]
        song_Ds=fill_zero_days(song_listen_table.groupby('Ds').size(),a=0)
        song_listen_sum.append(song_Ds)
    song_listen_sum=np.array(song_listen_sum)
    songs_listen_sum={'listen_sum':song_listen_sum,
                'song_list':song_names}
    if save_mask:        
        f = open(outdir+artist_name+'.pkl','wb')
        pickle.dump(songs_listen_sum,f)
        f.close()
    
    return songs_listen_sum    
def get_artist_songs_matrix(artist_name):
    artist_table=read_artist_csv(artist_name)
    songs_listen_sum=get_songs(artist_table,artist_name,save_mask=True)
    return 

f=open('data/artist_songs_Ds/c5f0170f87a2fbb17bf65dc858c745e2.pkl','r')  
artist=pickle.load(f)  
f.close()

#artist=pickle.load('data/artist_songs_Ds/4b8eb68442432c242e9242be040bacf9.pkl')

#a=get_artist_songs_matrix('c5f0170f87a2fbb17bf65dc858c745e2')  
#artist_names=pd.unique(songs['artist_id'])
#
#for artist_name in artist_names:
#    print artist_name
#    get_artist_songs_matrix(artist_name)
#def analysis_artist_songs(songs_listen_sum):
    #计算所有的songs的均值
    