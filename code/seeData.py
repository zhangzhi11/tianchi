# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 08:58:28 2016

@author: Administrator
"""

import pandas as pd
import numpy as np
import  matplotlib.pyplot   as plt  
uheader=['user_id','song_id','gmt_create','action_type','Ds']
users=pd.read_csv('mars_tianchi_user_actions.csv',header=None,names=uheader)
users['Ds']=pd.to_datetime(users['Ds'])
sheader=['song_id','artist_id','publish_time','song_init_plays','Language','Gender']
songs=pd.read_csv('mars_tianchi_songs.csv',header=None,names=sheader)

#播放歌曲的表user表
#play_songs=users[users['action_type']==1]
song_artist=songs.loc[:,['song_id','artist_id']]
song_artist_user=pd.merge(users,song_artist,on='song_id')
song_artist_user.dropna(axis=0)

artist_names=pd.unique(song_artist_user['artist_id'])
for artist_name in artist_names:
#artist_name=artist_names[0]
    print artist_name
    table=song_artist_user[(song_artist_user['artist_id']==artist_name)]
    #计算table中每天‘action——type=1的数目’
    plt.figure()
    sum=table.groupby('Ds').size()
    
    plt.plot(sum)
    plt.savefig('picture\\'+str(artist_name)+'.jpg')
    