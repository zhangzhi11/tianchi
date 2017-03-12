# -*- coding: utf-8 -*-
"""
Created on Sat Apr 09 09:15:22 2016

@author: Administrator
"""
import pandas as pd
import  matplotlib.pyplot   as plt  
import numpy as np 






uheader=['user_id','song_id','gmt_create','action_type','Ds']
user_actions=pd.read_csv('data/mars_tianchi_user_actions.csv',header=None,names=uheader)
user_actions['Ds']=pd.to_datetime(user_actions['Ds'],format='%Y%m%d')
sheader=['song_id','artist_id','publish_time','song_init_plays','Language','Gender']
songs=pd.read_csv('data/mars_tianchi_songs.csv',header=None,names=sheader)
songs['publish_time']=pd.to_datetime(songs['publish_time'],format='%Y%m%d')

#"画出歌曲播放量的统计直方图"
def plot_song_play_hist(user_actions):
    song_action=user_actions.loc[:,['song_id','action_type']]
    song_action_play=song_action[(song_action['action_type']==1)]
  #  song_id=pd.unique(song_action_play['song_id'])
    
    song_listen_sum=song_action_play.groupby('song_id').size()
    #song_listen_hist=song_listen_sum.value_counts()
    #song_listen_sum.hist(bins=50,range=(0,10000)).get_figure().savefig('plot.png')
    plt.figure()
    plt.hist(song_listen_sum, bins = 50,range=(0,1000))
    plt.xlabel('sum of plays')
    plt.ylabel('sum of songs')
    plt.savefig('song_listen_hist.jpg')
    plt.show()
    
#画出歌曲初始播放量和发行时间的统计图
def plot_init_plays(songs):
    song_init=songs.loc[:,['publish_time','song_init_plays']]
    song_init_sum=song_init.groupby('publish_time').sum()
    song_init_sum1=pd.DataFrame({'time':song_init_sum.index,'init_hot':song_init_sum['song_init_plays']})
    plt.figure()
    song_init_sum1.plot()
    plt.xlabel('time')
    plt.ylabel('sum of init_plays')
    plt.savefig('init_plays.jpg')
    plt.show()    
plot_song_play_hist(user_actions)
plot_init_plays(songs)
#plot_song_init(songs)