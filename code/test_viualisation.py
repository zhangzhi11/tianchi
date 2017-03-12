# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:28:04 2016

@author: Wei
"""


import pandas as pd
import  matplotlib.pyplot   as plt  


#此函数用于绘制每个歌手每天的被操作数，包括播放，下载，收藏
def plot_artist_daily_action(user_actions,songs):
    song_artist=songs.loc[:,['song_id','artist_id']]
    song_artist_user_action=pd.merge(user_actions,song_artist,on='song_id')
    song_artist_user_action.dropna(axis=0)

    artist_names=pd.unique(song_artist_user_action['artist_id'])
    for artist_name in artist_names:
        #artist_name=artist_names[0]
        print artist_name
        table=song_artist_user_action[(song_artist_user_action['artist_id']==artist_name)]
        listen_table=table[(table['action_type']==1)]
        download_table=table[(table['action_type']==2)]
        collect_table=table[(table['action_type']==3)]
        listen_sum=listen_table.groupby('Ds').size()
        download_sum=download_table.groupby('Ds').size()
        collect_sum=collect_table.groupby('Ds').size()

        #plot
        plt.figure()
        plt.plot(listen_sum,label='listen')
        plt.plot(download_sum,label='download')
        plt.plot(collect_sum,label='collect')
        plt.legend()
        plt.savefig('picture\\artist_daily_action'+str(artist_name)+'.jpg')
    return
 

#此函数用于绘制用户的日均操作   
#横轴x表示日均操作数，纵轴y表示人数
#min_op和max_op限制x轴的范围以更好地可视化图像
def cal_avg_artist_daily_action(user_actions,songs,min_op,max_op):
    #series index 为每日操作数，数值表示该操作数对应的人数
    user_daily_op=(user_actions.groupby('user_id').size()/10).value_counts()
    plt.figure()
    user_daily_op.plot()
    plt.xlim(min_op,max_op)
    count_max=user_daily_op[min_op+1]
    plt.ylim(0,count_max)
    return 

#此函数用于统计播放量和发行新歌的行为之间的关系    
def cor_action_play_nums(user_actions,songs):
    song_artist=songs.loc[:,['song_id','artist_id']]
    song_artist_user_action=pd.merge(user_actions,song_artist,on='song_id')
    song_artist_user_action.dropna(axis=0)

    artist_names=pd.unique(song_artist_user_action['artist_id'])
    for artist_name in artist_names:
        #artist_name=artist_names[0]
        print artist_name
        table=song_artist_user_action[(song_artist_user_action['artist_id']==artist_name)]
        listen_table=table[(table['action_type']==1)]
        listen_sum=listen_table.groupby('Ds').size()
        songs_publish_date=songs[(songs['artist_id']==artist_name)].loc[:,['song_id','publish_time']]
        songs_publish_date=songs_publish_date.groupby('publish_time').size()
        songs_publish_date=songs_publish_date
        #plot
        plt.figure()
        time1=pd.to_datetime('20150301',format='%Y%m%d')
        time2=pd.to_datetime('20150830',format='%Y%m%d')
        #plt.xlim(time1,time2)
        plt.plot(listen_sum,label='listen')
        plt.plot(songs_publish_date,'ro')
        plt.legend()
        plt.savefig('picture/artist_initial_hot/artist_daily_action'+str(artist_name)+'.jpg')
    
    return    

uheader=['user_id','song_id','gmt_create','action_type','Ds']
user_actions=pd.read_csv('data/mars_tianchi_user_actions.csv',header=None,names=uheader)
user_actions['Ds']=pd.to_datetime(user_actions['Ds'],format='%Y%m%d')
sheader=['song_id','artist_id','publish_time','song_init_plays','Language','Gender']
songs=pd.read_csv('data/mars_tianchi_songs.csv',header=None,names=sheader)
songs['publish_time']=pd.to_datetime(songs['publish_time'],format='%Y%m%d')


#plot_artist_daily_action(user_actions,songs)
#plot_artist_daily_action(user_actions,songs)
#cal_avg_artist_daily_action(user_actions,songs,0,50)
cor_action_play_nums(user_actions,songs)