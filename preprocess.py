# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:28:04 2016

@author: Wei
"""



#import pkgs
import numpy as np
import pandas as pd
import  matplotlib.pyplot   as plt  
import time
#from statsmodels.tsa.arima_model import ARIMA







#此函数用于绘制每个歌手每天的被操作数，包括播放，下载，收藏
def plot_artist_daily_action(user_actions,songs):
    song_artist=songs.loc[:,['song_id','artist_id']]
    song_artist_user_action=pd.merge(user_actions,song_artist,on='song_id')
    song_artist_user_action.dropna(axis=0)

    artist_names=pd.unique(songs['artist_id'])
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
        plt.savefig('picture\\artist_daily_action\\rand'+str(artist_name)+'.jpg')
    return
    
def fill_zero_days(dta,a=1):
#填补没有信息的天数
    days= pd.date_range('3/1/2015',periods=183,freq='D')
    #这里为没有数值的设置为局部平均值  
    #ts = pd.Series(1,index = days)
    ts=pd.Series(a,index=days)   
    for day in dta.index:
        ts[day]=dta[day]      
    return ts 
    
def plot_artist_everysong_daily_action(user_actions,songs):
    song_artist=songs.loc[:,['song_id','artist_id']]
    song_artist_user_action=pd.merge(user_actions,song_artist,on='song_id')
    song_artist_user_action.dropna(axis=0)

    artist_names=pd.unique(songs['artist_id'])
    for artist_name in artist_names:
        #artist_name=artist_names[0]
        print artist_name
        table=song_artist_user_action[(song_artist_user_action['artist_id']==artist_name)]
        listen_table=table[(table['action_type']==1)]
        #download_table=table[(table['action_type']==2)]
        #collect_table=table[(table['action_type']==3)]
        listen_sum=listen_table.groupby('Ds').size()
        song_names=pd.unique(listen_table['song_id'])
        plt.figure()
        plt.plot(listen_sum,label='listen')
        for song_name in song_names:
            song_table=listen_table[(listen_table['song_id']==song_name)]
            song_listen_sum=song_table.groupby('Ds').size()
            song_listen_sum=fill_zero_days(song_listen_sum,a=0)
            plt.plot(song_listen_sum)
            #画出这首歌的发行时间，如果在时间范围之内
#            song_info=songs[songs['song_id']==song_name]
#            publish_time=song_info['publish_time'].values[0]
#            if publish_time>pd.datetime(2015,3,1) and publish_time<pd.datetime(2015,8,30):
#                day=publish_time-pd.datetime(2015,3,1)
#                day_int=day.days
#                plt.plot(day_int,-20,'ro')
        plt.legend()
        plt.savefig('picture\\artist_songs_daily_action\\'+str(artist_name)+'.jpg')
    return
plot_artist_everysong_daily_action(user_actions,songs)    
    
#此函数用于绘制用户的日均操作   
#横轴x表示用户人数，纵轴y表示总播放量
def cal_avg_artist_daily_action(user_actions,songs):
    #series index 为每日操作数，数值表示该操作数对应的人数
    #实际上有183天的数据
    user_daily_op=((user_actions.groupby('user_id').size()/183.0).value_counts()).sort_index(axis=0)
    user_daily_op1=user_daily_op.copy()
    #plt.figure()
    #user_daily_op.plot()#.bar()
    #plt.xlim(min_op,max_op)
    #count_max=user_daily_op.values[min_op+1]
    #plt.ylim(0,count_max)
    
    #统计人数
    ac_cnt=0
    for index in user_daily_op.index:
        ac_cnt=user_daily_op[index]*index+ac_cnt
        user_daily_op[index]=ac_cnt
        
    #统计播放量    
    ac_cnt=0   
    for index in user_daily_op1.index:
        ac_cnt=user_daily_op1[index]+ac_cnt
        user_daily_op1[index]=ac_cnt
        
    #生成人数和播放量之间的关系    
    df=pd.DataFrame(user_daily_op.values,index=user_daily_op1.values)    
    plt.figure()
    df.plot()    
    plt.xlabel('people count')
    plt.ylabel('total operation')
    return df


#此函数用于绘制用户的日均操作   
#横轴x表示日均操作数，纵轴y表示人数
#min_op和max_op限制x轴的范围以更好地可视化图像
def cal_avg_artist_daily_action1(user_actions,songs,min_op,max_op):
    #series index 为每日操作数，数值表示该操作数对应的人数
    user_daily_op=((user_actions.groupby('user_id').size()/183.0).value_counts()).sort_index(axis=0)
    #限制图像大小    
    user_daily_op=user_daily_op[min_op:max_op]
    user_daily_op1=user_daily_op.copy()
    #plt.figure()
    #user_daily_op.plot()#.bar()
    #plt.xlim(min_op,max_op)
    #count_max=user_daily_op.values[min_op+1]
    #plt.ylim(0,count_max)
    ac_cnt=0
    for index in user_daily_op.index:
        user_daily_op[index]
        ac_cnt=user_daily_op[index]+ac_cnt
        user_daily_op[index]=ac_cnt

    plt.figure()
    user_daily_op.plot()
    plt.xlabel('operation per day')
    plt.ylabel('people count')
    
    
    ac_cnt=0
    for index in user_daily_op1.index:
        user_daily_op1[index]
        ac_cnt=user_daily_op1[index]*index+ac_cnt
        user_daily_op1[index]=ac_cnt

    plt.figure()
    user_daily_op1.plot()
    plt.xlabel('operation per day')
    plt.ylabel('total operation')
    return 
    
#此函数用于统计播放量和发行新歌的行为之间的关系    
def cor_action_play_nums(rand_user_actions,frequent_user_actions,songs):
    rand_song_artist=songs.loc[:,['song_id','artist_id']]
    rand_song_artist_user_action=pd.merge(rand_user_actions,rand_song_artist,on='song_id')
    rand_song_artist_user_action.dropna(axis=0)



    frequent_song_artist=songs.loc[:,['song_id','artist_id']]
    frequent_song_artist_user_action=pd.merge(frequent_user_actions,frequent_song_artist,on='song_id')
    frequent_song_artist_user_action.dropna(axis=0)



    artist_names=pd.unique(songs['artist_id'])
    
    for artist_name in artist_names:
        #artist_name=artist_names[0]
        print artist_name
       
        songs_publish_date=songs[(songs['artist_id']==artist_name)].loc[:,['song_id','publish_time']]
        songs_publish_date=songs_publish_date.groupby('publish_time').size()
        songs_publish_date=songs_publish_date        
        
        
        
        
        rand_table=rand_song_artist_user_action[(rand_song_artist_user_action['artist_id']==artist_name)]
        rand_listen_table=rand_table[(rand_table['action_type']==1)]
        rand_listen_sum=rand_listen_table.groupby('Ds').size()
        
        
        
        
        frequent_table=frequent_song_artist_user_action[(frequent_song_artist_user_action['artist_id']==artist_name)]
        frequent_listen_table=frequent_table[(frequent_table['action_type']==1)]
        frequent_listen_sum=frequent_listen_table.groupby('Ds').size()

        #plot
        plt.figure()
        time1=pd.to_datetime('20150301',format='%Y%m%d')
        time2=pd.to_datetime('20150830',format='%Y%m%d')
        plt.xlim(time1,time2)
        plt.plot(rand_listen_sum,label='rand_listen')
        plt.plot(frequent_listen_sum,label='frequent_listen')
        plt.plot(songs_publish_date,'ro')
        plt.legend()
        plt.savefig('picture\\artist_daily_action'+str(artist_name)+'.jpg')
    
    return    
    
    
    
    
    
def split_user(user_actions):  
    user_total_op_num=user_actions.groupby('user_id').size()
    random_user=user_total_op_num[user_total_op_num<15]
    frequent_user=user_total_op_num[user_total_op_num>=15]
    return (random_user,frequent_user)
  
  
  
  
  
#此函数用于生成每个歌手的时间序列
def get_time_series(user_action,songs):
    rand_song_artist=songs.loc[:,['song_id','artist_id']]
    rand_song_artist_user_action=pd.merge(user_action,rand_song_artist,on='song_id')
    rand_song_artist_user_action.dropna(axis=0)
    
    artist_names=pd.unique(songs['artist_id'])
    rand_listen_sum=[]    
    cnt=0    
    
    #get time series
    #ts=rand_listen_table.groupby('Ds').index
    
    for artist_name in artist_names:
        rand_table=rand_song_artist_user_action[(rand_song_artist_user_action['artist_id']==artist_name)]
        rand_listen_table=rand_table[(rand_table['action_type']==1)]
        rand_listen_sum.append(rand_listen_table.groupby('Ds').size())
        cnt=cnt+1
    listen_sum={'listen_sum':rand_listen_sum,
                'artist_list':artist_names}
    
    return listen_sum
    
        
def medianBlur(series,w=3):
    result=[]
    series=np.array(series)
 #   a=series.copy()
    l=len(series)
    for i in range(l):
        if i<np.floor(w/2) or i>l-np.floor(w/2)-1:
            result.append(series[i])
        else:
            w_series=series[i-np.floor(w/2):i+np.floor(w/2)].copy()
            w_series.sort()
            mid=w_series[np.floor(w/2)]
            result.append(mid)
    return np.array(result)
#b=medianBlur(listen_sum[3],w=5)
def plot_2(series):
    x=range(183)
    plt.figure()
  #  plt.plot(np.array(series),label='raw')
    plt.plot(medianBlur(series,w=5),label='mid')
    plt.legend()
#for s in listen_sum:
#    plot_2(s)    
    
    
#读入数据
    
#time1 = time.time()
#uheader=['user_id','song_id','gmt_create','action_type','Ds']
#user_actions=pd.read_csv('data/p2_mars_tianchi_user_actions.csv',header=None,names=uheader)
#user_actions['Ds']=pd.to_datetime(user_actions['Ds'],format='%Y%m%d')
#sheader=['song_id','artist_id','publish_time','song_init_plays','Language','Gender']
#songs=pd.read_csv('data/p2_mars_tianchi_songs.csv',header=None,names=sheader)
#songs['publish_time']=pd.to_datetime(songs['publish_time'],format='%Y%m%d')
##ts=user_actions.groupby('Ds').index
#time2 = time.time()
#print  'read data:'
#print time2-time1
#artist_listen=get_time_series(user_actions,songs)



