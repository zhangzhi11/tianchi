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
from sklearn import linear_model,preprocessing
rcParams['figure.figsize']=15, 6

#对没有全的序列补零
def fill_zero_days(dta,a=1):
#填补没有信息的天数
    days= pd.date_range('3/1/2015',periods=183,freq='D')
    #这里为没有数值的设置为局部平均值  
    #ts = pd.Series(1,index = days)
    ts=pd.Series(a,index=days)   
    for day in dta.index:
        ts[day]=dta[day]      
    return ts

def split_train_predict(dta):#划分训练集和测试集
    #采用前五个月训练
    train_data = dta[0:122]
    #预测最后一个月的数据
    pred_data = dta[122:183]
    return (train_data,pred_data)

#计算月均值（好像在后面的函数里没有用过）
def pred_month_avg(train_data):
    month1=np.average(train_data[-60:-30])
    month2=np.average(train_data[-30:])
    sum1 =  month1 * 0.5 + month2 * 0.5 
    sum2 =  month2 * 0.5 + sum1 * 0.5
    return (sum1 ,sum2)

def linear_regression(data,cnt,pre_len):
    w_data=data[cnt:]       
    train_len=len(w_data)     
    regr = linear_model.LinearRegression()    
    #regr=linear_model.RANSACRegressor(min_samples=0.4)
    # Train the model using the training sets
    w_data=np.array(w_data)
    data_x=np.array([np.arange(0,train_len)]).T
    regr.fit(data_x,w_data)
    pre_x=np.array([np.arange(train_len,train_len+pre_len)]).T
    pre_y=regr.predict(pre_x)
    train_pre=regr.predict(data_x)
    return pre_y,train_pre   
    
def remove_outlier(data,win=5,a=2):
    #窗内的最高值取中位数
    data_len=len(data)
    #局部大于avg+a*var的点
    for i in range(win,data_len-win):
        win_data=data[i-win:i+win]
        wavg=np.average(win_data)
        var=np.sqrt(np.var(win_data))
        win_data[win_data>wavg+a*var]=wavg
   #     win_data[np.abs(win_data-wavg)>a*var]=wavg

        data[i-win:i+win]=win_data    

    return data              
#l=artist_listen['listen_sum']
#s=l[15]
#plt.figure()
#plt.plot(s)
#t=remove_outlier(s)
#plt.plot(t)           
def limit_linear_regression(data,pre_len,cnt=30,win=3):
    #要去掉一些异常值 
#    win=7
    a=2
    data=remove_outlier(data,win=7,a=3)
    rolmean = pd.rolling_mean(data, window=win)
    rolmean[0:win]=rolmean[win]  
    
    avg=np.average(rolmean[-30:])
    var=np.sqrt(np.var(rolmean[-30:]))
    

    pre_y,train_pre_y=linear_regression(rolmean,cnt,pre_len)
    pre_y[pre_y>avg+a*var]=avg+a*var
    pre_y[pre_y<avg-a*var]=avg-a*var
    pre_y[pre_y<=0]=1
    return pre_y,train_pre_y,cnt    
    
def sigmoid_regression(data,pre_len,cnt=5,win=5):
    data=remove_outlier(data,win=5,a=2)
    rolmean = pd.rolling_mean(data, window=win)
    w_data=rolmean[cnt:]
    train_len=len(w_data)
    sig_reg=linear_model.LogisticRegression()
    #将y值归一化到【0,1】
    min_max_scaler = preprocessing.MinMaxScaler()
    nor_data = min_max_scaler.fit_transform(w_data)
    
    data_x=np.array([np.arange(0,train_len)]).T    
    sig_reg.fit(data_x,nor_data)
    pre_x=np.array([np.arange(train_len,train_len+pre_len)]).T    
    #在预测集上的回归结果    
    pre_y=sig_reg.predict(pre_x)   
    pre_y=min_max_scaler.inverse_transform(pre_y)
    #在训练集上的回归结果
    train_pre_y=sig_reg.predict(data_x)
    train_pre_y=min_max_scaler.inverse_transform(train_pre_y)    
    return pre_y,train_pre_y
    
                

    
#对数据进行指数函数拟合
def exp_regression(data,cnt,base,pre_len=60):
    w_data=data[cnt:]
    w_data=w_data-base
    w_data[w_data<=1]=1   
    log_train_data=np.log(w_data)
    train_len=len(w_data) 
    
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    data_x=np.array([np.arange(0,train_len)]).T
    log_train_data=np.array(log_train_data)
    regr.fit(data_x,log_train_data)
    pre_x=np.array([np.arange(train_len,train_len+pre_len)]).T
    pre_y=regr.predict(pre_x)
    pre_y=np.exp(pre_y)+base
    return pre_y

#对数据进行pow函数的拟合
def pow_regression(data,cnt,base,pre_len=60):
    w_data=data[cnt:]
    w_data=remove_outlier(w_data,win=10,a=3)
    w_data=w_data-base
    w_data[w_data<=1]=1    
    log_train_data=np.log(w_data)
    train_len=len(w_data)    
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    data_x=np.log(np.array([np.arange(1,train_len+1)]).T)
    log_train_data=np.array(log_train_data)
    regr.fit(data_x,log_train_data)
    pre_x=np.log(np.array([np.arange(train_len+1,train_len+pre_len+1)]).T)
    #在预测时间段的回归结果    
    pre_y=regr.predict(pre_x)
    pre_y=np.exp(pre_y)+base
    #在训练时间段的回归结果
    train_pre_y=regr.predict(data_x)
    train_pre_y=np.exp(train_pre_y)+base    
    return pre_y,train_pre_y    
    
def find_start(train_data,limit=15,a=2):
    avg=np.average(train_data)
    var=np.var(train_data)  
    train_len=len(train_data)
    for i in range(train_len):
        cnt=train_len-i-1
        if train_data[cnt]>avg+3*np.sqrt(var):
            if cnt<train_len-15:                
                break  
#    for i in range(limit,train_len-limit):
#        cnt=-i-1
#        win_data=train_data[cnt-limit:cnt+limit]
#        avg=np.average(win_data)
#        var=np.var(win_data) 
#        if train_data[cnt]>avg+a*np.sqrt(var):
#            if cnt<train_len-10:                
#                break                  
        
    return cnt
    
def compute_error(data,pre):
    d_len=len(data)
    data=data.astype('float')
    abs_error=np.abs(data-pre)/data
    return np.sum(abs_error**2)/d_len
        
#计算两种预测的误差大小，1大取1,2大取0
def compare_error(train_data,train_pre1,train_pre2):
    #在最后的15天的数据中计算误差
    win=-15   
    data=train_data[win:]
    pre1=train_pre1[win:]
    pre2=train_pre2[win:]
    error1 = compute_error(data,pre1)
    error2 = compute_error(data,pre2)
    if error1>error2:
        return 0
    else:
        return 1

    
    
#def compute_regression_error()
#对一个序列进行预测
def pred(train_data,pre_len,win=3):
    
    rolmean = pd.rolling_mean(train_data, window=win)
    rolmean[0:win]=rolmean[win]
    sort_data=np.sort(rolmean)
    #sort_data=rolmean.sort(reverse = False)
    low_avg=np.average(sort_data[:30])    
    #找到开始预测的点cnt
    cnt=find_start(train_data)    
  
    pre_y1=np.zeros(pre_len)
    #pre_y2=np.zeros(pre_len)    
    
    if not cnt==0:
        #exp回归
       # wavg_data=rolmean[cnt:] #得到用于exp拟合的序列，这里需不需要一个平滑
        #pre_y1=exp_regression(rolmean,cnt,pre_len=60)
        pre_y1,train_pre_y1=pow_regression(rolmean,cnt,low_avg,pre_len)
    else:
        #linear回归 or logistic回归  
        #pre_y1,train_pre_y1,cnt=limit_linear_regression(rolmean,pre_len)
        cnt=5
        pre_y1,train_pre_y1=sigmoid_regression(rolmean,pre_len,cnt)  
    #线性预测  
    return pre_y1,train_pre_y1,cnt
def pred_compare(train_data,pre_len,win=7):
    
    rolmean = pd.rolling_mean(train_data, window=win)
    rolmean[0:win]=rolmean[win]
    sort_data=np.sort(rolmean)
    #sort_data=rolmean.sort(reverse = False)
    low_avg=np.average(sort_data[:30])    
    #找到开始预测的点cnt
    cnt=find_start(train_data)    
  
    pre_y1=np.zeros(pre_len)
    #pre_y2=np.zeros(pre_len)    
    

    #    pre_y1=exp_regression(rolmean,cnt,pre_len=60)
    pre_y1,train_pre_y1=pow_regression(rolmean,cnt,low_avg,pre_len) 
    #pre_y2,train_pre_y2,cnt1=limit_linear_regression(rolmean,pre_len)
    cnt1=30
    pre_y2,train_pre_y2=sigmoid_regression(rolmean,pre_len,cnt1)
    if compare_error(rolmean,train_pre_y1,train_pre_y2):
        return pre_y1,train_pre_y1,cnt
    else: 
        return pre_y2,train_pre_y2,cnt1
#结果画图
def show_result(dta,result,train_pre,cnt,i,status='offline'):
    
    if status=='offline':
        dta=pd.Series(dta)
        dta.index = pd.Index(range(183))
        plt.figure()  
        dta.plot(label='raw_data')
        
        result=pd.Series(result)
        result.index = pd.Index(range(122,183))
        result.plot(label='predict',color='r')
        
        train_pre=pd.Series(train_pre)
        train_pre.index = pd.Index(range(cnt,122))
        train_pre.plot(label='train_pre',color='g')
        
        plt.plot(cnt,0,'ro')
        plt.legend()        
    else:
        dta=pd.Series(dta)
        dta.index = pd.Index(range(183))
        plt.figure()  
        dta.plot(label='raw_data')    
        
        result=pd.Series(result)
        result.index = pd.Index(range(183,243))
        result.plot(label='predict',color='r')  
        
        train_pre=pd.Series(train_pre)
        train_pre.index = pd.Index(range(cnt,183))
        train_pre.plot(label='train_pre',color='g') 
        plt.legend()
 #   plt.savefig('picture\\regression_result\\hybird\\'+str(i)+'.jpg')  
    #plt.savefig('picture\\regression_result\\sigmoid1\\'+str(i)+'.jpg')
    return   

#测试单个序列    
def test_single(dta,i):
    dta = fill_zero_days(dta)	#将null项补零
       #     dta = regular_month(dta)	#月归一化到30天
    (train_data,pred_data) = split_train_predict(dta)  	#分割训练集和测试集
    ##预测
    result,train_pre,cnt = pred(train_data,pre_len=61)   

    show_result(dta,result,train_pre,cnt,i)
#test_single(listen_sum[1],1)

#测试全部序列
def test_all(listen_sum):
    total_num = len(listen_sum) #所有歌手的播放量的时间序列矩阵
    for i in range(total_num):
        dta = listen_sum[i]		#取出一个歌手的时间序列
        test_single(dta,i)
#test_all(listen_sum)
def show_online(listen_sum):
    total_num = len(listen_sum) #所有歌手的播放量的时间序列矩阵
    for i in range(total_num):
        dta = listen_sum[i]		#取出一个歌手的时间序列 
        dta = fill_zero_days(dta)	#将null项补零
        train_data=dta
        result,train_pre,cnt = pred(train_data,pre_len=60)
        show_result(train_data,result,train_pre,cnt,i,status='online')
#show_online(artist_listen['listen_sum'])
#线下预测
def get_expregress_All_results_offline(listen_sum): #计算所有的预测
    results = []
    preds = []
    total_num = len(listen_sum) #所有歌手的播放量的时间序列矩阵
    for i in range(total_num):
        dta = listen_sum[i]		#取出一个歌手的时间序列
        dta = fill_zero_days(dta)	#将null项补零
   #     dta = regular_month(dta)	#月归一化到30天
        (train_data,pred_data) = split_train_predict(dta)  	#分割训练集和测试集
        ##预测
        result,train_pre,cnt = pred(train_data,pre_len=61)
        results.append(result)
        #show_result(dta,result,cnt)    
        preds.append(pred_data)        
    return preds,results     

#线上预测
def get_expregress_All_results_online(listen_sum): #计算所有的预测
    results = []
    total_num = len(listen_sum) #所有歌手的播放量的时间序列矩阵
    for i in range(total_num):
        dta = listen_sum[i]		#取出一个歌手的时间序列
        dta = fill_zero_days(dta)	#将null项补零
   #     dta = regular_month(dta)	#月归一化到30天
        train_data=dta
        ##预测
        result,train_pre,cnt = pred(train_data,pre_len=60)
        #result[:3]=train_data[-1]
        results.append(result)
        #show_result(dta,result,cnt)          
    return results
#get_expregress_All_results_online(artist_listen['listen_sum'])
#测评代码
def eval(preds,results):
    singer_num = len (preds)
    sigmas = []
    phis = []
    F = 0
    for i in range(singer_num):
        pred = preds[i]
        result = results[i]
        day_num = len(pred)
        t_sqr_diff = 0
        t_play = 0
        for j in range(day_num):
            t_sqr_diff = t_sqr_diff + np.square((result[j]-pred[j])/pred[j])
            t_play = t_play + pred[j]
        sigma =  np.sqrt( t_sqr_diff/day_num)
        sigmas.append(sigma)
        phi = np.sqrt(t_play)
        phis.append(phi)
     
    for  i in range(singer_num):
        F = F + (1-sigmas[i])*phis[i]
    return F



#test_all(artist_listen['listen_sum'])
#preds,results=get_expregress_All_results_offline(artist_listen['listen_sum'])
###results1,results2=get_expregress_All_results_online(artist_listen['listen_sum'])
#F1=eval(preds,results)
#print '1预测F值为'
#print F1
