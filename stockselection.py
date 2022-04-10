#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import rqdatac



#获取交易日列表

def get_last_trading_day_of_month():
    import csv
    trade_month_list = []
    trademl=open('trade_month_list.csv')
    trade_month_listl = list(csv.reader(trademl))
    for i in trade_month_listl:
        for j in i:
            trade_month_list.append(j)
    print(trade_month_list)

    return trade_month_list

def get_close_prices(tradeDate_list):
    """
    获取全部股票的收盘价
    输入：
        tradeDate_list：list，交易日
    返回：
        DataFrame：index为‘date’，columns为order_book_id，value为close—price
    """
    close = pd.DataFrame()
    for date in trade_month_list:
        temp_univ = rqdatac.all_instruments(type='CS',date=date)['order_book_id'].tolist()
        temp_close = rqdatac.get_price(temp_univ, start_date=date, end_date=date, fields='close')
        temp_close_new =temp_close.reset_index()
        close = close.append(temp_close_new)
    return close

p = get_close_prices(trade_month_list)
p.to_csv('close_prices_a_monthly.csv')

def get_fac(tradeDate_list, fac):
    print ('get factor:')
    factor = pd.DataFrame()
    for date in tradeDate_list:
        univ = rqdatac.all_instruments(type='CS',date=date)['order_book_id'].tolist()
        temp = rqdatac.get_factor(univ,factor=fac,start_date=date,end_date=date)
        temp_new =temp.reset_index()
        factor = factor.append(temp_new)
    return factor


def get_industry(tradeDate_list, source_='sws'):
    ind_df = pd.DataFrame()
    for date in tradeDate_list:
        univ = rqdatac.all_instruments(type='CS',date=date)['order_book_id'].tolist()
        univ = del_suspended(univ, date)
        univ = del_st(univ, date)
        univ = del_new(univ, date, day = 60)

        univ_ind = rqdatac.get_instrument_industry(univ, source=source_, date=date)
        univ_ind.reset_index(inplace=True)
        univ_ind['date'] = date
        ind_df = ind_df.append(univ_ind)

    ind_df = ind_df[['date','order_book_id', 'first_industry_name']]
    ind_df.columns = ['date', 'order_book_id', 'industry']
    ind_df.set_index(['date','order_book_id'], inplace=True)
    return ind_df


def del_st(univ, date):
    """
    剔除ST的股票
    输入：
        univ：list，股票池
        date：str，日期
    返回：
        remove_st_univ：list，处理后的股票池
    """
    st_df = rqdatac.is_st_stock(univ, start_date=date, end_date=date)
    remove_st_univ = st_df[st_df == False].columns.tolist()
    return remove_st_univ


def del_suspended(univ, date):
    """
    剔除停牌，新上市，退市的股票
    输入：
        univ：list，股票池
        date：str，日期
    返回：
        remove_susp_univ：list，处理后的股票池
    """
    suspended_df = rqdatac.is_suspended(univ, start_date=date, end_date=date)   
    remove_susp_univ = suspended_df[suspended_df == False].columns.tolist()
    return remove_susp_univ


def del_st(univ, date):
    st_df = rqdatac.is_st_stock(univ, start_date=date, end_date=date)
    remove_st_univ = st_df[st_df == False].columns.tolist()
    return remove_st_univ
def del_suspended(univ, date):
    suspended_df = rqdatac.is_suspended(univ, start_date=date, end_date=date)   
    remove_susp_univ = suspended_df[suspended_df == False].columns.tolist()
    return remove_susp_univ
def del_new(univ, date, day = 60):
    remove_new_univ = []
    for stk in univ:
        listDate = rqdatac.instruments(stk).days_from_listed(date)
        
        if listDate > day:
            remove_new_univ.append(stk)
    return remove_new_univ


def del_new(univ, date, day = 60):
    """
    剔除某个日期前多少个交易日,之后上市的新股
    输入：
        univ：list，股票池
        date：str，日期
        day：floor，新股的天数
    返回：
        remove_new_univ：list，处理后的股票池
    """
    remove_new_univ = []
    for stk in univ:
        listDate = rqdatac.instruments(stk).days_from_listed(date)
        
        if listDate > day:
            remove_new_univ.append(stk)
    return remove_new_univ





# In[42]:
#ind_df = pd.read_csv('industry_a_sws_monthly.csv', index_col=0)




def fac_pct_process(factor_name, data):
    """
    对因子进行行业内百分比转化
    输入：
        factor_name: 需要进行处理的因子名
        data: 某日的原始因子数据
    返回：
        data: 对指定factor_name做完处理的因子数据
    """
    data_pct = pd.DataFrame()
    for ind_name in data['industry'].unique():

        temp_data = data[data['industry'] == ind_name].copy() 
        temp_data[factor_name] = temp_data[factor_name].rank() / len(temp_data[factor_name])
        data_pct = data_pct.append(temp_data)
        
    data_pct.reset_index(drop=True ,inplace=True)
    return data_pct



def label_by_quantile(r, quantiles=(0.2, 0.8)): 
    lower_bound, upper_bound = r.quantile(quantiles)
    label = np.repeat(np.nan, len(r))
    label[r > upper_bound] = 1
    label[r < upper_bound] = 0
    return label


def label_return(data, label_func=label_by_quantile):
    label = (data['forward_return']
             .groupby(level='date')
             .transform(label_func)
             .dropna())
    return label


def training_window(rebalance_date, window, listname = get_last_trading_day_of_month()):
    """给定一个预测日，和一个固定的回溯窗口，返回所用数据集的开始和结束日期。
    这一对开始结束日用于从样本数据中切分出相应的数据作为训练集"""
    ind = listname.index(rebalance_date)
    end = listname[ind-1] # 换仓日当月月初
    start = listname[ind-window]
    # start 和 end 都是月初, 虽然不一定是交易日, 但在月末换仓的假设下
    # 能保证必定覆盖 `window` 个月, 同时rebalance_date不在窗口内
    return start, end


def get_training_dataset(rebalance_date, data, window):
    start, end = training_window(rebalance_date, window)
    data = (
        data
        .loc[start:end]
    )
    label = label_return(data)
    
    features = data.drop(['forward_return'], axis=1)
    label, features = label.align(features, join='inner')
    return features, label

def get_prediction_dataset(rebalance_date, data):
    data = (
        data
        .loc[rebalance_date]
    )
    features = data.drop(['forward_return'], axis=1)
    return features

#setting random forest optimizer with specified parameters.

def optimize(rebalance_date, data, window):
    param_grid = {'n_estimators': [30, 40, 50],
              'max_depth': [2, 3, 4, 5, 6, 7, 8, 9]}
    model = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    X, y = get_training_dataset(pd.Timestamp(rebalance_date), data, window)
    model.fit(X, y)
    return model.best_params_
    
def make_model(rebalance_date, data, window, n_estimators=50, max_depth=9,):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    X, y = get_training_dataset(rebalance_date, data, window)
    model.fit(X, y)
    return model

def make_factor(rebalance_date, model, data):
    features = get_prediction_dataset(rebalance_date, data) #换仓日数据
    values =  model.predict_proba(features)[:,1] #输入因子预测01,输出预测1的概率
    index = pd.MultiIndex.from_product([[rebalance_date], features.index], #换仓日、代码生成多索引
                                       names=['date', 'order_book_id'])
    
    res = pd.Series(values, index)
    return res

def get_feature_importances(model, names):
    value = model.feature_importances_
    return pd.Series(value, index=names)








