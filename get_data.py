#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta, date
import time
import statsmodels.api as sm
import scipy.stats as st
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import itertools
from functools import partial
import rqdatac

pd.set_option('display.max_columns', None)   
sns.set_style('white')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  
rqdatac.init()


# In[2]:


#获取交易日列表
start_1 = '20100101'
end_1 = '20200331'
def get_last_trading_day_of_month(start_date, end_date):
    """
    获取月末的交易日列表
    输入：
       start_date：str，datetime.datetime
       end_date：str，datetime.datetime
    返回：
        list：月末的交易日列表
    """   
    all_trading_days = rqdatac.get_trading_dates(start_date, end_date)
    
    month_ends = []
    for i, day in enumerate(all_trading_days[:-1]):
        # 若本交易日的月份 != 下一交易日的月份, 则当日为本月最后一个交易日
        # 只能用 != 判断, 不能用 <, 否则会漏掉本年12月到明年1月的情况
        next_day = all_trading_days[i + 1]
        if day.month != next_day.month:
            month_ends.append(day)
    return month_ends
trade_month_list = get_last_trading_day_of_month(start_1, end_1)
trade_month_list = [date.strftime('%Y-%m-%d') for date in trade_month_list]
print(trade_month_list)
trade_month_list.to_csv('trade_month_list.csv')


# In[3]:
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
    print ('get factor:', fac)
    factor = pd.DataFrame()
    for date in tradeDate_list:
        univ = rqdatac.all_instruments(type='CS',date=date)['order_book_id'].tolist()
        temp = rqdatac.get_factor(univ,factor=fac,start_date=date,end_date=date)
        temp_new =temp.reset_index()
        factor = factor.append(temp_new)     
    return factor

#get data from ricequant platform and normalized and save then

# 存货周转率 inventory_turnover （财务因子）
factor = get_fac(trade_month_list, 'inventory_turnover')
factor.to_csv('factor_inventory_turnover_monthly.csv')

# 应收账款周转率 account_receivable_turnover_rate（财务因子）
factor = get_fac(trade_month_list, 'account_receivable_turnover_rate')
factor.to_csv('factor_account_receivable_turnover_rate_monthly.csv')

# 营业成本率 total_operating_cost_to_total_operating_revenueTTM（财务因子）
factor = get_fac(trade_month_list, 'total_operating_cost_to_total_operating_revenueTTM')
factor.to_csv('factor_total_operating_cost_to_total_operating_revenueTTM_monthly.csv')
# In[11]:

# 流动负债 / 负债合计 current_debt_to_total_debt（财务因子）
factor = get_fac(trade_month_list, 'current_debt_to_total_debt')
factor.to_csv('factor_current_debt_to_total_debt_monthly.csv')

# 总资产净利率ttm return_on_assetTTM（财务因子）
factor = get_fac(trade_month_list, 'return_on_assetTTM')
factor.to_csv('factor_return_on_assetTTM_monthly.csv')

# 递延所得税负债 deferred_income_tax_liabilities（财务因子）
factor = get_fac(trade_month_list, 'deferred_income_tax_liabilities')
factor.to_csv('factor_deferred_income_tax_liabilities_monthly.csv')

# 流通市值 
factor = get_fac(trade_month_list, 'a_share_market_val_2')
factor.to_csv('factor_share_market_val_2_monthly.csv')

# ROE （财务因子）
factor = get_fac(trade_month_list, 'return_on_equity')
factor.to_csv('factor_return_on_equity_monthly.csv')

# 净利润(同比增长率) （成长因子）
factor = get_fac(trade_month_list, 'inc_net_profit')
factor.to_csv('factor_inc_net_profit_monthly.csv')

# PE （估值因子）
factor = get_fac(trade_month_list, 'pe_ratio')
factor.to_csv('factor_pe_ratio_monthly.csv')

# 期间费率  [营业费用（TTM）+管理费用（TTM）+财务费用（TTM）]/营业收入（TTM））*100% （财务因子）
factor = get_fac(trade_month_list, 'period_costs_rateTTM')
factor.to_csv('factor_period_costs_rateTTM_monthly.csv')

# MA20 （技术因子）
factor = get_fac(trade_month_list, 'MA20')
factor.to_csv('factor_MA20.csv')


# MA60 （技术因子）
factor = get_fac(trade_month_list, 'MA60')
factor.to_csv('factor_MA60.csv')


# MACD_DIFF （技术因子）
factor = get_fac(trade_month_list, 'MACD_DIFF')
factor.to_csv('MACD_DIFF.csv')


# 市盈率ttm （估值因子）
factor = get_fac(trade_month_list, 'pe_ratio_ttm')
factor.to_csv('pe_ratio_ttm.csv')

# 市销率ttm（估值因子）
factor = get_fac(trade_month_list, 'ps_ratio_ttm')
factor.to_csv('ps_ratio_ttm.csv')


# 企业倍数（EV/EBITDA）（估值因子）
factor = get_fac(trade_month_list, 'ev_to_ebitda_ttm')
factor.to_csv('ev_to_ebitda_ttm.csv')


# 息税前利润（财务因子）
factor = get_fac(trade_month_list, 'ebit_ttm')
factor.to_csv('ebit_ttm')


# 营收增长率（成长因子）
factor = get_fac(trade_month_list, 'inc_revenue_ttm')
factor.to_csv('inc_revenue_ttm')


# 经营现金流增长率（成长因子）
factor = get_fac(trade_month_list, 'net_operate_cash_flow_growth_ratio_ttm')
factor.to_csv('net_operate_cash_flow_growth_ratio_ttm')


#相对强弱指标（技术因子）
factor = get_fac(trade_month_list, 'RSI10')
factor.to_csv('RSI10')


#多空指标（技术因子）
factor = get_fac(trade_month_list, 'BBI')
factor.to_csv('BBI')


# 随机波动指标（技术因子）
factor = get_fac(trade_month_list, 'KDJ_K')
factor.to_csv('KDJ_K')


# 平均换手率
factor = get_fac(trade_month_list, 'VOL20')
factor.to_csv('VOL20')


# 人气意愿指标
factor = get_fac(trade_month_list, 'AR')
factor.to_csv('AR')


# 布林带
factor = get_fac(trade_month_list, 'BOLL')
factor.to_csv('BOLL')





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


# In[35]:


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


# In[ ]:


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


# In[36]:


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



# 财务因子
IT = pd.read_csv('factor_inventory_turnover_monthly.csv', index_col=0) #存货周转率
ART = pd.read_csv('factor_account_receivable_turnover_rate_monthly.csv', index_col=0) #应收账款周转率
TOC = pd.read_csv('factor_total_operating_cost_to_total_operating_revenueTTM_monthly.csv', index_col=0) #营业成本率
CDT = pd.read_csv('factor_current_debt_to_total_debt_monthly.csv', index_col=0) #流动负债/负债
ROA = pd.read_csv('factor_return_on_assetTTM_monthly.csv', index_col=0) #总资产净利率
ROE = pd.read_csv('factor_return_on_equity_monthly.csv', index_col=0) #净资产收益率
PCR = pd.read_csv('factor_period_costs_rateTTM_monthly.csv', index_col=0) #期间费用率
EBIT = pd.read_csv('ebit_ttm', index_col=0) #毛利率

# 市值因子
MKV = pd.read_csv('factor_share_market_val_2_monthly.csv', index_col=0) #市值

#成长因子
NPgrowth = pd.read_csv('factor_inc_net_profit_monthly.csv', index_col=0) #净利润增长率
Revgrowth = pd.read_csv('inc_revenue_ttm', index_col=0) #营收增长率
NOCgrowth = pd.read_csv('net_operate_cash_flow_growth_ratio_ttm', index_col=0) #经营性现金流增长率

# 估值水平因子
PE = pd.read_csv('factor_pe_ratio_monthly.csv', index_col=0) #市盈率
PS = pd.read_csv('ps_ratio_ttm.csv', index_col=0) #市销率
EVtEBITDA = pd.read_csv('ev_to_ebitda_ttm.csv', index_col=0) #企业倍数

# 技术因子
MA20 = pd.read_csv('factor_MA20.csv', index_col=0) #20日均值
MA60 = pd.read_csv('factor_MA60.csv', index_col=0) #60日均值
MACD = pd.read_csv('MACD_DIFF.csv', index_col=0) #60日均值
RSI = pd.read_csv('RSI10', index_col=0) #相对强弱指标
BBI = pd.read_csv('BBI', index_col=0) #多空指标
KDJ = pd.read_csv('KDJ_K', index_col=0) #随机波动指标
AR = pd.read_csv('AR', index_col=0) #人气意愿指标
BOLL = pd.read_csv('BOLL', index_col=0) #布林带


# In[38]:


ind_df = pd.DataFrame()
for date in trade_month_list:
    univ = rqdatac.all_instruments(type='CS',date=date)['order_book_id'].tolist()
    univ = del_suspended(univ, date)
    univ = del_st(univ, date)
    univ = del_new(univ, date, day = 60)

    univ_ind = rqdatac.get_instrument_industry(univ, source='sws', date=date)
    univ_ind.reset_index(inplace=True)
    univ_ind['date'] = date
    ind_df = ind_df.append(univ_ind)
ind_df = ind_df[['date','order_book_id', 'first_industry_name']]

ind_df.columns = ['date', 'order_book_id', 'industry']
ind_df.set_index(['date','order_book_id'], inplace=True)


# In[39]:


ind_df.head()


# In[40]:


ind_df.to_csv('industry_a_sws_monthly.csv')


# In[41]:


afactor = pd.merge(IT, ART, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
bfactor = pd.merge(afactor, TOC, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
cfactor = pd.merge(bfactor, CDT, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
dfactor = pd.merge(cfactor, ROA, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
efactor = pd.merge(dfactor, ROE, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
ffactor = pd.merge(efactor, PCR, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
gfactor = pd.merge(ffactor, EBIT, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
hfactor = pd.merge(dfactor, MKV, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
ifactor = pd.merge(hfactor, NPgrowth, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
jfactor = pd.merge(ifactor, Revgrowth, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
kfactor = pd.merge(jfactor, NOCgrowth, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
lfactor = pd.merge(kfactor, PE, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
mfactor = pd.merge(lfactor, PS, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
nfactor = pd.merge(mfactor, EVtEBITDA, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
ofactor = pd.merge(nfactor, MA20, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
pfactor = pd.merge(ofactor, MA60, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
qfactor = pd.merge(pfactor, MACD, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
rfactor = pd.merge(qfactor, RSI, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
sfactor = pd.merge(rfactor, BBI, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
tfactor = pd.merge(sfactor, KDJ, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
ufactor = pd.merge(tfactor, AR, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')
all_factor = pd.merge(ufactor, BOLL, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')


# In[42]:


all_factor = pd.merge(all_factor, ind_df, 
                      left_on=['date', 'order_book_id'], right_on=['date', 'order_book_id'], how='inner')

all_factor.reset_index



# In[46]:


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


# In[47]:


all_factor_process = []

factor_list = ['inventory_turnover','account_receivable_turnover_rate','total_operating_cost_to_total_operating_revenueTTM',
               'current_debt_to_total_debt','return_on_assetTTM','a_share_market_val_2','inc_net_profit',
               'inc_revenue_ttm','net_operate_cash_flow_growth_ratio_ttm','pe_ratio','ps_ratio_ttm','ev_to_ebitda_ttm',
               'MA20','MA60','MACD_DIFF','RSI10','BBI','KDJ_K','AR','BOLL','industry']
               
date_list = sorted(all_factor['date'].unique())
for date in date_list:
    
    tdata = all_factor[all_factor['date'] == date]
    tdata = tdata.dropna(how='all')
    tdata.reset_index(drop=True ,inplace=True)
    
    for factor_name in factor_list:
        tdata = fac_pct_process(factor_name, tdata)
    all_factor_process.append(tdata)

all_factor_process = pd.concat(all_factor_process)
      
all_factor_process.sort_values(by=['date', 'order_book_id'])
all_factor_process.reset_index(drop=True, inplace=True)
all_factor_process.fillna(0, inplace=True)

all_factor_process.to_csv('factor_process.csv', index=False)


# In[48]:


all_factor = pd.read_csv('factor_process.csv')



# In[50]:


cp_a = pd.read_csv('close_prices_a_monthly.csv',header = 0)




ind = set(list(cp_a['order_book_id']))


# In[54]:




ft = pd.DataFrame()
for i in ind:
    temp = cp_a[cp_a['order_book_id']==i]
    temp['return'] = temp['close'].pct_change()
    temp['forward_return'] = temp['return'].shift(-1)
    ft = pd.concat([ft,temp])


# In[56]:


ft.head()


# In[57]:


ft.to_csv('price_related.csv')


len(all_factor)


# In[59]:


del ft['return']
del ft['close']


# In[60]:


raw_data = pd.merge(all_factor, ft, how='left')




raw_data.to_csv('raw_data.csv')




raw_data.head()





import os
import pickle
from collections import OrderedDict
import pandas as pd
from IPython import get_ipython
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


raw_data = pd.read_csv('raw_data.csv', index_col=0)

data = raw_data.set_index(['date', 'order_book_id'])

# In[67]:
del data['industry']


def label_by_quantile(r, quantiles=(0.2, 0.8)): 
    lower_bound, upper_bound = r.quantile(quantiles)
    label = pd.np.repeat(pd.np.nan, len(r))
    label[r > upper_bound] = 1
    label[r < upper_bound] = 0
    return label


def label_return(data, label_func=label_by_quantile):
    label = (data['forward_return']
             .groupby(level='date')
             .transform(label_func)
             .dropna())
    return label


# In[69]:


start_1 = '20100101'
end_1 = '20181201'
def get_last_trading_day_of_month(start_date, end_date):
    """
    获取月末的交易日列表
    输入：
       start_date：str，datetime.datetime
       end_date：str，datetime.datetime
    返回：
        list：月末的交易日列表
    """
    all_trading_days = rqdatac.get_trading_dates(start_date, end_date)
    
    month_ends = []
    for i, day in enumerate(all_trading_days[:-1]):
        # 若本交易日的月份 != 下一交易日的月份, 则当日为本月最后一个交易日
        # 只能用 != 判断, 不能用 <, 否则会漏掉本年12月到明年1月的情况
        next_day = all_trading_days[i + 1]
        if day.month != next_day.month:
            month_ends.append(day)
    return month_ends
trade_month_list = get_last_trading_day_of_month(start_1, end_1)
trade_month_list = [date.strftime('%Y-%m-%d') for date in trade_month_list]


def training_window(rebalance_date, window, listname = trade_month_list):
    """给定一个预测日，和一个固定的回溯窗口，返回所用数据集的开始和结束日期。
    这一对开始结束日用于从样本数据中切分出相应的数据作为训练集"""
    ind = listname.index(rebalance_date)
    end = trade_month_list[ind-1] # 换仓日当月月初
    start = trade_month_list[ind-window]
    # start 和 end 都是月初, 虽然不一定是交易日, 但在月末换仓的假设下
    # 能保证必定覆盖 `window` 个月, 同时rebalance_date不在窗口内
    return start, end


def get_training_dataset(rebalance_date, data, window=36):
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


# In[71]:


def optimize(rebalance_date):
    param_grid = {'n_estimators': [30, 40, 50],
              'max_depth': [2, 3, 4, 5, 6, 7, 8, 9]}
    model = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    X, y = get_training_dataset(pd.Timestamp(rebalance_date), raw_data)
    model.fit(X, y)
    return model.best_params_
    
def make_model(rebalance_date, data, n_estimators=50, max_depth=9):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    
    X, y = get_training_dataset(rebalance_date, data)
    model.fit(X, y)
    return model


def make_factor(rebalance_date, model, data):
    features = get_prediction_dataset(rebalance_date, data)
    
    values =  model.predict_proba(features)[:,1]
    index = pd.MultiIndex.from_product([[rebalance_date], features.index],
                                       names=['date', 'order_book_id'])
    
    res = pd.Series(values, index)
    return res


# In[72]:




# In[73]:



# In[74]:










# In[79]:












# In[90]:



# In[91]:








feature_cols = ['inventory_turnover','account_receivable_turnover_rate',
                'total_operating_cost_to_total_operating_revenueTTM','current_debt_to_total_debt',
                'return_on_assetTTM','a_share_market_val_2','inc_net_profit','inc_revenue_ttm',
                'net_operate_cash_flow_growth_ratio_ttm','pe_ratio','ps_ratio_ttm',
                'ev_to_ebitda_ttm','MA20','MA60','MACD_DIFF','RSI10','BBI','KDJ_K','AR','BOLL','industry']


# In[94]:







