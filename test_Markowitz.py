import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams


stockpoolfile = '1018/stock/stockpool20e.csv' #selected stock pool by Random Forest
def get_tradedate(stockpoolfile):
    stockpool = pd.read_csv(stockpoolfile, index_col=None)
    print(stockpool)
    tradedatess = list(stockpool.date) # input date xxxx/xx/xx not "xxxx-xx-xx"
    tradedates = []
    for i in tradedatess:
        if not i in tradedates:
            tradedates.append(i)
    return tradedates

def get_stock(date,stockpoolfile):
    stockpool = pd.read_csv(stockpoolfile, index_col=0, header=None)
    stock_hold = stockpool.loc[date,3] # input date xxxx/xx/xx not "xxxx-xx-xx"
    return list(stock_hold)



pricebook0 = pd.read_csv('dayopenclose2.csv', index_col=0)  #pre-saved open close price data
pricebook1 = pd.read_csv('dayopenclose2.csv', index_col=None)
pricebook = pricebook0.set_index('code', append=True)

def get_tradedays(pricebook):
    tradedays = list(set(pricebook.date))
    return tradedays
tradedays = get_tradedays(pricebook1)
print(tradedays)

def get_prices(tradedate, stocklist, pricebook):
    dic=[]
    for stock in stocklist:
        try:
            open_price=pricebook.loc[(tradedate,stock),'open']

            t = (tradedate, stock, open_price,)
            dic.append(t)
        except:
            open_price=888888
    print(dic)
    return dic

#def buystock(date, money, tradeprices):

dates=get_tradedate(stockpoolfile) #days of tradings
tradeprices=[]
for tradedate in dates:
    stocklist = get_stock(tradedate,stockpoolfile)
    tradepricess = get_prices(tradedate, stocklist, pricebook)
    tradeprices.extend(tradepricess)

tradeprices0 = pd.DataFrame(tradeprices, columns=['tradedate','code','open'])
print(tradeprices0)
tradelist1 = tradeprices0.set_index('tradedate')
print(tradelist1)
tradelist2 = tradelist1.set_index('code', append=True)
print(tradelist2)


def create_datelist(datestart , dateend):
	if datestart is None:
		datestart = '2016-01-01'
	if dateend is None:
		dateend = datetime.datetime.now().strftime('%Y-%m-%d')
	datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
	dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
	date_list = []
	date_list.append(datestart.strftime('%Y-%m-%d'))
	while datestart<dateend:
	    datestart+=datetime.timedelta(days=+1)
	    date_list.append(datestart.strftime('%Y-%m-%d'))
	return date_list

def update_holding_stock(date):
    stock0 = tradelist1.loc[date,'code']
    return stock0

def update_holding_amount(date, new_holding, money):
    if isinstance(new_holding,str):
        new_holding = [new_holding]
    value= money/len(new_holding)
    q=[]
    for stock in new_holding:
        price = tradelist2.loc[(date, stock,), 'open']
        quantity = value/price
        q.append(quantity)
    new_holding_amount = dict(zip(new_holding,q))
    return new_holding_amount


def wealth(date, holding_amount):
    wealth_se=[]
    for stock in list(holding_amount.keys()):
        price = pricebook.loc[(date, stock,), 'open']
        quantity = holding_amount[stock]
        value = price * quantity
        wealth_se.append(value)
    wealth= sum(wealth_se)
    return wealth
def get_HS300(date):
    HS1 = pd.read_csv('HS.csv')
    HS2 = HS1.set_index('date')
    HS300 = HS2.loc[date, 'open']
    return HS300

weight = [0.0118712, 0, 0, 0, 0.0319169, 0, 0.1786744, 0, 0, 0, 0, 0.0251533, 0, 0.3291843,
 0, 0.0061892, 0, 0, 0, 0, 0.1437864, 0.0618081, 0,      0,     0,   0,      0,      0.027787,
 0, 0, 0, 0,   0.0314599, 0, 0.067914,
 0, 0, 0.0842553, 0, 0,]  #markowizt weight porfolio

mk = ['SZ.300228','SH.601888','SH.600519','SZ.002318','SH.601628','SH.600389',
'SH.601318','SH.601009','SH.600340','SH.600385','SZ.000333','SH.601799',
'SH.603816','SH.600900','SZ.000001','SZ.300228','SZ.002015','SZ.002572',
'SZ.002594','SH.603986','SH.600036','SH.603696','SZ.300571','SZ.002310',
'SH.600104','SH.603031','SZ.000555','SH.601012','SH.600585','SH.600301',
'SH.600309','SZ.000333','SZ.002034','SH.603816','SZ.002776','SH.600519',
'SH.600585','SZ.002832','SH.600309','SZ.000526',]

period = create_datelist('2013-01-31' , '2018-12-28')
holding = {}
record=[]
HS = get_HS300('2013-01-31')
m=1
for day in period:
    if day in tradedays:
        if day in dates:
            if holding:
                HS300 = get_HS300(day)
                wealth0 = wealth(day, holding)
                new_holding = update_holding_stock(day)
                holding = update_holding_amount(day, new_holding, wealth0)
                record.append([day, wealth0/1000000, HS300/HS])
                print(wealth0,'换仓')


            else:
                wealth0 = 1000000
                new_holding = update_holding_stock(day)
                holding = update_holding_amount(day, new_holding, wealth0)
                print(wealth0, '买入')
                record.append([day, wealth0/1000000, HS/HS])

        else:
            HS300 = get_HS300(day)
            wealth0 = wealth(day, holding)
            print(wealth0, '持仓')
            record.append([day, wealth0/1000000, HS300/HS])

records = pd.DataFrame(record, columns=['date','return', 'HS300'])
print(records)
records1=records.set_index('date')
records1.to_csv('r20.csv')













