import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams


stockpoolfile = '1018/stock/stockpool40e.csv'
def get_tradedate(stockpoolfile):
    stockpool = pd.read_csv(stockpoolfile, index_col=None)
    print(stockpool)
    tradedatess = list(stockpool.date) # date是/
    tradedates = []
    for i in tradedatess:
        if not i in tradedates:
            tradedates.append(i)
    return tradedates

def get_stock(date,stockpoolfile):
    stockpool = pd.read_csv(stockpoolfile, index_col=0, header=None)
    stock_hold = stockpool.loc[date,3] #date是/
    return list(stock_hold)



pricebook0 = pd.read_csv('dayopenclose2.csv', index_col=0)
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

dates=get_tradedate(stockpoolfile) #换仓日期
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


period = create_datelist('2013-01-31' , '2018-12-28')
holding = {}
record=[]
HS = get_HS300('2013-01-31')
m=1
for day in period:
    if day in tradedays:
        if day in dates:
            if holding:
                n=(-1)^m
                if n==-1:
                    HS300 = get_HS300(day)
                    wealth0 = wealth(day, holding)
                    new_holding = update_holding_stock(day)
                    holding = update_holding_amount(day, new_holding, wealth0)
                    record.append([day, wealth0/1000000, HS300/HS])
                    print(wealth0,'换仓')
                    m=m+1
                else:
                    HS300 = get_HS300(day)
                    wealth0 = wealth(day, holding)
                    print(wealth0, '持仓')
                    record.append([day, wealth0 / 1000000, HS300 / HS])

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
records1.to_csv('40-2month.csv')

plt.rcParams['axes.unicode_minus'] = False
rcParams['font.family'] = 'STIXGeneral'
records1.plot(kind="line" ,figsize=(8, 6), title='Cumulative Return on 2-Month Rolling Positions (40 stocks)', color=['dodgerblue','sandybrown'], lw=1.2 )
plt.xticks(rotation=270)
plt.xlabel("Trading Date")
plt.ylabel("Cumulative Return")
plt.legend(labels=['Cumulative Return','HS300'])
plt.axhline(y=1,c='salmon',ls='--',lw=1)
plt.tight_layout()
plt.show()
#plt.savefig('5-2.png',dpi=1000,bbox_inches = 'tight')

#plt.subplots_adjust(left=0.3, bottom=0.5, right=0.8,)
#plt.gcf().subplots_adjust(left=0.2,top=0.91,bottom=0.2)












