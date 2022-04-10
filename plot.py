import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

plt.rcParams['axes.unicode_minus'] = False
rcParams['font.family'] = 'STIXGeneral'
record=pd.read_csv('r40m.csv')
records1=record.set_index('date')

records1.plot(kind="line" ,figsize=(8, 6), title='Cumulative Return of Naive Portfolios with 1-Month and 2-Month Dynamic Positions (40 Stocks)', color=['dodgerblue','mediumorchid','sandybrown'], lw=0.8 )
plt.xticks(rotation=270)
plt.xlabel("Trading Date")
plt.ylabel("Cumulative Return")
#plt.legend(labels=['Markowitz Portfolio','Naive Portfolio','HS300'])
plt.legend(labels=['1-Month Dynamic Position','2-Month Dynamic Position','HS300'])
plt.axhline(y=1,c='salmon',ls='--',lw=1)
plt.tight_layout()
#plt.show()
plt.savefig('naive40m.png',dpi=600,bbox_inches = 'tight')
