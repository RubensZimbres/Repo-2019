import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
#pip install pandas-datareader

data=pd.read_csv('Lista_Acoes.csv',sep=';',header=0)
nomes=data.iloc[:,0]
nomes;

nomes2=[]
for i in range(0,len(nomes)):

    try:
        stock = str(nomes[i])
        source = 'yahoo'

        # Set date range (Google went public August 19, 2004)
        start = datetime.datetime(2010, 7, 18)
        end = datetime.datetime(2019, 7, 19)

        # Collect Google stock data
        goog_df = data.DataReader(stock, source, start, end)
        nomes2.append(stock)
    except:
        stock = str(nomes[i])+'.SA'
        nomes2.append(stock)
        
from pandas_datareader import data

data2=[]
import time

def extract(inputs):
    try:
        stock = inputs
        source = 'yahoo'

        start = datetime.datetime(2019, 1, 19)
        end = datetime.datetime(2019, 7, 19)
        goog_df = data.DataReader(stock, source, start, end)


        data2.append(list(goog_df['Adj Close']))
    except:
        data2.append([])
        print(stock,'error')
    return data2

cotacoes=list(map(extract,nomes2))

import requests
from bs4 import BeautifulSoup

PE = []
Exp1Y = []
MargemLucro = []
Crescimento4 = []
Debt_Equity = []
UmAno = []


for name in nomes2:

    page = requests.get('https://finance.yahoo.com/quote/{}'.format(name))

    soup = BeautifulSoup(page.text, "html")

    req=soup.find_all(attrs={"class": "Trsdu(0.3s)","data-reactid":"66"})

    for i in req:
        if i.get_text()!=[]:
            PE.append(i.get_text())
        else:
            PE.append(['--'])

for i in range(0,len(nomes2)):
    try:
        a=pd.Series(data2[i])
        a.plot(kind='line', grid=True, title='{} Adjusted Closes'.format(nomes[i]))
        plt.show()
        print(nomes2[i],'',max(data2[i])/min(data2[i]))
        print('PE',PE[i])
    except:
        pass
