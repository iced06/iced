#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
import ftplib
import io
import ta as ta
from ta.volatility import BollingerBands
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from termcolor import colored
import requests, time, urllib3, xlsxwriter
from datetime import datetime
from tqdm import tqdm
import mplfinance as mpf
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_auth
import plotly.subplots as ms


# In[2]:


from dash.dependencies import Input,Output


# In[3]:


app =dash.Dash()


# In[4]:


server = app.server


# In[ ]:





# In[5]:


plt.rcParams.update({'figure.max_open_warning': 0})


# In[6]:


today = date.today()
print("Today's date:", today)


# In[ ]:





# In[7]:


IMKBListe={"DOHOL.IS":"Holding","EGSER.IS":"Bina Malzemeleri","ERBOS.IS":"Bina Malzemleri","SISE.IS":"Bina Malzemeleri",
           "USAK.IS":"Bina Malzemeleri","ALCAR.IS":"Bina Malzemeleri","BRSAN.IS":"Bina Malzemeleri","CUSAN.IS":"Bina Malzemeleri",
           "DOGUB.IS":"Bina Malzemeleri","KLKIM.IS":"Bina Malzemeleri","KLMSN.IS":"Bina Malzemeleri","SAFKR.IS":"Bina Malzemeleri",
           "SISE.IS":"Bina Malzemeleri","ENJSA.IS":"Enerji","GWIND.IS":"Enerji","ODAS.IS":"Enerji","BUCIM.IS":"Çimento","CIMSA.IS":"Çimento",
           "OYAKC.IS":"Çimento","GOLTS.IS":"Çimento","NUHCM.IS":"Çimento","ARCLK.IS":"Beyaz Eşya","VESBE.IS":"Beyaz Eşya","EREGL.IS":"Demir Çelik",
           "KRDMD.IS":"Demir Çelik","TUCLK.IS":"Demir Çelik","YKSLN.IS":"Demir Çelik","AYEN.IS":"Enerji","FRIGO.IS":"Gıda",
           "PETUN.IS":"Gıda","TUKAS.IS":"Gıda","YYLGD.IS":"Gıda","AEFES.IS":"Gıda","CCOLA.IS":"Gıda","EKIZ.IS":"Gıda","KERVT.IS":"Gıda",
           "KNFRT.IS":"Gıda","MERKO.IS":"Gıda","PENGD.IS":"Gıda","TATGD.IS":"Gıda","ALARK.IS":"Holding","BRYAT.IS":"Holding","DOHOL.IS":"Holding",
           "GSDHO.IS":"Holding","KCHOL.IS":"Holding","SAHOL.IS":"Holding","TKFEN.IS":"Holding","AGHOL.IS":"Holding","ALKA.IS":"Kağıt",
           "BAKAB.IS":"Kağıt","KARTN.IS":"Kağıt","MNDTR.IS":"Kağıt","TCELL.IS":"Mobil","DOAS.IS":"Otomotiv",
           "FROTO.IS":"Otomotiv","OTKAR.IS":"Otomotiv","TOASO.IS":"Otomotiv","TTRAK.IS":"Otomotiv","CEMAS.IS":"Oto Malz","BRISA.IS":"Oto Malz",
           "CELHA.IS":"Oto Malz","CEMTS.IS":"Oto Malz","DITAS.IS":"Oto Malz","GOODY.IS":"Oto Malz","JANTS.IS":"Oto Malz","AKSA.IS":"Kimya",
           "ALKIM.IS":"Kimya","DYOBY.IS":"Kimya","EGPRO.IS":"Kimya","KORDS.IS":"Kimya","BOSSA.IS":"Tekstil","DAGI.IS":"Tekstil","DERIM.IS":"Tekstil",
           "DESA.IS":"Tekstil","KRTEK.IS":"Tekstil","MNDRS.IS":"Tekstil","YATAS.IS":"Tekstil","YUNSA.IS":"Tekstil","ALCTL.IS":"Teknoloji",
          "DGATE.IS":"Teknoloji","DESPC.IS":"Tekstil","FONET.IS":"Teknoloji","INDES.IS":"Teknoloji","LINK.IS":"Teknoloji","TKNSA.IS":"Teknoloji",
          "AYCES.IS":"Turizm","MAALT.IS":"Turizm","PKENT.IS":"Turizm","PGSUS.IS":"Ulastirma","THYAO.IS":"Ulastirma","ALGYO.IS":"Gayrimenkul",
          "EKGYO.IS":"Gayrimenkul","OZKGY.IS":"Gayrimenkul","BNTAS.IS":"Diger","DGNMO.IS":"Diger","MAVI.IS":"Diger","ORGE.IS":"Diger",
           "VAKKO.IS":"Diger","LKMNH.IS":"Saglık","BEYAZ.IS":"Diger","MEGAP.IS":"Kimya","METUR.IS":"Turizm",
          "PARSN.IS":"Otomotiv","LIDFA.IS":"Factoring"}


# In[8]:


IMKB=IMKBListe.keys()
IMKB


# In[9]:


HisseAdet=0
for i in IMKB:
    HisseAdet+=1


# In[10]:


HisseAl=[]
q=0
for a in IMKB:
    q+=1
    Hisse=yf.download(a,
                     start="2021-01-01",
                     end=today,
                     progress=False)
    Hisse.index=pd.to_datetime(Hisse.index)
    Hisse["Return"]=Hisse["Close"].diff()
    Hisse["Return_pct"]=Hisse["Close"].pct_change()
    Hisse["Target_Cls"]=np.where(Hisse.Return>0,1,0)
    Hisse["Vol_diff"]=Hisse["Volume"].diff()
    Hisse["Vol_change"]=Hisse["Volume"].pct_change()
    indicator_bb= BollingerBands(close=Hisse["Close"],window=20,window_dev=2)
    Hisse["bb_bbm"]=indicator_bb.bollinger_mavg()
    Hisse["bb_bbh"]=indicator_bb.bollinger_hband()
    Hisse["bb_bbl"]=indicator_bb.bollinger_lband()
    Hisse["MACD"] = ta.trend.macd(Hisse["Close"], window_slow = 26, window_fast= 12, fillna=False)
    Hisse["MACDS"] = ta.trend.macd_signal(Hisse["Close"], window_sign= 9, fillna=False)
    Hisse["Buy_MACD"]=np.where((Hisse["MACD"]>Hisse["MACDS"]),1,0)
    Hisse["Buy_MACDS"]=np.where((Hisse["Buy_MACD"]>Hisse["Buy_MACD"].shift(1)),1,0)
    Hisse['OBV'] = ta.volume.on_balance_volume(Hisse['Close'], Hisse['Volume'])
    Hisse["RSI"]=ta.momentum.rsi(Hisse["Close"],window= 14, fillna= False)
    Hisse["Buy_RSI"]=np.where((Hisse["RSI"]>30),1,0)
    Hisse["Buy_RSIS"]=np.where((Hisse["Buy_RSI"]>Hisse["Buy_RSI"].shift(1)),1,0)
    Hisse["AO"]=ta.momentum.awesome_oscillator(Hisse["High"],Hisse["Low"],window1=5,window2=34,fillna=True)
    Hisse["Buy_AO"]=np.where((Hisse["AO"]>0),1,0)
    Hisse["Buy_AOS"]=np.where((Hisse["Buy_AO"]>Hisse["Buy_AO"].shift(1)),1,0)
    Hisse["CCI"]=ta.trend.cci(Hisse["High"],Hisse["Low"],Hisse["Close"],window=20,fillna=False)
    Hisse["Buy_CCI"]=np.where((Hisse["CCI"]>0),1,0)
    Hisse["Buy_CCIS"]=np.where((Hisse["Buy_CCI"]>Hisse["Buy_CCI"].shift(1)),1,0)
    Hisse["EMA10"]=ta.trend.ema_indicator(Hisse["Close"],window=10,fillna=False)
    Hisse["EMA30"]=ta.trend.ema_indicator(Hisse["Close"],window=30,fillna=False)
    Hisse["Buy_EMA10"]=np.where((Hisse["Close"]>Hisse["EMA10"]),1,0)
    Hisse["Buy_EMA10S"]=np.where((Hisse["Buy_EMA10"]>Hisse["Buy_EMA10"].shift(1)),1,0)
    Hisse["Buy_EMA10_EMA30"]=np.where((Hisse["EMA10"]>Hisse["EMA30"]),1,0)
    Hisse["Buy_EMA10_EMA30S"]=np.where((Hisse["Buy_EMA10_EMA30"]>Hisse["Buy_EMA10_EMA30"].shift(1)),1,0)
    Hisse["Stochastic"]=ta.momentum.stoch_signal(Hisse["High"],Hisse["Low"],Hisse["Close"],window=3,fillna=False)
    Hisse["Stochastic_Buy"]=np.where((Hisse["Stochastic"]>20),1,0)
    Hisse["Stochastic_BuyS"]=np.where((Hisse["Stochastic_Buy"]>Hisse["Stochastic_Buy"].shift(1)),1,0)
    Hisse["KAMA"]=ta.momentum.kama(Hisse["Close"],window=10,pow1=2,pow2=30, fillna=False)
    Hisse["Buy_KAMA"]=np.where((Hisse["Close"]>Hisse["KAMA"]),1,0)
    Hisse["Buy_KAMAS"]=np.where((Hisse["Buy_KAMA"]>Hisse["Buy_KAMA"].shift(1)),1,0)
    Hisse['SMA5'] = ta.trend.sma_indicator(Hisse['Close'], window=5)
    Hisse['SMA22'] = ta.trend.sma_indicator(Hisse['Close'], window=22)
    Hisse['SMA50'] = ta.trend.sma_indicator(Hisse['Close'], window=50)
    Hisse["Buy_SMA5"]=np.where((Hisse["Close"]>Hisse["SMA5"]),1,0)
    Hisse["Buy_SMA22"]=np.where((Hisse["Close"]>Hisse["SMA22"]),1,0)
    Hisse["Buy_SMA50"]=np.where((Hisse["Close"]>Hisse["SMA50"]),1,0)
    Hisse["Buy_SMA5S"]=np.where((Hisse["Buy_SMA5"]>Hisse["Buy_SMA5"].shift(1)),1,0)
    Hisse["Buy_SMA22S"]=np.where((Hisse["Buy_SMA22"]>Hisse["Buy_SMA22"].shift(1)),1,0)
    Hisse["Buy_SMA50S"]=np.where((Hisse["Buy_SMA50"]>Hisse["Buy_SMA50"].shift(1)),1,0)
    Hisse["CMF"]=ta.volume.chaikin_money_flow(Hisse["High"],Hisse["Low"],Hisse["Close"],Hisse["Volume"],window=20,fillna=False)
    Hisse["Buy_CMF"]=np.where((Hisse["CMF"]>0),1,0)
    Hisse["Buy_CMFS"]=np.where((Hisse["Buy_CMF"]>Hisse["Buy_CMF"].shift(1)),1,0)
    HisseDeger=Hisse.tail(1).squeeze()
    f=(HisseDeger["Vol_diff"]/HisseDeger["Volume"]+HisseDeger["Volume"]/Hisse["Volume"].mean())/2
    Score1=HisseDeger["Buy_MACDS"]+HisseDeger["Buy_AOS"]+HisseDeger["Buy_EMA10_EMA30S"]+HisseDeger["Buy_SMA5S"]+HisseDeger["Buy_SMA22S"]+HisseDeger["Buy_RSIS"]+HisseDeger["Stochastic_BuyS"]+HisseDeger["Buy_CCIS"]+HisseDeger["Buy_KAMAS"]+HisseDeger["Buy_CMFS"]+f
    print(a,Score1)
    if Score1>2.5:
        HisseAl.append(a)
       


# In[11]:


app.layout=html.Div(["TEKNIK YUKSELISTE HISSELER",
    html.Div([
        html.P(html.Label("Hisse Seçiniz")),
        html.Hr(),
        dcc.Dropdown(id="Stocks",
                    options=[{"label":i,"value":i}for i in HisseAl],
                    value='Stocks')
    ],style={"width":"48%"}),
    
    dcc.Graph(id="graph",style={"width":"150vh","height":"70vh"})
    
],style={"color":"blue","padding":10})


# In[12]:


@app.callback(
    Output("graph","figure"),
    [Input("Stocks","value")])
def update_graph(Stock):
    Hisse=yf.download(Stock,
                     start="2021-01-01",
                     end=today,
                     progress=False)
    Hisse.index=pd.to_datetime(Hisse.index)
    Hisse["Return"]=Hisse["Close"].diff()
    Hisse["Return_pct"]=Hisse["Close"].pct_change()
    Hisse["Target_Cls"]=np.where(Hisse.Return>0,1,0)
    Hisse["Vol_diff"]=Hisse["Volume"].diff()
    Hisse["Vol_change"]=Hisse["Volume"].pct_change()
    indicator_bb= BollingerBands(close=Hisse["Close"],window=20,window_dev=2)
    Hisse["bb_bbm"]=indicator_bb.bollinger_mavg()
    Hisse["bb_bbh"]=indicator_bb.bollinger_hband()
    Hisse["bb_bbl"]=indicator_bb.bollinger_lband()
    Hisse["MACD"] = ta.trend.macd(Hisse["Close"], window_slow = 26, window_fast= 12, fillna=False)
    Hisse["MACDS"] = ta.trend.macd_signal(Hisse["Close"], window_sign= 9, fillna=False)
    Hisse["Buy_MACD"]=np.where((Hisse["MACD"]>Hisse["MACDS"]),1,0)
    Hisse["Buy_MACDS"]=np.where((Hisse["Buy_MACD"]>Hisse["Buy_MACD"].shift(1)),1,0)
    Hisse['OBV'] = ta.volume.on_balance_volume(Hisse['Close'], Hisse['Volume'])
    Hisse["RSI"]=ta.momentum.rsi(Hisse["Close"],window= 14, fillna= False)
    Hisse["Buy_RSI"]=np.where((Hisse["RSI"]>30),1,0)
    Hisse["Buy_RSIS"]=np.where((Hisse["Buy_RSI"]>Hisse["Buy_RSI"].shift(1)),1,0)
    Hisse["AO"]=ta.momentum.awesome_oscillator(Hisse["High"],Hisse["Low"],window1=5,window2=34,fillna=True)
    Hisse["Buy_AO"]=np.where((Hisse["AO"]>0),1,0)
    Hisse["Buy_AOS"]=np.where((Hisse["Buy_AO"]>Hisse["Buy_AO"].shift(1)),1,0)
    Hisse["CCI"]=ta.trend.cci(Hisse["High"],Hisse["Low"],Hisse["Close"],window=20,fillna=False)
    Hisse["Buy_CCI"]=np.where((Hisse["CCI"]>0),1,0)
    Hisse["Buy_CCIS"]=np.where((Hisse["Buy_CCI"]>Hisse["Buy_CCI"].shift(1)),1,0)
    Hisse["CCI"]=ta.trend.cci(Hisse["High"],Hisse["Low"],Hisse["Close"],window=20,fillna=False)
    Hisse["Buy_CCI"]=np.where((Hisse["CCI"]>0),1,0)
    Hisse["Buy_CCIS"]=np.where((Hisse["Buy_CCI"]>Hisse["Buy_CCI"].shift(1)),1,0)
    Hisse["EMA10"]=ta.trend.ema_indicator(Hisse["Close"],window=10,fillna=False)
    Hisse["EMA30"]=ta.trend.ema_indicator(Hisse["Close"],window=30,fillna=False)
    Hisse["Buy_EMA10"]=np.where((Hisse["Close"]>Hisse["EMA10"]),1,0)
    Hisse["Buy_EMA10S"]=np.where((Hisse["Buy_EMA10"]>Hisse["Buy_EMA10"].shift(1)),1,0)
    Hisse["Buy_EMA10_EMA30"]=np.where((Hisse["EMA10"]>Hisse["EMA30"]),1,0)
    Hisse["Buy_EMA10_EMA30S"]=np.where((Hisse["Buy_EMA10_EMA30"]>Hisse["Buy_EMA10_EMA30"].shift(1)),1,0)
    Hisse["Stochastic"]=ta.momentum.stoch_signal(Hisse["High"],Hisse["Low"],Hisse["Close"],window=3,fillna=False)
    Hisse["Stochastic_Buy"]=np.where((Hisse["Stochastic"]>20),1,0)
    Hisse["Stochastic_BuyS"]=np.where((Hisse["Stochastic_Buy"]>Hisse["Stochastic_Buy"].shift(1)),1,0)
    Hisse["KAMA"]=ta.momentum.kama(Hisse["Close"],window=10,pow1=2,pow2=30, fillna=False)
    Hisse["Buy_KAMA"]=np.where((Hisse["Close"]>Hisse["KAMA"]),1,0)
    Hisse["Buy_KAMAS"]=np.where((Hisse["Buy_KAMA"]>Hisse["Buy_KAMA"].shift(1)),1,0)
    Hisse['SMA5'] = ta.trend.sma_indicator(Hisse['Close'], window=5)
    Hisse['SMA22'] = ta.trend.sma_indicator(Hisse['Close'], window=22)
    Hisse['SMA50'] = ta.trend.sma_indicator(Hisse['Close'], window=50)
    Hisse["Buy_SMA5"]=np.where((Hisse["Close"]>Hisse["SMA5"]),1,0)
    Hisse["Buy_SMA22"]=np.where((Hisse["Close"]>Hisse["SMA22"]),1,0)
    Hisse["Buy_SMA50"]=np.where((Hisse["Close"]>Hisse["SMA50"]),1,0)
    Hisse["Buy_SMA5S"]=np.where((Hisse["Buy_SMA5"]>Hisse["Buy_SMA5"].shift(1)),1,0)
    Hisse["Buy_SMA22S"]=np.where((Hisse["Buy_SMA22"]>Hisse["Buy_SMA22"].shift(1)),1,0)
    Hisse["Buy_SMA50S"]=np.where((Hisse["Buy_SMA50"]>Hisse["Buy_SMA50"].shift(1)),1,0)
    Hisse["CMF"]=ta.volume.chaikin_money_flow(Hisse["High"],Hisse["Low"],Hisse["Close"],Hisse["Volume"],window=20,fillna=False)
    Hisse["Buy_CMF"]=np.where((Hisse["CMF"]>0),1,0)
    Hisse["Buy_CMFS"]=np.where((Hisse["Buy_CMF"]>Hisse["Buy_CMF"].shift(1)),1,0)
    Hisse["Target_Close_Next_3Day"]=round((Hisse["Target_Cls"].shift(-1)+Hisse["Target_Cls"].shift(-2)+Hisse["Target_Cls"].shift(-3))/3,0)
    Hisse["Target_Close_Next_3Day"].fillna(1,inplace=True)
    Hisse["Return_pct_next_day"]=Hisse["Return_pct"].shift(-1)
    Hisse["Return_pct_next_day"].fillna(Hisse["Return_pct_next_day"].mean(),inplace=True)
    
    fig2 = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = Score1,
    mode = "gauge+number+delta",
    title = {'text': "Speed"},
    delta = {'reference': 1},
    gauge = {'axis': {'range': [None, 5]},
             'steps' : [
                 {'range': [0, 2], 'color': "lightgray"},
                 {'range': [2, 5], 'color': "gray"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1}}))

    fig2.show()
    
    
    fig = ms.make_subplots(rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05)
    
    fig.add_trace(go.Candlestick(x = Hisse.index,
                                low = Hisse["Low"],
                                high = Hisse["High"],
                                close = Hisse["Close"],
                                open = Hisse["Open"],
                                name="Price"),
                  
                                row=1,
                                col=1)
    
    
    fig.add_trace(go.Bar(x=Hisse.index,
                        y=Hisse["Volume"],
                        name="Volume"),
                        row=2,
                        col=1)
    
    fig.add_trace(go.Scatter(x=Hisse.index,
                        y=Hisse["MACD"],
                        name="MACD"),
                        row=3,
                        col=1)
    fig.add_trace(go.Scatter(x=Hisse.index,
                        y=Hisse["MACDS"],
                        mode="lines+markers",
                        name="MACDS"),
                        row=3,
                        col=1)
    fig.add_trace(go.Scatter(x=Hisse.index,
                        y=Hisse["OBV"],
                        name="OBV"),
                        row=4,
                        col=1)
    
    
    fig.update_layout(title = "Interactive CandleStick & Volume Chart",
    yaxis1_title = "Stock Price ($)",
    yaxis2_title = "Volume (M)",
    yaxis3_title = "MACD Value",   
    yaxis4_title = "OBV",                  
    xaxis4_title = "Time",
    xaxis1_rangeslider_visible = False,
    )
    
    return fig

      


# In[ ]:


if __name__== "__main__":
    app.run_server()


# In[ ]:





# In[ ]:




