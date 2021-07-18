# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 19:12:59 2021

@author: Ajay kumar
"""
###############################################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pmdarima as pm
###############################################################################

######################Importing the data#######################################
data1=pd.read_csv('https://api.covid19india.org/csv/latest/state_wise_daily.csv')
data1.pop('DD') #Removing the state Daman and Diu(DD)
data1.pop('UN') #Removing the column "state unassigned"(UN)
l=(data1.columns) #Getting the columns
l=l[3:]  #First 3 columns are not required

data2=pd.read_csv('https://api.covid19india.org/csv/latest/state_wise.csv')
indexNames = data2[data2['State_code'] == 'UN'].index   #Removing the row "state unassigned"(UN)
data2.drop(indexNames,inplace=True)
df=pd.DataFrame()
df=data2['State']
df=df.rename_axis('Index').reset_index()
df.pop('Index')
df.iloc[0,0]='India'

for i in range(len(l)):
    data1=data1.rename(columns = {l[i]:df.iloc[i,0]})  #Renaming the column of data1 dataframe
###############################################################################

###############################################################################
def forecast(state,attr):
    print("For state:",state)
    days=14
    state='India'
    attr='Confirmed Cases'
    li1=[]
    li2=[]
    li3=[]
    for i in range(len(data1)):
        if data1.loc[i,'Status']=='Confirmed':     #Status is Confrimed
            li2.append(data1.loc[i,state])
            li1.append(data1.loc[i,'Date_YMD'])
        if data1.loc[i,'Status']=='Deceased':
            li3.append(data1.loc[i,state])
   
    data_state=pd.DataFrame(list(zip(li1,li2,li3)),columns=['Date','Confirmed Cases','Death']) 
    n_periods = days
    x=[dt.datetime.strptime(data_state.iloc[-1]['Date'],'%Y-%m-%d').date()]
    for i in range(n_periods):
        x.append(x[-1]+dt.timedelta(days=1))
    x.pop(0)
    data_state.set_index('Date',inplace=True)
    model = pm.auto_arima(data_state[attr], start_p=1, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=50, max_q=50, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    
    
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    fc=fc.astype(int)
    confint=confint.astype(int)
    index_of_fc = np.arange(len(data_state[attr]), len(data_state[attr])+n_periods)


    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    with plt.style.context(('dark_background')):
        fig=plt.figure(figsize=(12,10))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
        plt.plot(data_state[attr])
        plt.plot(fc_series, color='red')
        plt.gcf().autofmt_xdate(rotation=90)
        plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='white', alpha=.35)
        plt.grid()
    plt.show()
    
    df=pd.DataFrame()
    df['Date']=x
    df['Lower_value']=list(lower_series)
    df['Upper_value']=list(upper_series)
    df['Predicted value']=list(fc_series)
    print(df)
###############################################################################

while(True):
    state=input("Enter the state:");
    x=int(input("Enter 1 for Confirmed case 2.Death"))
    if x==1:
        attr='Confirmed Cases'
    elif x==2:
        attr='Death'
    forecast(state, attr)
###############################################################################
xyz=len(data_state)-days
data_state=data_state.iloc[:xyz,:]
train=data_state.iloc[xyz:,:]

from sklearn.metrics import r2_score



df1=pd.DataFrame()

df1['Lower_value']=list(lower_series)
df1['Upper_value']=list(upper_series)
df1['Predicted value']=list(fc_series)
df1['Actual value']=list(train['Confirmed Cases'])

print("Metric for Death=", sklearn.metrics.r2_score(actual,forecast))

forecast=np.array(df1['Lower_value'])
forecast=np.array(df1['Upper_value'])
forecast=np.array(df1['Predicted value'])
actual=np.array(df1['Actual value'])
print("MAPE=",np.mean(np.abs(forecast - actual)/np.abs(actual)) ) # MAPE
print("ME=",np.mean(forecast - actual) )            # ME
print("MAE=",np.mean(np.abs(forecast - actual)))    # MAE
print("MPE=",np.mean((forecast - actual)/actual))   # MPE
print("RMSE=",np.mean((forecast - actual)**2)**.5)  # RMSE
print("corr=",np.corrcoef(forecast, actual)[0,1])   # corr

      

      

      

      

      

      

      

      

      

      
