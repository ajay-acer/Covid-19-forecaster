from flask import Flask,render_template,request  
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pmdarima as pm
import datetime as dt
from io import BytesIO
import base64

app = Flask(__name__) #creating the Flask class object   

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

@app.route('/') #decorator drfines the   
def home():
    return render_template('index.html',states=df['State']) 

@app.route('/predict',methods=['POST'])
def hello():
    print('Submitted')
    form_data=request.form
    print(form_data)
    print(form_data['States'])
    #print(form_data['days'])
    if(form_data['days']=='14-day'):
        days=14
    else:
        days=7
    df,plot_url=forecast(form_data['States'], form_data['cases-deaths'],days)
    #print(df.iloc[0,'Date'])
    return render_template('result.html',df=df,state=form_data['States'],attr=form_data['cases-deaths'],days=days,plot_url=plot_url)


def forecast(state,attr,days):
    print("For state:",state)
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

    img = BytesIO()
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
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    df=pd.DataFrame()
    df['Date']=x
    df['Lower_value']=list(lower_series)
    df['Upper_value']=list(upper_series)
    df['Predicted value']=list(fc_series)
    print(df)
    return df,plot_url
