import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re

from IPython.core.pylabtools import figsize
from rich import columns


# importando a base de dados
def Importar():
    df = pd.read_csv('covid_19_data.csv',parse_dates=['ObservationDate','Last Update'])
    return df

def Formatar(df):
    df.columns = [re.sub(r"[/| ]",'',col_name).lower() for col_name in df.columns]

def Grafico_confirmados_brasil(df):
    px.line(df, 'observationdate', 'confirmed', title='Casos confirmados no Brasil')
    plt.plot(df.observationdate, df.confirmed, color='red')
    plt.title('Casos confirmados no Brasil')
    plt.xlabel('observationdate')
    plt.ylabel('confirmed')
    plt.show()


def Grafico_novos_casos_brasil(df):
    px.line(df, 'observationdate', 'novoscasos', title='Novos casos por dia')
    plt.plot(df.observationdate, df.novoscasos, color='black')
    plt.title('Novos casos por dia')
    plt.xlabel('observationdate')
    plt.ylabel('novos casos')
    plt.show()

dados = Importar()
print(dados.head())
print(dados.dtypes)

Formatar(dados)
print(dados)


brasil = dados.loc[(dados.countryregion == 'Brazil') & (dados.confirmed > 0)]
# grafico da evolução dos casos covid19
Grafico_confirmados_brasil(brasil)

# novos casos por dia
brasil['novoscasos'] = list(map(lambda x: 0 if (x==0) else brasil['confirmed'].iloc[x] - brasil['confirmed'].iloc[x-1], np.arange(brasil.shape[0])))
Grafico_novos_casos_brasil(brasil)

# grafico das mortes
fig = go.Figure()
fig.add_trace(go.Scatter(x=brasil.observationdate,y=brasil.deaths,name='Mortes',mode='lines+markers',line = {'color':'red'}))
fig.update_layout(title='Mortes por COVID-19 no Brasil')
fig.show()

def Taxa_crescimento(df,var,data_ini=None,data_fim=None):
    if data_ini == None:
        data_ini = df.observationdate.loc[df[var] > 0].min()
    else:
        data_ini = pd.to_datetime(data_ini)

    if data_fim == None:
        data_fim = df.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)

    passado = df.loc[df.observationdate == data_ini,var].values[0]
    presente = df.loc[df.observationdate == data_fim,var].values[0]

    n = (data_fim-data_ini).days

    taxa = (presente/passado)**(1/n) -1
    return taxa*100

print(f"A taxa é {round(Taxa_crescimento(brasil,'confirmed'),3)}")

def Taxa_crescimento_diaria(df,var,data_ini=None):
    if data_ini == None:
        data_ini = df.observationdate.loc[df[var] > 0].min()
    else:
        data_ini = pd.to_datetime(data_ini)

    data_fim = df.observationdate.max()
    n = (data_fim - data_ini).days

    taxa = list(map(lambda x: (df[var].iloc[x] - df[var].iloc[x-1])/df[var].iloc[x-1],range(1,n+1)))
    return np.array(taxa) * 100

tx_dia = Taxa_crescimento_diaria(brasil,'confirmed')
print(tx_dia)


def Grafico_taxa_por_dia_brasil(t):
    primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()
    plt.plot(pd.date_range(primeiro_dia,brasil.observationdate.max())[1:],t)
    plt.title('Taxa de crescimento de casos confirmados no Brasil por dia')
    plt.xlabel('observationdate')
    plt.ylabel('Casos')
    plt.show()
Grafico_taxa_por_dia_brasil(tx_dia)

# Predições
from statsmodels.tsa.seasonal import seasonal_decompose

confirmados = brasil.confirmed
confirmados.index = brasil.observationdate

result = seasonal_decompose(confirmados)

fig, (ax,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(10,8))

ax.plot(result.observed)
ax2.plot(result.trend)
ax3.plot(result.seasonal)
ax4.plot(confirmados.index, result.resid)
ax4.axhline(0,linestyle='dashed',c='black')
plt.show()

# modelo arima para series temporais
from pmdarima.arima import auto_arima
modelo = auto_arima(confirmados)

# grafico de previsão de casos no brasil
fig = go.Figure(go.Scatter(x=confirmados.index, y=confirmados,name='Observados'))
fig.add_trace(go.Scatter(x=confirmados.index, y=modelo.predict_in_sample(),name='Preditos'))
fig.add_trace(go.Scatter(x=pd.date_range('2020-05-20','2020-06-20'),y=modelo.predict(31),name='Forecast'))

fig.update_layout(title='Previsão de casos no Brasil para os próximos 30 dias')
fig.show()

# modelo de crescimento
from fbprophet import Prophet

# processamento dos dados
treino = confirmados.reset_index()[:-5]
teste = confirmados.reset_index()[-5:]

treino.rename(columns={'observationdate':'ds','confirmed':'y'},inplace=True)
teste.rename(columns={'observationdate':'ds','confirmed':'y'},inplace=True)

# modelo
profeta = Prophet(growth= 'logistic',changepoints=['2020-03-21','2020-03-30','2020-04-25','2020-05-03','2020-05-10'])

pop = 211463256
treino['cap'] = pop
# treino do modelo
profeta.fit(treino)

#construindo previsões
future_dates = profeta.make_future_dataframe(periods=200)
future_dates['cap'] = pop
forecast = profeta.predict(future_dates)

fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast.ds,y = forecast.yhat,name='Predição'))
fig.add_trace(go.Scatter(x=treino.ds,y=treino.y,name='Observados - Treino'))
fig.update_layout(title='Predições de casos confirmados no Brasil')
fig.show()

