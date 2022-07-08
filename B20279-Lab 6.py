# Aryan Ali
# B20279
# 9027209190

from statsmodels.tsa.ar_model import AutoReg as AR
import math
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Q1(a)-----------------------------------------------------------------------------------------------------------------
df = pd.read_csv('daily_covid_cases.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
fig = go.Figure([go.Scatter(x=df['Date'], y=df['new_cases'])])
fig.update_xaxes(title_text='Month-Year', dtick="M1", tickformat="%b<br>%Y")
fig.update_yaxes(title_text='New confirmed cases')
fig.show()
# (b)--------------------------------------------------------------
df_lag = df.drop([0], axis=0)
df_org = df.drop([611], axis=0)
a_corr = np.corrcoef(df_org['new_cases'], df_lag['new_cases'])
print(a_corr)
# (c)--------------------------------------------------------------
plt.scatter(df_org['new_cases'], df_lag['new_cases'])
plt.title('Scatter plot between original and lagged dataframe')
plt.show()
# (d)--------------------------------------------------------------
lag = [1, 2, 3, 4, 5, 6]
ac = sm.tsa.acf(df['new_cases'], nlags=6)
ac = np.delete(ac, 0)
print(ac)
plt.plot(lag, ac, '-ro')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.show()
# (e)---------------------------------------------------------------
fig5 = tsaplots.plot_acf(df['new_cases'], lags=6)
plt.show()

# Q2(a)-----------------------------------------------------------------------------------------------------------------
series = pd.read_csv('daily_covid_cases.csv', index_col=['Date'])
test_size = 0.35  # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

window = 5  # The lag=5
model = AR(train, lags=window)
model_fit = model.fit()  # fit/train the model
coef = model_fit.params
print(coef)
# (b)-------------------------------------------------------------------------
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()  # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window, length)]
    yhat = coef[0]  # Initialize to w0
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]  # Add other values
    obs = test[t]
    predictions.append(yhat)  # Append predictions to compute RMSE later
    history.append(obs)  # Append actual test value to history, to be used in next step.

plt.scatter(test, predictions)
plt.title('Actual vs Predictions')
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.show()

dates = df['Date']
dates = dates.iloc[397:612]

plt.plot(dates, test)
plt.plot(dates, predictions)
plt.xlabel('Date')
plt.ylabel('New confirmed cases')
plt.legend(['Actual Data', 'predicted Data'])
plt.show()

rmspe = np.sqrt(np.mean(np.square(((test - predictions) / test))))*100
print('RMSPE: ', str(rmspe))
def mape(actual, predic):
    actual, predic = np.array(actual), np.array(predic)
    return np.mean(np.abs((actual - predic) / actual)) * 100


map_err = mape(test, predictions)
print('MAPE: ', str(map_err))

# Q3--------------------------------------------------------------------------------------------------------------------
rmse_err = []
map_error = []
lags = [1, 5, 10, 15, 25]
def autoreg(n):
    model3 = AR(train, lags=n)
    model3_fit = model3.fit()
    coef3 = model3_fit.params
    history3 = train[len(train) - n:]
    history3 = [history3[i] for i in range(len(history3))]
    preds = list()
    for k in range(len(test)):
        length3 = len(history3)
        lag3 = [history3[i] for i in range(length3 - n, length3)]
        yhat3 = coef3[0]
        for l in range(n):
            yhat3 += coef3[l + 1] * lag3[n - l - 1]
        obs3 = test[k]
        preds.append(yhat3)
        history3.append(obs3)
    rmspe3 = np.sqrt(np.mean(np.square(((test - preds) / test)))) * 100
    rmse_err.append(rmspe3)
    mape3 = mape(test, preds)
    map_error.append(mape3)


for i in lags:
    autoreg(i)
plt.plot(lags, rmse_err, '-ro')
plt.title('RMSE vs time-lag')
plt.xlabel('time-lag')
plt.ylabel('RMSE(%)')
plt.show()
plt.plot(lags, map_error, '-ro')
plt.title('MAPE vs time-lag')
plt.xlabel('time-lag')
plt.ylabel('MAPE')
plt.show()
# Q4--------------------------------------------------------------------------------------------------------------------
ac4 = sm.tsa.acf(train, nlags=100)
ac4 = np.delete(ac4, 0)
for i in range(len(ac4)):
    if ac4[i] < 0.100:
        print(i)
    else:
        continue

model4 = AR(train, lags=78)
model4_fit = model4.fit()
coef4 = model4_fit.params

history4 = train[len(train)-78:]
history4 = [history4[i] for i in range(len(history4))]
pred = list()
for m in range(len(test)):
    length4 = len(history4)
    lag4 = [history4[i] for i in range(length4-78, length4)]
    yhat4 = coef4[0]
    for p in range(78):
        yhat4 += coef4[p+1] * lag4[78-p-1]
    obs4 = test[m]
    pred.append(yhat4)
    history4.append(obs4)

rmspe4 = np.sqrt(np.mean(np.square(((test - pred) / test)))) * 100
mape4 = mape(test, pred)
print(rmspe4)
print(mape4)
# ----------------------------------------------------------------------------------------------------------------------
