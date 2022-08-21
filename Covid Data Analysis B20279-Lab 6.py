'''
NAME - Aryan Ali
BRANCH - DSE
ROLL NO. - B20279
MOBILE NO. - 9027209190
'''
#%% Importing libraries
import math
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Question 1
print()
print('______________________Question 1__________________________\n')
# Reading data
df = pd.read_csv("daily_covid_cases.csv")
df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)

#%% Part (a)
print('___________________Part (a)_______________________')
# Plotting line plot for new confirmed cases
x = ['Feb-20','Apr-20','Jun-20','Aug-20','Oct-20','Dec-20','Feb-21','Apr-21','Jun-21','Aug-21','Oct-21']
fig, ax = plt.subplots()
ax.plot(df['new_cases'])
ax.set_xticks(np.linspace(0,612,11))
ax.set_xticklabels(x)
ax.set_xlabel("Month-Year")
ax.set_ylabel("New confirmed cases")
plt.show()
print()

#%% Part (b)
print('___________________Part (b)_______________________')
# Finding corr coeff b/w 1 day time lag and given time seq
df1bi = df.iloc[1: , :]
df1bii = df.set_index(["Date"]).shift(1)
df1bii = df1bii.dropna()
corr, _ = pearsonr(df1bi['new_cases'], df1bii['new_cases'])
print('Pearson correlation (autocorrelation) coefficient: ' ,round(corr,3),'\n')

#%% Part (c)
print('___________________Part (c)_______________________')
# Plotting scatter plot b/w one-day time lag and given time sequence
plt.scatter(df1bii['new_cases'],df1bi['new_cases'])
plt.xlabel("Given time sequence")
plt.ylabel("One day lag sequence")
plt.show()
print()

#%% Part (d)
print('___________________Part (d)_______________________')
# Defining a function for calculating autocorrelation
l = []
def autocorr(n):
    df1di = df.iloc[n: , :]
    df1dii = df.set_index(["Date"]).shift(n)
    df1dii = df1dii.dropna()
    corr, _ = pearsonr(df1di['new_cases'], df1dii['new_cases'])
    l.append(corr)
    print('Pearson correlation (autocorrelation) coefficient (lag = ',n,'): ',round(corr,3))

# Calculating autocorrelation for different time lag values and plotting them
for i in range(1,7):
    autocorr(i)
plt.plot(np.linspace(1,6,6),l)
plt.xlabel("Lag values")
plt.ylabel("Correlation coefficient")
plt.show()
print()

#%% Part (e)
print('___________________Part (e)_______________________')
# Plotting correlogram using python inbuilt library
df1e = df.set_index(["Date"])
plot_acf(df1e, lags=6)
plt.xlabel("Lag values")
plt.ylabel("Correlation coefficient")
plt.show()
print()

#%% Question 2
print('______________________Question 2__________________________\n')
print('___________________Part (a)_______________________')

series = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'],index_col=['Date'],sep=',')
# Splitting the data for training and testing
test_size = 0.35 
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# Plotting training and testing data
fig, ax = plt.subplots()
ax.plot(train, label='Train Data')
ax.plot(test, label='Test Data')
ax.legend()
plt.show()

# Obtaining coefficient of the AR model
window = 5
model = AutoReg(train, lags=5)
model_fit = model.fit()
coef = model_fit.params
print("The coefficients obtained from the AR model are :",coef,'\n')

#%% Part (b)
print('___________________Part (b)_______________________')
# Predicting the value of the dataset
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
exp = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	exp.append(obs)
	history.append(obs)

#%% Part (b)(i)
print('___________________Part (b)(i)____________________')
# Plotting the scatter plot b/w actual and predicted values
plt.scatter(exp,predictions)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()
print()

#%% Part (b) (ii)
print('___________________Part (b)(ii)___________________')
# Plotting line plot b/w actual and predicted values
fig, ax = plt.subplots()
ax.plot(test, label="Actual data")
ax.plot(predictions, label="Predicted data")
ax.set_xticks(np.linspace(0,214,3))
ax.set_xticklabels(x[8:])
ax.set_xlabel("Month-Year")
ax.set_ylabel("New confirmed cases")
plt.legend()
plt.show()
print()

#%% Part (b) (iii)
print('___________________Part (b)(iii)__________________')
# Computing RMSE b/w actual and predicted values
rmse = math.sqrt(mean_squared_error(exp, predictions))/(np.mean(exp))
print("RMSE(%):", round(rmse*100,3))

# Computing MAPE b/w actual and predicted values
mape = np.mean(np.abs((np.array(exp) - np.array(predictions))/np.array(exp)))*100
print("MAPE:", round(mape,3),'\n')

#%% Question 3
print('______________________Question 3__________________________\n')
l = [1, 5, 10, 15, 25]

# Training AR model and predicting values for test dataset using it
rm = []
ma = []
for window in l:
	model = AutoReg(train, lags=window)
	model_fit = model.fit()
	coef = model_fit.params
	history = train[len(train) - window:]
	history = [history[i] for i in range(len(history))]
	predictions = list()                
	exp = list()
	for t in range(len(test)):
		length = len(history)
		lag = [history[i] for i in range(length - window, length)]
		yhat = coef[0]
		for d in range(window):
			yhat += coef[d + 1] * lag[window - d - 1]
		obs = test[t]
		predictions.append(yhat)
		exp.append(obs)
		history.append(obs)
    # Computing RMSE for predicted values
	rmse = math.sqrt(mean_squared_error(exp, predictions)) / (np.mean(exp))
	rm.append(round(rmse*100,3))
    # Computing MAPE for predicted values
	mape = np.mean(np.abs((np.array(exp) - np.array(predictions)) / np.array(exp))) * 100
	ma.append(round(mape,3))

# Plotting the bar chart for RMSE and MAPE with lag values
plt.bar(l,rm)
plt.title('RMSE(%) v/s Time lag')
plt.show()
print('RMSE(%)',rm,'\n')
plt.bar(l,ma)
plt.title('MAPE v/s Time lag')
plt.show()
print('MAPE',ma,'\n')

#%% Question 4
print('______________________Question 4__________________________\n')
m = 2/np.sqrt(len(train))
i = 1

# Defining function for calculating autocorrelation
def autocorr(n):
    train_t0 = [item for elem in train[n:] for item in elem]
    train_tn = [item for elem in train[:(-1*n)] for item in elem]
    corr, _ = pearsonr(train_t0,train_tn)
    return corr
while True:
	if autocorr(i) > m:
		i+=1
	else:
		break
i-=1

# Printing heuristics value
print ('The heuristic value for the optimal number of lags is: ',i)

# Training AR model
window = i
model = AutoReg(train, lags=window)
model_fit = model.fit()
coef = model_fit.params
history = train[len(train) - window:]
history = [history[i] for i in range(len(history))]

# Predicting values using AR model
predictions = list()
exp = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length - window, length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d + 1] * lag[window - d - 1]
    obs = test[t]
    predictions.append(yhat)
    exp.append(obs)
    history.append(obs)

# Computing RMSE for predicted values
rmse = math.sqrt(mean_squared_error(exp, predictions)) / (np.mean(exp))
print("RMSE(%) :",round(rmse*100,3))

# Computing MAPE for predicted values
mape = np.mean(np.abs((np.array(exp) - np.array(predictions)) / np.array(exp))) * 100
print("MAPE :",round(mape,3))


