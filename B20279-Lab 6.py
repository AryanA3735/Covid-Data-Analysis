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
print(" ___________________________________Question 1_________________________________")
df = pd.read_csv("daily_covid_cases.csv")
df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
#%% Part A
print('Part A')
# Plotting the Data
x = ['Feb-20','Apr-20','Jun-20','Aug-20','Oct-20','Dec-20','Feb-21','Apr-21','Jun-21','Aug-21','Oct-21']
fig, ax = plt.subplots()
ax.plot(df['new_cases'])
ax.set_xticks(np.linspace(0,612,11))
ax.set_xticklabels(x)
ax.set_xlabel("Month-Year")
ax.set_ylabel("New confirmed cases")
#plt.savefig('Q1a.eps', format = 'eps')
plt.show()

#%% Part B
print()
print('Part B')
# Shifting the data by 1 day
# Computing the pearson correlation coefficient
df1bi = df.iloc[1: , :]
df1bii = df.set_index(["Date"]).shift(1)
df1bii = df1bii.dropna()
corr, _ = pearsonr(df1bi['new_cases'], df1bii['new_cases'])
print('Pearson correlation (autocorrelation) coefficient: %.3f' % corr)

#%% Part C
print()
print('Part C')
# Plotting the lagged vs given sequence
plt.scatter(df1bii['new_cases'],df1bi['new_cases'], color = 'b')
plt.xlabel("Given time sequence")
plt.ylabel("One day lag sequence")
#plt.savefig('Q1c.eps', format = 'eps')
plt.show()

#%% Part D
print()
print('Part D')
# Computing the correlation coefficient for lag 1-6
# Storing it into a list
# Ploting the correlation coefficient
l = []
def autocorr(n):
    df1di = df.iloc[n: , :]
    df1dii = df.set_index(["Date"]).shift(n)
    df1dii = df1dii.dropna()
    corr, _ = pearsonr(df1di['new_cases'], df1dii['new_cases'])
    l.append(corr)
    print('Pearson correlation (autocorrelation) coefficient (lag = ',n,'): ',round(corr,3))

for i in range(1,7):
    autocorr(i)
plt.plot(np.linspace(1,6,6),l)
plt.xlabel("Lag values")
plt.ylabel("Correlation coefficient")
#plt.savefig('Q1d.eps', format = 'eps')
plt.show()

#%% Part E
# Plotting the correlation coefficient using plot.acf
print()
print('Part E')
df1e = df.set_index(["Date"])
plot_acf(df1e, lags=5)
plt.xlabel("Lag values")
plt.ylabel("Correlation coefficient")
#plt.savefig('Q1e.eps', format = 'eps')
plt.show()


#%% Question 2
print(" ___________________________________Question 2_________________________________")

#%% Part A
# Splitting our dataset into training and testing (75-35 split)
print()
print('Part A')
series = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'], index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# Plotting the testing and training data
fig, ax = plt.subplots()
ax.plot(train, label='Train Data')
ax.plot(test, label='Test Data')
ax.legend()
plt.show()

# Building AR model for lags = 5
# Printing the obtained weight coefficients
window = 5
model = AutoReg(train, lags=5)
model_fit = model.fit()
coef = model_fit.params
print("The coefficients obtained from the AR model are :",coef)

#%% Part B
print()
print('Part B')
# Taking 5 lag values (independent variable)
# Multiplying with the weights and computing the dependent variable (t value)
# walk forward over time steps in test
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

#%% Part B (i)
# Plotting the actual vs predicted series
print('Part B (i)')
plt.scatter(exp,predictions)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
#plt.savefig('Q2bi.eps', format = 'eps')
plt.show()

#%% Part B (ii)
print('Part B (ii)')
# Plotting the actual vs predicted series on same plot
fig, ax = plt.subplots()
ax.plot(test, label="Actual data")
ax.plot(predictions, label="Predicted data")
ax.set_xticks(np.linspace(0,214,3))
ax.set_xticklabels(x[8:])
ax.set_xlabel("Month-Year")
ax.set_ylabel("New confirmed cases")
plt.legend()
#plt.savefig('Q2bii.eps', format = 'eps')
plt.show()


#%% Part B (iii)
print('Part B (ii)')
# Computing RMSE(%) and MAPE value
rmse = math.sqrt(mean_squared_error(exp, predictions))/(np.mean(exp))
print("RMSE(%):", round(rmse*100,3))

mape = np.mean(np.abs((np.array(exp) - np.array(predictions))/np.array(exp)))*100
print("MAPE:", round(mape,3))

#%% Question 3
print()
print(" ___________________________________Question 3_________________________________")
# Similar to above question repeating it for lag 1, 5, 10, 15, 25.
# Plotting bar graphs for for RMSE(%) and MAPE error
l = [1, 5, 10, 15, 25]
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
	rmse = math.sqrt(mean_squared_error(exp, predictions)) / (np.mean(exp))
	rm.append(round(rmse*100,3))
	mape = np.mean(np.abs((np.array(exp) - np.array(predictions)) / np.array(exp))) * 100
	ma.append(round(mape,3))
for i in range(len(l)):
	print("RMSE(%) for lag ",l[i],":",rm[i])
print()
for i in range(len(l)):
	print("MAPE for lag ",l[i],":",ma[i])

plt.bar(l,rm, width = 1.5)
plt.xlabel("Lag values")
plt.ylabel("RMSE(%)")
#plt.savefig('Q3i.eps', format = 'eps')
plt.show()
plt.bar(l,ma, width = 1.5)
plt.xlabel("Lag values")
plt.ylabel("MAPE")
#plt.savefig('Q3ii.eps', format = 'eps')
plt.show()

#%% Question 4
# Computing the heuristic value
# Running our AR model for heurisic value
# Obtaining the RMSE(%) and MAPE error
print()
print(" ___________________________________Question 4_________________________________")
m = 2/np.sqrt(len(train))
i = 1

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

print ('The heuristic value for the optimal number of lags is: ',i)
window = i
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
rmse = math.sqrt(mean_squared_error(exp, predictions)) / (np.mean(exp))
print("RMSE(%) :",round(rmse*100,3))
mape = np.mean(np.abs((np.array(exp) - np.array(predictions)) / np.array(exp))) * 100
print("MAPE :",round(mape,3))


