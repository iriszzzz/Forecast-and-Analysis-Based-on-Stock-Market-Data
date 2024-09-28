#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load and print available input files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load data
data_path = '../input/asia-stock/Asia_Stock.xlsx'
jpn = pd.read_excel(data_path, usecols=['JAPAN']).to_numpy()
twn = pd.read_excel(data_path, usecols=['TAIWAN']).to_numpy()
kor = pd.read_excel(data_path, usecols=['KOREA']).to_numpy()
ind = pd.read_excel(data_path, usecols=['INDIA']).to_numpy()
dates = pd.read_excel(data_path, usecols=['Date'])

# Function to calculate and print error metrics
def q1Cal(act, pred):
    MSE = mean_squared_error(act, pred)
    RMSE = math.sqrt(MSE)
    MAE = mean_absolute_error(act, pred)
    MAPE = mean_absolute_percentage_error(act, pred)
    print(f"Root Mean Square Error: {RMSE}")
    print(f"Mean Absolute Error: {MAE}")
    print(f"Mean Absolute Percentage Error: {MAPE}")

# Calculate moving averages
def calculate_moving_average(data, window):
    moving_average = []
    for i in range(len(data)):
        if i < window:
            moving_average.append(float(data[window - 1]))
        else:
            moving_average.append(data[i - window:i].sum() / window)
    return moving_average

jpn10 = calculate_moving_average(jpn, 10)
jpn20 = calculate_moving_average(jpn, 20)

# Plot moving averages for Japan
def plot_moving_average(title, x, y_data, labels, colors):
    plt.figure(figsize=(15, 10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=180))
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Index")
    for y, label, color in zip(y_data, labels, colors):
        plt.plot(x, y, label=label, color=color)
    plt.grid(axis="y")
    plt.legend(loc=0, prop={'size': "x-large"})
    plt.show()

plot_moving_average("2016_Japanese_Stock", dates[:250], [jpn10[:250], jpn20[:250], jpn[:250]], 
                    ["Move 10 days", "Move 20 days", "Japan"], ["red", "blue", "green"])

# Print errors for Japan stock data
print("2016 Japan actual vs. Japan move 10 days")
q1Cal(jpn[:250], jpn10[:250])
print("2016 Japan actual vs. Japan move 20 days")
q1Cal(jpn[:250], jpn20[:250])

# Calculate exponential smoothing for India
def exponential_smoothing(data, alpha):
    smoothed_data = []
    for i in range(len(data)):
        if i == 0:
            smoothed_data.append(float(data[i]))
        else:
            smoothed_data.append(float(alpha * data[i - 1] + (1 - alpha) * smoothed_data[i - 1]))
    return smoothed_data

ind025 = exponential_smoothing(ind, 0.25)
ind045 = exponential_smoothing(ind, 0.45)

# Plot exponential smoothing for India
plot_moving_average("2016_India_Stock", dates[:250], [ind025[:250], ind045[:250], ind[:250]], 
                    ["Alpha = 0.25", "Alpha = 0.45", "India"], ["red", "blue", "green"])

# Print errors for India stock data
print("2016 India actual vs. India alpha = 0.25")
q1Cal(ind[:250], ind025[:250])
print("2016 India actual vs. India alpha = 0.45")
q1Cal(ind[:250], ind045[:250])

# Predict stock data using linear regression
def q2predict(act, name):
    pred_names = ['NASDAQ', 'US10YY', 'SHI', 'SHE', 'SOX', 'DJI', 'SP 500', 'USD', 'Oil.price', 'Gold.price', 
                  "JAPAN", "TAIWAN", "KOREA", "INDIA"]
    for idx, predictor in enumerate(predictor_list):
        x = predictor
        reg = LinearRegression().fit(x, act)
        pred = reg.predict(x)
        
        plt.figure(figsize=(12, 8))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365))
        plt.title(f"{name} stock and {pred_names[idx]}")
        plt.xlabel("Date")
        plt.ylabel("Index")
        plt.plot(dates, act, label='Actual')
        plt.plot(dates, pred, label=f"Use {pred_names[idx]} predict")
        plt.grid(axis="y")
        plt.legend(loc=2, prop={'size': "x-large"})
        plt.show()
        print(f"{pred_names[idx]} R squared: {reg.score(x, act)}")

# Summarize model results
def q2summary(act, name, alpha):
    df = pd.read_excel(data_path)
    df = df.drop(columns=[name])
    x = df.iloc[:, 1:15].values
    y = act
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    plt.figure(figsize=(12, 8))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365))
    plt.title(name)
    plt.xlabel("Date")
    plt.ylabel("Index")
    plt.plot(dates, y, label='Actual')
    plt.plot(dates, model.predict(), label="Multi predict")
    plt.grid(axis="y")
    plt.legend(loc=2, prop={'size': "x-large"})
    plt.show()
    print(f"Multi-predict {name}")
    q1Cal(y, model.predict())
    print(model.summary(alpha=alpha, yname=name))

# Load predictor data
nasdaq = pd.read_excel(data_path, usecols=['NASDAQ']).to_numpy()
us10yy = pd.read_excel(data_path, usecols=['US10YY']).to_numpy()
shi = pd.read_excel(data_path, usecols=['SHI']).to_numpy()
she = pd.read_excel(data_path, usecols=['SHE']).to_numpy()
sox = pd.read_excel(data_path, usecols=['SOX']).to_numpy()
dji = pd.read_excel(data_path, usecols=['DJI']).to_numpy()
sp500 = pd.read_excel(data_path, usecols=['SP 500']).to_numpy()
usd = pd.read_excel(data_path, usecols=['USD']).to_numpy()
oil = pd.read_excel(data_path, usecols=['Oil.price']).to_numpy()
gold = pd.read_excel(data_path, usecols=['Gold.price']).to_numpy()
predictor_list = [nasdaq, us10yy, shi, she, sox, dji, sp500, usd, oil, gold, jpn, twn, kor, ind]

# Predict and summarize Japan stock data
q2predict(jpn, "JAPAN")
q2summary(jpn, "JAPAN", 0.1)

# Predict and summarize Taiwan stock data
q2predict(twn, "TAIWAN")
q2summary(twn, "TAIWAN", 0.1)

# Predict and summarize Korea stock data
q2predict(kor, "KOREA")
q2summary(kor, "KOREA", 0.1)

# Predict and summarize India stock data
q2predict(ind, "INDIA")
q2summary(ind, "INDIA", 0.1)
