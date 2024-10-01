# Forecast-and-Analysis-Based-on-Stock-Market-Data
> This record and share the project of [Operation Management course](https://timetable.nycu.edu.tw/?r=main/crsoutline&Acy=112&Sem=1&CrsNo=517411&lang=zh-tw) about useing Python to perform predictive analysis on Asian stock markets, including the following main components:

### 1. Moving Average Prediction (Japanese Stock Market)

- Simple Moving Average (SMA) used for 10-day and 20-day predictions
- Visualization of prediction results
- Performance evaluation using RMSE, MAE, and MAPE<br>
- Key findings:
  - [x] Short-term (10-day) moving average is more sensitive to price changes
  - [x] Long-term (20-day) moving average better reflects long-term trends

### 2. Exponential Smoothing Prediction (Indian Stock Market)

- Exponential smoothing method applied with λ values of 0.25 and 0.45
- Visualization of prediction results
- Performance evaluation using RMSE, MAE, and MAPE
- Key findings:
  - [x] Higher λ values result in predictions more sensitive to recent data
  - [x] Predictions with λ = 0.45 are closer to actual values than those with λ = 0.25

### 3. Multiple Linear Regression Analysis

Analysis conducted for Taiwan, Korea, Japan, and India:

- R-squared value calculation
- Identification of significant predictors
- Comparison of significant predictors across different countries
- Key findings:
  - [x] Taiwan and Korean markets are significant predictors for each other
  - [x] U.S. indices (e.g., Dow Jones, S&P 500) significantly influence Asian markets
  - [x] Influencing factors vary across countries, reflecting differences in economic structures

### 4. Comparison of Prediction Methods

Accuracy comparison of different prediction methods:

1. Exponential smoothing (α = 0.45)
2. Exponential smoothing (α = 0.25)
3. 10-day Moving Average
4. Multiple Linear Regression
5. 20-day Moving Average <br>

💡Conclusion : Exponential smoothing methods generally perform best, while multiple linear regression and long-term moving averages show relatively poorer performance.

---


👨‍🏫 Advicing Professor : ZHI-XUAN WANG

###### tags:  `Production and Operation Management` `Statsmodels API`  `sklearn` `matplotlib` `numpy` `SP500` `Linear Regression` `Exponential Smoothing`

> 🔍 Watch MORE ➜ [My GitHub](https://github.com/iriszzzz)
