# Forecasting Using R

Editor: Shawn Ng<br>
Content Author: **Rob J. Hyndman**<br>
[Site](https://www.datacamp.com/courses/forecasting-using-r)<br>

1. [Exploring and visualizing time series in R](#1-exploring-and-visualizing-time-series-in-r)
	- [Ljung-Box test](#ljung-box-test)
2. [Benchmark methods and forecast accuracy](#2-benchmark-methods-and-forecast-accuracy)
	- Forecast is the __mean/median__ of simulated futures of a time series
	- [Naive forecast](#naive-forecast): Uses the most recent observation
	- [Time series residuals](#time-series-residuals)
		- Fitted value: 1-step forecast of all previous observations, not true forecast as all params are estimated
		- Residuals: 1-step forecast error, diff bet fitted value and obs
	- Forecast error: diff bet. observed value and its multi-steps forecast in the test set
		- It is not residuals
		- Errors on training vs test set
	- Measures of forecast accuracy
		1. Mean abs error (MAE)
		2. Mean sq error (MSE)
		3. Mean abs percentage error (MAPE)
		4. Mean abs scaled error (MASE)
	- [Evaluating forecast accuracy](#evaluating-forecast-accuracy)
		- Training set: is a data set that is used to discover possible relationships
		- Test set: is a data set that is used to verify the strength of these potential relationships
	- [Time series cross-validation: tsCV()](#time-series-cross-validation-tscv)
3. [Exponential smoothing](#3-exponential-smoothing)
	- [SES VS NAIVE](#ses-vs-naive)
	- [Holt's trend methods](#holts-trend-methods)
	- [Fit errors, trend, and seasonality (ETS) model to data](#fit-errors-trend-and-seasonality-ets-model-to-data)
	- [ETS VS seasonal naive](#ets-vs-seasonal-naive)
4. [Forecasting with ARIMA models](#4-forecasting-with-arima-models)
	- [Box-Cox transformations](#box-cox-transformations): stabilize the variance
	- [Non-seasonal differencing for stationarity](#non-seasonal-differencing-for-stationarity)
		- Differencing is a way of making a time series stationary; this means that you remove any systematic patterns such as trend and seasonality from the data
		- A white noise series is considered a special case of a stationary time series
	- [Seasonal differencing for stationarity](#seasonal-differencing-for-stationarity)
		- For example, with quarterly data, one would take the difference between Q1 in one year and Q1 in the previous year. This is called seasonal differencing
		- Sometimes you need to apply both seasonal differences and lag-1 differences to the same series, thus, calculating the differences in the differences
	- [Automatic ARIMA models for non-seasonal time series](#automatic-arima-models-for-non-seasonal-time-series)
		- `auto.arima()` function will select an appropriate autoregressive integrated moving average (ARIMA) model given a time series
	- [Forecasting with ARIMA models](#forecasting-with-arima-models)
		- The `arima()` function can be used to select a specific ARIMA model
		- Its first argument, order, is set to a vector that specifies the values of `p`, `d` and `q`
		- The second argument, `include.constant`, is a booolean that determines if the constant cc, or drift, should be included
	- [Comparing auto.arima() and ets() on non-seasonal data](#comparing-autoarima-and-ets-on-non-seasonal-data)
		- The AICc statistic is useful for selecting between __models in the same class__
		- For example, you can use it to select an ETS model or to select an ARIMA model 
		- However, you cannot use it to compare ETS and ARIMA models because they are in different model classes
		- Use __time series cross-validation__ to compare an ARIMA model and an ETS model
	- [Automatic ARIMA models for seasonal time series](#automatic-arima-models-for-seasonal-time-series)
		- Note that setting `lambda = 0` in the `auto.arima()` function - applying a log transformation - means that the model will be fitted to the transformed data, and that the forecasts will be back-transformed onto the original scale
	- [Exploring auto.arima() options](#exploring-autoarima-options)
		- The `auto.arima()` function needs to estimate a lot of different models, and various short-cuts are used to try to make the function as fast as possible
		- This can cause a model to be returned which does not actually have the smallest AICc value. To make `auto.arima()` work harder to find a good model, add the optional argument `stepwise = FALSE` to look at a much larger collection of models
	- [Comparing auto.arima() and ets() on seasonal data](#comparing-autoarima-and-ets-on-seasonal-data)
		- If the series is very long, you can afford to use a training and test set rather than time series cross-validation. This is much faster
5. [Advanced methods](#5-advanced-methods)
	- [Forecasting sales allowing for advertising expenditure](#forecasting-sales-allowing-for-advertising-expenditure)
	- [Forecasting weekly data](#forecasting-weekly-data)
		- With weekly data, it is difficult to handle seasonality using ETS or ARIMA models as the seasonal length is too large (approximately 52). Instead, you can use harmonic regression which uses sines and cosines to model the seasonality
	- [Harmonic regression for multiple seasonality](#harmonic-regression-for-multiple-seasonality)
		- Harmonic regressions are also useful when time series have multiple seasonal patterns. `auto.arima()` would take a long time to fit a long time series. Instead you will fit a standard regression model with Fourier terms using the `tslm()` function
	- [TBATS models](#tbats-models)
		- TBATS model is a special kind of time series model. It can be very slow to estimate, especially with multiple seasonal time series





## 1. Exploring and visualizing time series in R
```r
library(forecast)
library(ggplot2)
library(fpp2)

autoplot(ts_data)

# To find outlier
which.max(ts_data)

# No. of obs per unit time
frequency(ts_data)

ggseasonplot(ts_data)
ggseasonplot(ts_data, polar=T)

# Restrict the data to start in year
data_subset <- window(ts_data, start=year)
ggsubseriesplot(data_subset)
```

When data are either seasonal/cyclic, the ACF will peak around the seasonal lags or at the average cycle length.

### Ljung-Box test
```r
autoplot(ts_data)

# Plots a lag plot using ggplot
gglagplot(ts_data)

ggAcf(ts_data)

# Ljung-Box test confirms the randomness of a series
# p-value > 0.05 suggests that the ts_data are not significantly different from white noise
Box.test(diff(ts_data), lag = 10, type = "Ljung")
```





## 2. Benchmark methods and forecast accuracy
### Naive forecast
```r
data_naive <- naive(ts_data, h=20)
data_seasonal_naive <- snaive(ts_data, h = 16)

autoplot(ts_data)
summary(ts_data)
```

### Time series residuals
```r
# p-value >= 0.05 -> white noise
checkresiduals(naive(ts_data))

# same as %>% pipe operator
ts_data %>% naive() %>% checkresiduals()


checkresiduals(snaive(ts_data))
# same as
ts_data %>% snaive() %>% checkresiduals()
```

### Evaluating forecast accuracy
We are interested to see the forecast accuracy of test set<br>
__Root mean squared error (RMSE)__ which is the square root of the mean squared error (MSE). Smaller RMSE = better accuracy

```r
## Forecast accuracy of non-seasonal methods
# assume data has 1108 data points
train <- subset.ts(ts_data, end=1000)

# naive forecast, simple forcasting functions, h=1108-1000
naive_fc <- naive(train, h=108)

# meanf(): gives forecasts equal to the mean of all observations
mean_fc <- meanf(train, h=108)

# smaller root mean squared error (RMSE) -> better accuracy
accuracy(naive_fc, ts_data)
accuracy(mean_fc, ts_data)


## Forecast accuracyy of seasonal methods
# Create three training series omitting the last 1, 2, and 3 years
train1 <- window(ts_data[, "COLUMN"], end = c(2014, 4))
train2 <- window(ts_data[, "COLUMN"], end = c(2013, 4))
train3 <- window(ts_data[, "COLUMN"], end = c(2012, 4))

# Produce forecasts using snaive()
fc1 <- snaive(train1, h = 4)
fc2 <- snaive(train2, h = 4)
fc3 <- snaive(train3, h = 4)

# Use accuracy() to compare the MAPE of each series
accuracy(fc1, ts_data[, "COLUMN"])["Test set", "MAPE"]
accuracy(fc2, ts_data[, "COLUMN"])["Test set", "MAPE"]
accuracy(fc3, ts_data[, "COLUMN"])["Test set", "MAPE"]
```

Good forecasting model?
* Small residuals only means that the model fits the training data well, not that it produces good forecasts.
* A good model forecasts has low RMSE on the test set and has white noise residuals

### Time series cross-validation: tsCV()
```r
# Compute cross-validated errors for up to 8 steps ahead
e <- matrix(NA_real_, nrow = 1000, ncol = 8)

for (h in 1:8)
	e[, h] <- tsCV(ts_data, forecastfunction = naive, h = h)

# Compute the MSE values and remove missing values
mse <- colMeans(e^2, na.rm = TRUE)

# Plot the MSE values against the forecast horizon
data.frame(h = 1:8, MSE = mse) %>% ggplot(aes(x = h, y = MSE)) + geom_point()
```





## 3. Exponential smoothing
```r
# ses(): simple exponential smoothing. The parameters are estimated using least squares estimation.
# h: horizon, years
fc <- ses(ts_data, h=10)

# smoothing parameters, alpha=0.3457 -> 34.57% of emphasis on latest data
summary(fc)

# Add 1 step forecast for training data
autoplot(fc) + autolayer(fitted(fc))
```

### SES VS NAIVE
```r
train <- subset.ts(ts_data, end=length(ts_data)-length(test))

fcSes <- ses(train, h=PERIOD)
fcNaive <- naive(train, h=PERIOD)
accuracy(fcSes, ts_data)
accuracy(fcNaive, ts_data)
```

### Holt's trend methods
```r
fcholt <- holt(ts_data, h=PERIOD)
summary(fcholt)
autoplot(fcholt)
checkresiduals(fcholt)

# Holt with monthly data
fc <- holt(ts_data, seasonal="multiplicative", h=12)
```

### Fit errors, trend, and seasonality (ETS) model to data
```r
fitdata <- ets(ts_data)
checkresiduals(fitdata)

# p value > 0.05 => model fails the Ljung-Box test
autoplot(forecast(fitdata))
```

### ETS VS seasonal naive
```r
# Function to return ETS forecasts
fets <- function(y, h) {
  forecast(ets(y), h = h)
}

# Apply tsCV() for both methods
e1 <- tsCV(ts_data, fets, h = 4)
e2 <- tsCV(ts_data, snaive, h = 4)

# Compute MSE of resulting errors
mean(e1^2, na.rm=TRUE)
mean(e2^2, na.rm=TRUE)
```





## 4. Forecasting with ARIMA models
### Box-Cox transformations
```r
# Try four values of lambda in Box-Cox transformations
ts_data %>% BoxCox(lambda = 0.0) %>% autoplot()
ts_data %>% BoxCox(lambda = 0.1) %>% autoplot()
ts_data %>% BoxCox(lambda = 0.2) %>% autoplot()
ts_data %>% BoxCox(lambda = 0.3) %>% autoplot()

# Compare with BoxCox.lambda()
BoxCox.lambda(ts_data)
```

### Non-seasonal differencing for stationarity
```r
# Plot the rate
autoplot(ts_data)

# Plot the differenced rate
autoplot(diff(ts_data))

# Plot the ACF of the differenced rate
ggAcf(diff(ts_data))
```

### Seasonal differencing for stationarity
```r
# Plot the data
autoplot(ts_data)

# Take logs and seasonal differences of ts_data
difflogData <- diff(log(ts_data), lag = 12)

# Plot difflogData
autoplot(difflogData)

# Take another difference and plot
ddifflogData <- diff(difflogData)
autoplot(ddifflogData)

# Plot ACF of ddifflogData
ggAcf(ddifflogData)
```

### Automatic ARIMA models for non-seasonal time series
```r
# Fit an automatic ARIMA model to the ts_data series
fit <- auto.arima(ts_data)

# Check that the residuals look like white noise
checkresiduals(fit)
summary(fit)

# Plot forecasts of fit
fit %>% forecast(h = 10) %>% autoplot()
```

### Forecasting with ARIMA models
```r
# Plot forecasts from an ARIMA(0,1,1) model with no drift
ts_data %>% Arima(order = c(0,1,1), include.constant = FALSE) %>% forecast() %>% autoplot()

# Plot forecasts from an ARIMA(2,1,3) model with drift
ts_data %>% Arima(order = c(2,1,3), include.constant = TRUE) %>% forecast() %>% autoplot()

# Plot forecasts from an ARIMA(0,0,1) model with a constant
ts_data %>% Arima(order = c(0,0,1), include.constant = TRUE) %>% forecast() %>% autoplot()

# Plot forecasts from an ARIMA(0,2,1) model with no constant
ts_data %>% Arima(order = c(0,2,1), include.constant = FALSE) %>% forecast() %>% autoplot()
```

### Comparing auto.arima() and ets() on non-seasonal data
```r
# Set up forecast functions for ETS and ARIMA models
fets <- function(x, h) {
  forecast(ets(x), h = h)
}
farima <- function(x, h) {
  forecast(auto.arima(x), h=h)
}

# Compute CV errors for ETS as e1
e1 <- tsCV(ts_data, fets, 1)

# Compute CV errors for ARIMA as e2
e2 <- tsCV(ts_data, farima, 1)

# Find MSE of each model class
mean(e1^2, na.rm=TRUE)
mean(e2^2, na.rm=TRUE)

# Plot 10-year forecasts using the best model class
ts_data %>% farima(h=10) %>% autoplot()
```

### Automatic ARIMA models for seasonal time series
```r
# Check that the logged ts_data have stable variance
ts_data %>% log() %>% autoplot()

fit <- auto.arima(ts_data, lambda=0)
summary(fit)

# Plot 2-year forecasts
fit %>% forecast(h=24) %>% autoplot()
```

### Exploring auto.arima() options
```r
auto.arima(ts_data, stepwise = FALSE)
```

### Comparing auto.arima() and ets() on seasonal data
```r
# Use 20 years of the ts_data beginning in 1988
train <- window(ts_data, start = c(1988,1), end = c(2007,4))

# Fit an ARIMA and an ETS model to the training data
fit1 <- auto.arima(train)
fit2 <- ets(train)

# Check that both models have white noise residuals
checkresiduals(fit1)
checkresiduals(fit2)

# Produce forecasts for each model
fc1 <- forecast(fit1, h = 25)
fc2 <- forecast(fit2, h = 25)

# Use accuracy() to find better model based on RMSE
accuracy(fc1, ts_data)
accuracy(fc2, ts_data)
```





## 5. Advanced methods
### Forecasting sales allowing for advertising expenditure
```r
# Time plot of both variables
autoplot(advert, facets = TRUE)

# Fit ARIMA model
fit <- auto.arima(advert[, "sales"], xreg = advert[, "advert"], stationary = TRUE)

# Check model. Increase in sales for each unit increase in advertising
salesincrease <- fit$coef[3]

# Forecast fit as fc
fc <- forecast(fit, xreg = rep(10,6))

# Plot fc with x and y labels
autoplot(fc) + xlab("Month") + ylab("Sales")
```

### Forecasting weekly data
```r
# Set up harmonic regressors of order 13
harmonics <- fourier(ts_data, K = 13)

# Fit regression model with ARIMA errors
fit <- auto.arima(ts_data, xreg = harmonics, seasonal = FALSE)

# Forecasts next 3 years
newharmonics <- fourier(ts_data, K = 13, h = 3*52)
fc <- forecast(fit, xreg = newharmonics)

# Plot forecasts fc
autoplot(fc)
```

### Harmonic regression for multiple seasonality
```r
# Fit a harmonic regression using order 10 for each type of seasonality
fit <- tslm(multiSeasonalTS ~ fourier(multiSeasonalTS, K = c(10, 10)))

# Forecast 20 working days ahead, data is half hour
fc <- forecast(fit, newdata = data.frame(fourier(multiSeasonalTS, K = c(10, 10), h = 20*24*2)))

# Plot the forecasts
autoplot(fc)

# Check the residuals of fit
checkresiduals(fit)


# Plot the multiSeasonalTS
autoplot(multiSeasonalTS)

# Set up the xreg matrix
xreg <- fourier(multiSeasonalTS, K = c(10,0))

# Fit a dynamic regression model
fit <- auto.arima(multiSeasonalTS, xreg = xreg, seasonal = FALSE, stationary = TRUE)

# Check the residuals
checkresiduals(fit)

# Plot forecasts for 10 working days ahead
fc <- forecast(fit, xreg =  fourier(multiSeasonalTS, c(10, 0), h = 169*10))
autoplot(fc)
```

### TBATS models
```r
# Plot the ts_data
autoplot(ts_data)

# Fit a TBATS model to the ts_data
fit <- tbats(ts_data)

# Forecast the series for the next 5 years
fc <- forecast(fit, h=5*12)

# Plot the forecasts
autoplot(fc)
```