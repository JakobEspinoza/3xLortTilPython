#%% GET DATA


# imports
import numpy as np
import pandas as pd
import yfinance as yf

# get down jones tickers
djones_data = pd.read_excel(r'holdings-daily-us-en-dia.xlsx', skiprows=4)
djones_ticks = list(djones_data['Ticker'].dropna()) 

# collect adjusted prices
rawdata = pd.DataFrame(yf.download(
    tickers=djones_ticks,
    start="2000-01-01",
    end="2023-12-31"
))
adj_close_hist = rawdata['Adj Close'].copy()
adj_close_hist.columns = djones_ticks



#%% Problem 1

# remove non-continuos assets
prices = adj_close_hist.dropna(axis=1, how='any')
print(f'Number of assets in cleaned dataset: {len(prices.columns)}\n')

# calc monthly returns for each asset
monthly_returns = prices.resample('ME').last().pct_change()
monthly_returns = monthly_returns.dropna()


#%% Problem 2

# Compute the sample means
mu = monthly_returns.mean()

# Compute the variance-covariance matrix
Sigma = monthly_returns.cov()

# Calculate Sharpe ratio for each asset
sharpe_ratios = mu / monthly_returns.std()

# Identify the asset with the highest Sharpe ratio
max_sharpe_tick = sharpe_ratios.idxmax()
max_sharpe = sharpe_ratios.max()

print(f"The asset with the highest Sharpe ratio is {max_sharpe_tick} with a Sharpe ratio of {round(max_sharpe,2)}\n")

