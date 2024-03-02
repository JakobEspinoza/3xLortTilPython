#%%

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


#%%
# remove non-continuos assets
prices = adj_close_hist.dropna(axis=1, how='any')
print(f'Number of assets in cleaned dataset: {len(prices.columns)}')