#%% GET DATA

# imports
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt

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
djones_ticks = list(prices.columns)
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


#%% Problem 3

'Func for computing efficient frontier'
def compute_efficient_frontier(Sigma_est, mu_est):

    # Number of assets
    N = len(mu_est)  
    
    # Objective function for minimum variance portfolio
    def min_variance(weights):
        return weights.T @ Sigma_est @ weights
    
    # Constraints
    weights_sum_to_one = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    
    # Initial guess for weights
    initial_guess = np.ones(N) / N
    
    # Bounds for weights
    bounds = tuple((0, 1) for asset in range(N))
    
    # Optimization for minimum variance portfolio
    min_var_result = sco.minimize(min_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=[weights_sum_to_one])
    omega_mvp = min_var_result.x
    
    # Compute expected return of MVP
    mvp_return = mu_est @ omega_mvp
    
    # Objective function for efficient portfolio
    def eff_portfolio(weights):
        return -mu_est @ weights  # We want to maximize return, hence the negative
    
    # Additional constraint for efficient portfolio to have double the MVP's return
    return_constraint = {'type': 'eq', 'fun': lambda weights: mu_est @ weights - 2 * mvp_return}
    
    # Optimization for efficient portfolio
    eff_var_result = sco.minimize(eff_portfolio, initial_guess, method='SLSQP', bounds=bounds, constraints=[weights_sum_to_one, return_constraint])
    omega_eff = eff_var_result.x
    
    # Compute range of portfolio weights using two-fund theorem
    c_values = np.arange(-0.1, 1.21, 0.1)  # Range of c values from -0.1 to 1.2
    portfolios = []

    for c in c_values:
        omega_c = c * omega_mvp + (1 - c) * omega_eff
        portfolios.append(omega_c)
    
    # Create a DataFrame to return the results
    df = pd.DataFrame(portfolios, columns=[f'Asset {i+1}' for i in range(N)])
    df['c'] = c_values
    
    return df[['c'] + [f'Asset {i+1}' for i in range(N)]]



#%% Problem 4



# get efficient_frontier df
efficient_frontier_df = compute_efficient_frontier(Sigma, mu)

# define assets in the df
assets = [f'Asset {i+1}' for i in range(len(mu))]

# Initialize returns and volatilit lists
expected_returns = []
volatilities = []

# Compute expected return and volatility for each value of c
for index, row in efficient_frontier_df.iterrows():
    omega_c = row[assets].values  # Portfolio weights
    expected_return = np.dot(omega_c, mu)  # Expected return
    volatility = np.sqrt(np.dot(omega_c.T, np.dot(Sigma, omega_c)))  # Volatility
    
    expected_returns.append(expected_return)
    volatilities.append(volatility)

# Plotting the efficient frontier with c's as labels for eahc datapoint
plt.figure(figsize=(12, 8))
for i, txt in enumerate(efficient_frontier_df['c']):
    plt.scatter(volatilities[i], expected_returns[i], c='blue', marker='o')
    plt.text(volatilities[i], expected_returns[i], f'{txt:.1f}', fontsize=9)

plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.grid(True)
plt.show()


#%% Problem 5

# Compute the inverse of the variance-covariance matrix
Sigma_inv = np.linalg.inv(Sigma)

# Compute the tangency portfolio weights
omega_tgc = np.dot(Sigma_inv, mu) / np.dot(np.ones(len(mu)), np.dot(Sigma_inv, mu))

# Compute the expected return and volatility of the tangency portfolio
expected_return_tgc = np.dot(omega_tgc, mu)
volatility_tgc = np.sqrt(np.dot(omega_tgc.T, np.dot(Sigma, omega_tgc)))

# Compute the Sharpe ratio of the tangency portfolio
sharpe_ratio_tgc = expected_return_tgc / volatility_tgc

# Plotting the efficient frontier with c's as labels for each datapoint
plt.figure(figsize=(12, 8))
for i, txt in enumerate(efficient_frontier_df['c']):
    plt.scatter(volatilities[i], expected_returns[i], c='blue', marker='o')
    plt.text(volatilities[i], expected_returns[i], f'{txt:.1f}', fontsize=9)

# Plotting the tangency portfolio on the efficient frontier
plt.scatter(volatility_tgc, expected_return_tgc, c='red', marker='*', s=150, label='Tangency Portfolio')
plt.legend()

plt.title('Efficient Frontier with Tangency Portfolio')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.grid(True)
plt.show()

# Output for tangency portfolio weights and Sharpe ratio
print("Tangency portfolio weights:", [round(o,2) for o in omega_tgc], '\n')
print("Maximum attainable Sharpe ratio:", round(sharpe_ratio_tgc,2), '\n')



#%% Problem 6


def simulate_returns(periods=200,
    expected_returns=mu,
    covariance_matrix=Sigma):
    """
    periods (int): Number of periods
    expected_returns (array-like): Expected returns for each asset
    covariance_matrix (array-like): Covariance matrix of returns
    """
    3
    returns = np.random.multivariate_normal(expected_returns,
    covariance_matrix,
    size=periods)
    return returns

'''
Simulation Process:
The function uses the np.random.multivariate_normal method from NumPy to generate random samples from a multivariate normal distribution. This method requires the mean (expected returns), the covariance matrix, and the number of samples (size) to generate.
expected_returns serves as the mean of the distribution, indicating the average return expected for each asset.
covariance_matrix provides the covariances between the asset returns, which are essential for capturing the relationships between different assets' performances.
periods determines how many sets of returns will be generated, with each set containing a return value for each asset.

Output:
The function returns a NumPy array with dimensions 
periods
periods x N, where each row represents a set of simulated returns for all assets in one period, and 
N is the number of assets.

In summary, the simulate_returns function is a tool for generating synthetic asset return data based on specified expected returns and covariance among the assets. 
This simulated data can be used to study the properties of financial models, test investment strategies, 
or understand the impact of estimation uncertainty on portfolio optimization, such as the construction of efficient frontiers.
'''

# Step 1: Generate simulated returns
simulated_returns = simulate_returns(periods=200, expected_returns=mu, covariance_matrix=Sigma)

# Step 2: Compute sample mean and variance-covariance matrix
sample_mu = np.mean(simulated_returns, axis=0)
sample_Sigma = np.cov(simulated_returns, rowvar=False)

# Compute the efficient frontier for the simulated data (using the sample estimates)
sample_efficient_frontier_df = compute_efficient_frontier(sample_Sigma, sample_mu)

# Compute expected returns and volatilities for the simulated efficient frontier
sample_expected_returns = []
sample_volatilities = []

for index, row in sample_efficient_frontier_df.iterrows():
    omega_c = row[assets].values  # Portfolio weights
    sample_expected_return = np.dot(omega_c, sample_mu)  # Expected return
    sample_volatility = np.sqrt(np.dot(omega_c.T, np.dot(sample_Sigma, omega_c)))  # Volatility
    sample_expected_returns.append(sample_expected_return)
    sample_volatilities.append(sample_volatility)

# Plotting both the true and simulated efficient frontiers
plt.figure(figsize=(12, 8))

# True Efficient Frontier
plt.scatter(volatilities, expected_returns, c='blue', marker='o', label='True Efficient Frontier')

# Simulated Efficient Frontier
plt.scatter(sample_volatilities, sample_expected_returns, c='green', marker='x', label='Simulated Efficient Frontier')

# Tangency Portfolio on the True Frontier
plt.scatter(volatility_tgc, expected_return_tgc, c='red', marker='*', s=150, label='Tangency Portfolio')

plt.title('True vs. Simulated Efficient Frontiers')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True)
plt.show()



#%% Problem 7

plt.figure(figsize=(14, 10))

# Plot the true efficient frontier
plt.scatter(volatilities, expected_returns, c='blue', marker='o', label='True Efficient Frontier')

# Repeat the simulation step 100 times
for simulation in range(100):
    # Generate simulated returns
    simulated_returns = simulate_returns(periods=200, expected_returns=mu, covariance_matrix=Sigma)
    
    # Compute sample mean and variance-covariance matrix
    sample_mu = np.mean(simulated_returns, axis=0)
    sample_Sigma = np.cov(simulated_returns, rowvar=False)
    
    # Compute the efficient frontier for the simulated data
    sample_efficient_frontier_df = compute_efficient_frontier(sample_Sigma, sample_mu)
    
    # Compute expected returns and volatilities for the simulated efficient frontier
    sample_expected_returns = []
    sample_volatilities = []

    for index, row in sample_efficient_frontier_df.iterrows():
        omega_c = row[assets].values  # Portfolio weights
        sample_expected_return = np.dot(omega_c, sample_mu)  # Expected return
        sample_volatility = np.sqrt(np.dot(omega_c.T, np.dot(sample_Sigma, omega_c)))  # Volatility
        sample_expected_returns.append(sample_expected_return)
        sample_volatilities.append(sample_volatility)
    
    # Plot each simulated efficient frontier
    plt.scatter(sample_volatilities, sample_expected_returns, c='grey', alpha=0.1)

# Final plot adjustments
plt.title('True vs. Simulated Efficient Frontiers (100 Simulations)')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True)
plt.show()