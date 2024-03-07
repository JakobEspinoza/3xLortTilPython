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

#%%  Problem 1

# remove non-continuos assets
prices = adj_close_hist.dropna(axis=1, how='any')
djones_ticks = list(prices.columns)
print(f'Number of assets in cleaned dataset: {len(prices.columns)}\n')

# calculate the monthly returns for each asset
prices.reset_index(inplace=True)
prices['Date'] = pd.to_datetime(prices['Date']) # This is done because python does not read the index as a date. 
prices.set_index('Date', inplace=True)
monthly_returns = prices.resample('M').last().pct_change()

#%% Problem 2

# Compute the sample means.
mu = monthly_returns.mean()

# To compute the variance-covariance matrix, we simply use the function cov which returns the covariance matrix
Sigma = monthly_returns.cov()

# Calculate Sharpe ratio for each asset using the formula for sharpe ratios
sharpe_ratios = mu / monthly_returns.std()

# Identifying the asset with the highest Sharpe ratio
max_sharpe_tick = sharpe_ratios.idxmax()
max_sharpe = sharpe_ratios.max()

print(f"The asset with the highest Sharpe ratio is {max_sharpe_tick} with a Sharpe ratio of {round(max_sharpe,2)}\n")

#%% Problem 3

def compute_efficient_frontier(Sigma_est, mu_est):
    N = len(mu_est)
    min_variance = lambda weights: weights.T @ Sigma_est @ weights # Calculating the weights by performing matrix calculations
    eff_portfolio = lambda weights: -mu_est @ weights # We wish to minimize, hence the negative
    
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    bounds = tuple((0, 1) for _ in range(N)) # Weights cannot exceed 100%, therefor we set the boundaries
    initial_guess = np.ones(N) / N
    
    min_var_result = sco.minimize(min_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints) # finding the portfolio with minimum variance using scipy
    omega_mvp = min_var_result.x # Extracting the weights for the minimum variance portfolio
    mvp_return = mu_est @ omega_mvp # Calculating the return
    
    return_constraint = {'type': 'eq', 'fun': lambda weights: mu_est @ weights - 2 * mvp_return} # Defining the equality constraint to reach target return (2 * MVP return)
    eff_var_result = sco.minimize(eff_portfolio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints + [return_constraint]) # finding the portfolio with highest sharpe ration using scipy
    omega_eff = eff_var_result.x # Extracting the weights for the minimum variance portfolio
    
    c_values = np.arange(-0.1, 1.21, 0.1) # Setting the given interval for values of c. 
    portfolios = [(c * omega_mvp + (1 - c) * omega_eff) for c in c_values]
    
    df = pd.DataFrame(portfolios, columns=[f'Asset {i+1}' for i in range(N)])
    df['c'] = c_values
    
    return df[['c'] + [f'Asset {i+1}' for i in range(N)]]

# Using the function to calculate the effeicient frontier
efficient_frontier_df = compute_efficient_frontier(Sigma, mu)

assets = [f'Asset {i+1}' for i in range(len(mu))]
expected_returns = []
volatilities = []

#Iterate over rows in the efficient_frontier_df DataFrame 
for _, row in efficient_frontier_df.iterrows():
    omega_c = row[assets].values # Finding portfolio weights for current row
    expected_returns.append(np.dot(omega_c, mu)) # Expected return for the portfolio
    volatilities.append(np.sqrt(np.dot(omega_c.T, np.dot(Sigma, omega_c)))) # Volatility for the portfolio

# Plotting the figure of the efficient frontier
plt.figure(figsize=(12, 8))
plt.scatter(volatilities, expected_returns, c='blue', marker='o')

plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.grid(True)
plt.show()


#%% Problem 4
# risk free asset is assumed to be zero
rf = 0

# To calculate tangency portfolio weights, we first have to find the inverse of the variance covariance matrix, and define "iota" as a column of ones
Sigma_inv = np.linalg.inv(Sigma)
iota = np.ones(len(mu))

# Computing the tangency portfolio weights
omega_tgc = (Sigma_inv @ (mu -rf*iota)) / np.dot(iota, np.dot(Sigma_inv, mu))
sum_omega_tgc = omega_tgc.sum()
print(f" The weights of the tangency portfolio sum to: {round(sum_omega_tgc,2)}\n")

# using the eqights we can calculate the expected return and volatility of the tangency portfolio
expected_return_tgc = np.dot(mu, omega_tgc)
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

#%% Problem 5

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
# Problem 6

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

# Running the simulation step 100 times
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

#%% Problem 8 and 9
sharpe_ratios_simulated = []

# We run the simulation again, but in this for loop we calculate the efficient tangent portfolio and the annualized sharpe ratio
for simulation in range(100):
    # Generate simulated returns
    simulated_returns = simulate_returns(periods=200, expected_returns=mu, covariance_matrix=Sigma)
    
    # Compute sample mean and variance-covariance matrix
    sample_mu = np.mean(simulated_returns, axis=0)
    sample_Sigma = np.cov(simulated_returns, rowvar=False)
    
    # Calculate tangency portfolio weights
    Sigma_inv = np.linalg.inv(sample_Sigma)
    iota = np.ones(len(sample_mu))
    numerator = Sigma_inv @ (sample_mu - rf * iota)
    denominator = iota.T @ Sigma_inv @ (sample_mu - rf * iota)
    omega_tgc = numerator / denominator
    
    # Calculate expected return and volatility of the efficient portfolio using true parameters
    expected_return_efficient_portfolio = mu @ omega_tgc
    volatility_efficient_portfolio = np.sqrt(omega_tgc.T @ Sigma @ omega_tgc)
    
    # Calculate annualized Sharpe ratio
    sharpe_ratio = np.sqrt(12) * expected_return_efficient_portfolio / volatility_efficient_portfolio
    
    # Append Sharpe ratio to the list
    sharpe_ratios_simulated.append(sharpe_ratio)

# Plot the distribution of simulated Sharpe ratios
plt.figure(figsize=(10, 6))
plt.hist(sharpe_ratios_simulated, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='Simulated Sharpe Ratios')
# We have the sharpe ratio of the tangent portfolio from ealier, why we can plot it in the figure as a dotted line. 
plt.axvline(x=sharpe_ratio_tgc, color='red', linestyle='--', linewidth=2, label='Tangency Portfolio Sharpe Ratio')

plt.xlabel('Simulated Annualized Sharpe Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of Simulated Annualized Sharpe Ratios')
plt.legend()
plt.grid(True)
plt.show()

#%% Problem 10
sharpe_ratios_simulated_2 = []

# We run the simulation again, but in this for loop we calculate the efficient tangent portfolio and the annualized sharpe ratio with a higher number of periods. this is run with 1000, but can be done with even higher numbers
for simulation in range(100):
    # Generate simulated returns
    simulated_returns = simulate_returns(periods=1000, expected_returns=mu, covariance_matrix=Sigma)
    
    # Compute sample mean and variance-covariance matrix
    sample_mu = np.mean(simulated_returns, axis=0)
    sample_Sigma = np.cov(simulated_returns, rowvar=False)
    
    # Calculate tangency portfolio weights
    Sigma_inv = np.linalg.inv(sample_Sigma)
    iota = np.ones(len(sample_mu))
    numerator = Sigma_inv @ (sample_mu - rf * iota)
    denominator = iota.T @ Sigma_inv @ (sample_mu - rf * iota)
    omega_tgc = numerator / denominator
    
    # Calculate expected return and volatility of the efficient portfolio using true parameters
    expected_return_efficient_portfolio = mu @ omega_tgc
    volatility_efficient_portfolio = np.sqrt(omega_tgc.T @ Sigma @ omega_tgc)
    
    # Calculate annualized Sharpe ratio
    sharpe_ratio = np.sqrt(12) * expected_return_efficient_portfolio / volatility_efficient_portfolio
    
    # Append Sharpe ratio to the list
    sharpe_ratios_simulated_2.append(sharpe_ratio)

# Plot the distribution of simulated Sharpe ratios
plt.figure(figsize=(10, 6))
plt.hist(sharpe_ratios_simulated_2, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='Simulated Sharpe Ratios')
# We have the sharpe ratio of the tangent portfolio from ealier, why we can plot it in the figure as a dotted line. 
plt.axvline(x=sharpe_ratio_tgc, color='red', linestyle='--', linewidth=2, label='Tangency Portfolio Sharpe Ratio')

plt.xlabel('Simulated Annualized Sharpe Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of Simulated Annualized Sharpe Ratios')
plt.legend()
plt.grid(True)
plt.show()