import pandas as pd
import numpy as np

data = pd.read_csv('Data.csv', sep=';')

n_forecast = 30  
n_vintage = data.shape[0]  
discount_rate = 0.025  

historical_cf = data.iloc[:, 2:].values
amount_originated = data.iloc[:, 1].values.reshape(-1, 1)  

periods_remaining = n_forecast - np.arange(n_vintage, 0, -1)

paid_percentages = historical_cf / amount_originated

first_period = np.diagonal(paid_percentages)
paid_subset = paid_percentages[:-1, 1:] 
second_period = np.append(np.diagonal(paid_subset), 2 * first_period[-1])

p = np.zeros((n_vintage, n_forecast))
p[:, 0] = first_period
p[:, 1] = second_period

for i in range(n_vintage):
    for j in range(2, n_forecast): 
        cumulative_repayment = np.sum(p[i, :j])
        log_term = np.log(1 + (1 - cumulative_repayment))
        decay_factor = 1 - j / n_forecast
        term = p[i, 1] * log_term * decay_factor
        p[i, j] = max(0, term)

p_forecast = np.zeros((n_vintage, n_forecast - 1))
for i in range(n_vintage):
    pr = periods_remaining[i]
    start_col = n_forecast - pr
    for j in range(pr):
        p_forecast[i, j] = p[i, start_col + j]

months = np.arange(1, n_forecast)
discount_factors = 1 / (1 + discount_rate) ** (months / 12)

pv = (p_forecast * discount_factors) * amount_originated
portfolio_value = np.sum(pv)

client_estimate = 84993122.67
absolute_diff = abs(portfolio_value - client_estimate)
relative_diff = absolute_diff / client_estimate

print(f"Calculated Portfolio Value: {round(portfolio_value, 2)} CHF")
print(f"Absolute Difference: {absolute_diff:.2f} CHF")
print(f"Relative Difference: {relative_diff:.2%}")
print(f"Threshold Acceptable: {absolute_diff < 500000}")