import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import WLS
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.tools import add_constant

# Load the data
data = pd.read_csv('M2_module_2_data.csv')

# Handle missing values if any
data = data.dropna()

# Select features and target
X = data[['DXY', 'METALS', 'OIL', 'INTL_STK', 'X13W_TB', 'X10Y_TBY', 'EURUSD']]
y = data['US_STK']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add constant for intercept
X_train_const = add_constant(X_train)
X_test_const = add_constant(X_test)

# Fit LS model
ls_model = sm.OLS(y_train, X_train_const).fit()
ls_predictions = ls_model.predict(X_test_const)
ls_mse = np.mean((y_test - ls_predictions) ** 2)

# Get residuals from LS model to estimate weights for WLS
residuals = ls_model.resid
abs_residuals = np.abs(residuals)
weights = 1 / (abs_residuals + 1e-6)  # Add small value to avoid division by zero

# Fit WLS model
wls_model = WLS(y_train, X_train_const, weights=weights).fit()
wls_predictions = wls_model.predict(X_test_const)
wls_mse = np.mean((y_test - wls_predictions) ** 2)

# Perform F-test to compare models
n = len(y_test)
p = X_train_const.shape[1]  # Number of parameters including intercept
ssr_ls = np.sum((y_test - ls_predictions) ** 2)
ssr_wls = np.sum((y_test - wls_predictions) ** 2)

f_statistic = ((ssr_ls - ssr_wls) / p) / (ssr_wls / (n - p - 1))
p_value = 1 - sm.stats.f_distribution.cdf(f_statistic, p, n - p - 1)

print(f"LS MSE: {ls_mse}")
print(f"WLS MSE: {wls_mse}")
print(f"F-statistic: {f_statistic}, p-value: {p_value}")

if p_value < 0.05:
    print("WLS performs significantly better than LS.")
else:
    print("No significant difference between LS and WLS.")