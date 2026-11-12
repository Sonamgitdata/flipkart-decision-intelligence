
import numpy as np
import csv

# Part 1: Statistical Analysis and Regression

import pandas as pd
import numpy as np

df = pd.read_csv("data/records.csv")

print("Dataset loaded successfully")
print(df.head())

from statsmodels.stats.weightstats import ztest

sample = df["customer_tenure_months"].dropna()

population_mean = 100  # Replace with the specified benchmark

z_stat, p_value = ztest(
    sample,
    value=population_mean
)

print("Z Statistic:", z_stat)
print("P Value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")
import math
from scipy.stats import norm

confidence = 0.95
margin_error = 5
sigma = 20

z_value = norm.ppf(1 - (1-confidence)/2)

sample_size = math.ceil(
    (z_value * sigma / margin_error)**2
)

print("Required Sample Size:", sample_size)

x = df["customer_tenure_months"].to_numpy()
y = df["order_value_inr_hundreds"].to_numpy()

slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)

intercept = np.mean(y) - slope * np.mean(x)

print("Slope:", slope)
print("Intercept:", intercept)

predicted_y = intercept + slope*x

print("First 5 Predictions:", predicted_y[:5])

x = df["customer_tenure_months"].to_numpy(dtype=float)
y = df["order_value_inr_hundreds"].to_numpy(dtype=float)

# Standardize X for stable gradient descent
x_mean = x.mean()
x_std = x.std()

x_scaled = (x - x_mean) / x_std

m = 0
c = 0

learning_rate = 0.01
n = len(x_scaled)

for i in range(500):

    y_pred = m*x_scaled + c

    dm = (-2/n) * np.sum(x_scaled*(y-y_pred))
    dc = (-2/n) * np.sum(y-y_pred)

    m -= learning_rate * dm
    c -= learning_rate * dc

print("Slope:", m)
print("Intercept:", c)
print("Total Updates: 500")

import matplotlib.pyplot as plt
import seaborn as sns

residuals = y - predicted_y

plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(predicted_y, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.show()

# Replace these with actual numerical column names
features = ["customer_tenure_months", "avg_order_value_inr_hundreds"]
target = "order_value_inr_hundreds"

data = df[features + [target]].dropna()

X = data[features].to_numpy()
Y = data[target].to_numpy()

# Add intercept column
X = np.c_[np.ones(X.shape[0]), X]

# Normal equation using pseudo-inverse
theta = np.linalg.pinv(X.T @ X) @ X.T @ Y

print("Regression Coefficients:", theta)

predictions = X @ theta

print("First 5 Predictions:", predictions[:5])
