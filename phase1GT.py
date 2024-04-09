# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset (assuming it's in CSV format)
data = pd.read_csv('portfolio_data.csv')

df = pd.DataFrame(data)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Check the first few rows of the dataframe
print(df.head())

# Plot the closing prices over time for each stock
plt.figure(figsize=(10, 6))
for column in df.columns:
    if column != 'Date':
        plt.plot(df.index, df[column], label=column)
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Perform time series decomposition (trend, seasonal, residual) for each stock
num_stocks = len(df.columns) - 1  # Exclude the Date column
num_rows = num_stocks * 3  # Each stock has 3 subplots (original, trend, seasonal, residual)

fig, axs = plt.subplots(num_rows, 1, figsize=(12, num_rows * 4))

for i in range(num_stocks):
    column = df.columns[i+1]  # Skip the Date column
    decomposition = seasonal_decompose(df[column], model='additive', period=5) # Adjust period as needed
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    axs[i*3].plot(df.index, df[column], label='Original', color='blue')
    axs[i*3].set_title(f'{column} Original')
    axs[i*3].legend(loc='upper left')

    axs[i*3+1].plot(df.index, trend, label='Trend', color='green')
    axs[i*3+1].set_title(f'{column} Trend')
    axs[i*3+1].legend(loc='upper left')

    axs[i*3+2].plot(df.index, seasonal, label='Seasonal', color='red')
    axs[i*3+2].plot(df.index, residual, label='Residual', color='purple')
    axs[i*3+2].set_title(f'{column} Seasonal and Residual')
    axs[i*3+2].legend(loc='upper left')

plt.tight_layout()
plt.show()

