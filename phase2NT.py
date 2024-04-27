import pandas as pd
import matplotlib.pyplot as plt

# Load the climate dataset
data = pd.read_csv('monthly_data.csv')

# Data cleaning and preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])  # Convert 'DATE' column to datetime format

# Convert numeric columns to appropriate data types
numeric_cols = ['MonthlyMeanTemperature', 'MonthlyTotalLiquidPrecipitation']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Basic data exploration
print(data.head())
print(data.info())

# Statistical summary
print(data.describe())

# Visualize temperature trend over time
plt.figure(figsize=(10, 6))
plt.plot(data['DATE'], data['MonthlyMeanTemperature'], color='blue')
plt.title('Temperature Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

# Visualize precipitation trend over time
plt.figure(figsize=(10, 6))
plt.plot(data['DATE'], data['MonthlyTotalLiquidPrecipitation'], color='green')
plt.title('Precipitation Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.grid(True)
plt.show()

# Monthly average temperature
monthly_avg_temp = data.groupby(data['DATE'].dt.to_period('M'))['MonthlyMeanTemperature'].mean()
print(monthly_avg_temp)

# Monthly total precipitation
monthly_total_precipitation = data.groupby(data['DATE'].dt.to_period('M'))['MonthlyTotalLiquidPrecipitation'].sum()
print(monthly_total_precipitation)