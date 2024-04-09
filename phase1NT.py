import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the transaction data
data = pd.read_csv('trans_data1.csv')

# Display the first few rows of the dataset
print(data.head())

# Get summary statistics of the dataset
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Convert the DAY column to datetime format
data['DAY'] = pd.to_datetime(data['DAY'])

# Extract year, month, and DAY of week from the DAY column
data['Year'] = data['DAY'].dt.year
data['Month'] = data['DAY'].dt.month
data['DAYOfWeek'] = data['DAY'].dt.dayofweek

# Visualize transaction trends over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='DAY', y='BILL_AMT')
plt.title('Transaction Trends Over Time')
plt.xlabel('DAY')
plt.ylabel('BILL_AMT')
plt.show()

# Visualize transaction distribution by month
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Month', y='BILL_AMT')
plt.title('Transaction Distribution by Month')
plt.xlabel('Month')
plt.ylabel('BILL_AMT')
plt.show()

# Analyze purchase frequency by DAY of week
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='DAYOfWeek')
plt.title('Purchase Frequency by DAY of Week')
plt.xlabel('DAY of Week')
plt.ylabel('Frequency')
plt.show()

# Analyze customer spending behavior
customer_spending = data.groupby('BILL_ID')['BILL_AMT'].sum()
plt.figure(figsize=(10, 6))
sns.histplot(customer_spending, bins=30)
plt.title('Customer Spending Behavior')
plt.xlabel('Total Spending')
plt.ylabel('Number of Customers')
plt.show()

# Identify top spending customers
top_spending_customers = customer_spending.nlargest(10)
print("Top 10 Spending Customers:")
print(top_spending_customers)

# Analyze popular products
popular_products = data['BRD'].value_counts().nlargest(10)
print("\nTop 10 Popular Products:")
print(popular_products)
