import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('socialdata.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Summary statistics
print(df.describe())

# Visualization
# Example: Plotting retweet count over time
df['Date1'] = pd.to_datetime(df['Date1'])
df.set_index('Date1', inplace=True)
df['retweet_count'].plot()
plt.title('Retweet Count Over Time')
plt.xlabel('Date')
plt.ylabel('Retweet Count')
plt.show()
