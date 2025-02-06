#EDA File

#I did EDA on JN but im transferring those figures to VSCODE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import norm
import seaborn as sns

df = pd.read_csv('thesis DSS.csv')


# Histogram for Contract_duur
plt.figure(figsize=(12, 6))
plt.hist(df['Contract_duur'], bins=20, alpha=0.7, label='Contract_duur', color='lightblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Contract_duur')
plt.ylabel('Frequency')
plt.title('Histogram of Contract_duur')

# Adding legend
plt.legend()

# Show the plot
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Standardize Contract_duur
mean = df['Contract_duur'].mean()
std = df['Contract_duur'].std()
standardized_data = (df['Contract_duur'] - mean) / std

# Generate x values for the normal curve
x = np.linspace(standardized_data.min(), standardized_data.max(), 100)
y = norm.pdf(x, 0, 1)  # Standard normal distribution (mean=0, std=1)

# Plot the histogram and normal curve
plt.figure(figsize=(10, 6))
plt.hist(standardized_data, bins=20, density=True, alpha=0.6, color='lightblue', edgecolor='black', label='Standardized Data')
plt.plot(x, y, color='red', label='Standard Normal Curve')

# Adding labels and title
plt.xlabel('Standardized Contract_duur')
plt.ylabel('Density')
plt.title('Standardized Distribution of Contract_duur')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()


# Calculate basic statistics
mean = df['Contract_duur'].mean()
median = df['Contract_duur'].median()
std_dev = df['Contract_duur'].std()
min_val = df['Contract_duur'].min()
max_val = df['Contract_duur'].max()

# Calculate skewness and kurtosis
skewness = df['Contract_duur'].skew()
kurtosis = df['Contract_duur'].kurt()

# Display results
print(f"Statistics for 'Contract_duur':")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")
print(f"Minimum: {min_val}")
print(f"Maximum: {max_val}")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")



# Ensure 'Ingangsdatum_contract' is in datetime format and handle invalid parsing
df['Ingangsdatum_contract'] = pd.to_datetime(df['Ingangsdatum_contract'], errors='coerce')

# Drop rows with missing values for 'Contract_duur' and 'Ingangsdatum_contract'
df_cluster = df.dropna(subset=['Contract_duur', 'Ingangsdatum_contract'])

# Plot the scatter plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_cluster, x='Ingangsdatum_contract', y='Contract_duur', alpha=0.6)

# Adding labels and title
plt.xlabel('Contract Start Date (Ingangsdatum_contract)')
plt.ylabel('Contract Duration (Contract_duur)')
plt.title('Scatter Plot of Contract Duration vs. Contract Start Date')

# Show the plot
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
