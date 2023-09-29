import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Generate synthetic random data
np.random.seed(0)
n_samples = 100
date_range = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
synthetic_gold_prices = np.random.rand(n_samples) * 1000 + 1200
synthetic_interest_rates = np.random.rand(n_samples) * 5 + 1
synthetic_inflation_rates = np.random.rand(n_samples) * 3 + 1

# Create a DataFrame to hold the synthetic data
synthetic_data = pd.DataFrame({
    'Date': date_range,
    'GoldPrice': synthetic_gold_prices,
    'InterestRate': synthetic_interest_rates,
    'InflationRate': synthetic_inflation_rates
})

# Sort the data by date
synthetic_data = synthetic_data.sort_values('Date')

# Calculate lag features (e.g., previous day's gold price)
synthetic_data['PreviousGoldPrice'] = synthetic_data['GoldPrice'].shift(1)

# Drop missing values resulting from lag calculation
synthetic_data = synthetic_data.dropna()

# Split the data into training and testing sets
X = synthetic_data[['PreviousGoldPrice', 'InterestRate', 'InflationRate']]
y = synthetic_data['GoldPrice']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate evaluation metrics (Mean Squared Error and Mean Absolute Error)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the evaluation results
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Visualize predicted vs. actual gold prices
plt.figure(figsize=(12, 6))
plt.plot(synthetic_data['Date'], synthetic_data['GoldPrice'], label='Actual Gold Price', color='blue')
plt.plot(synthetic_data['Date'].iloc[-len(y_test):], y_pred, label='Predicted Gold Price', color='red')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.legend()
plt.title('Gold Price Prediction')
plt.show()
