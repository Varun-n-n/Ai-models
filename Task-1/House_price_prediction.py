# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set a random seed for reproducibility
np.random.seed(42)

# Generate a smaller synthetic dataset for house price prediction
n_samples = 500  # Reduce the dataset size

# Generate random features for the houses
house_size = np.random.randint(800, 5000, n_samples)  # Square footage
num_bedrooms = np.random.randint(1, 6, n_samples)
num_bathrooms = np.random.randint(1, 4, n_samples)
latitude = np.random.uniform(-90, 90, n_samples)
longitude = np.random.uniform(-180, 180, n_samples)

# Generate random house prices with more fluctuation
house_prices = 1000 * house_size + 20000 * num_bedrooms + 15000 * num_bathrooms + \
               5000 * np.abs(latitude) + 3000 * np.abs(longitude) + np.random.normal(0, 100000, n_samples)

# Create a DataFrame from the generated data
data = pd.DataFrame({
    'HouseSize': house_size,
    'Bedrooms': num_bedrooms,
    'Bathrooms': num_bathrooms,
    'Latitude': latitude,
    'Longitude': longitude,
    'Price': house_prices
})

# Split the data into training and testing sets (80% train, 20% test)
X = data.drop('Price', axis=1)  # Features (input variables)
y = data['Price']  # Target variable (house prices)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Regressor model with hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Perform Randomized Search CV for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)

# Get the best model from the hyperparameter tuning
best_rf = random_search.best_estimator_

# Make predictions on the test data using the best model
y_pred = best_rf.predict(X_test)

# Evaluate the model using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R2)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Evaluation Metrics for Random Forest Regressor (Hyperparameter Tuning):")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Visualize actual vs. predicted house prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Actual vs. Predicted')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices (Random Forest Regressor)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.legend(loc='upper left')
plt.show()
