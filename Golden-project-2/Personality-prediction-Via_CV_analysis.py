import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate a synthetic dataset for personality prediction
num_samples = 1000

# Generate random image features (3 features for each sample)
np.random.seed(0)
image_features = np.random.rand(num_samples, 3)

# Generate random personality trait scores for five traits
personality_traits = {
    'Trait1': np.random.uniform(1, 7, num_samples),
    'Trait2': np.random.uniform(1, 7, num_samples),
    'Trait3': np.random.uniform(1, 7, num_samples),
    'Trait4': np.random.uniform(1, 7, num_samples),
    'Trait5': np.random.uniform(1, 7, num_samples)
}

# Create a DataFrame to store the synthetic dataset
synthetic_data = pd.DataFrame({**{'Feature_' + str(i+1): image_features[:, i] for i in range(3)}, **personality_traits})

# Split the dataset into features (X) and personality trait labels (y)
X = synthetic_data.drop(columns=['Trait1', 'Trait2', 'Trait3', 'Trait4', 'Trait5'])
y = synthetic_data[['Trait1', 'Trait2', 'Trait3', 'Trait4', 'Trait5']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest regressor for personality prediction
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Visualize the predicted vs. actual personality trait values
plt.figure(figsize=(12, 10))
traits = ['Trait1', 'Trait2', 'Trait3', 'Trait4', 'Trait5']
for i in range(5):
    plt.subplot(3, 2, i + 1)
    plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.7, label='Samples')
    plt.xlabel(f'Actual {traits[i]}')
    plt.ylabel(f'Predicted {traits[i]}')
    plt.title(f'Actual vs. Predicted {traits[i]}')
    plt.legend(loc='best')
    plt.plot([min(y_test.iloc[:, i]), max(y_test.iloc[:, i])], [min(y_test.iloc[:, i]), max(y_test.iloc[:, i])],
             color='red', linestyle='--', lw=2, label='Perfect Prediction')
    plt.legend(loc='best')

plt.tight_layout()
plt.show()
