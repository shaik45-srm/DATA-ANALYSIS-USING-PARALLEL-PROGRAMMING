import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed
import time

# Load sample weather data
data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
df = pd.read_csv(data_url, header=0, index_col=0, parse_dates=True)

# Explore the data
print(df.head())
print(df.describe())

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df)
plt.title('Daily Minimum Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# Define function for ARIMA model fitting and prediction
def fit_arima(train_data, test_data, order):
    # Define and fit the ARIMA model
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.forecast(steps=len(test_data))
    return predictions

# Define order for ARIMA model
order = (5, 1, 0)  # (p, d, q)

# Measure execution time
start_time = time.time()

# Parallel execution for ARIMA model fitting and prediction
predictions = Parallel(n_jobs=-1)(delayed(fit_arima)(train_data, test_data, order) for _ in range(10))

# Average predictions from multiple runs
avg_predictions = sum(predictions) / len(predictions)

# Measure end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("Parallel Execution Time:", execution_time, "seconds")

# Plot predictions against actual values
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data.values, label='Actual')
plt.plot(test_data.index, avg_predictions, color='red', label='Predicted (Average)')
plt.title('ARIMA Forecasting')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()