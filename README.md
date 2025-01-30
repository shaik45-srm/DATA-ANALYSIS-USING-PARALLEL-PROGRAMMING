# ARIMA Time Series Forecasting with Parallel Execution

This project demonstrates time series forecasting using the ARIMA (AutoRegressive Integrated Moving Average) model on weather data. We use parallel computation to speed up the ARIMA model fitting and prediction process. The main goal is to show how parallel execution can improve performance when performing multiple runs of the model and generating predictions.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Source](#data-source)
- [How to Run](#how-to-run)
- [Explanation of the Code](#explanation-of-the-code)
- [Results](#results)
- [Execution Time](#execution-time)
- [License](#license)

## Introduction

This project uses a dataset of daily minimum temperatures and applies the ARIMA model to predict future temperatures. By using parallel execution via the `joblib` library, the time taken to fit and forecast multiple ARIMA models is significantly reduced. The primary steps include:

1. Data exploration and visualization.
2. Splitting the data into training and test sets.
3. Fitting an ARIMA model for forecasting.
4. Running the model in parallel for multiple predictions.
5. Visualizing the predictions compared to the actual values.

## Requirements

To run this project, you need to install the following libraries:

- `pandas`
- `matplotlib`
- `statsmodels`
- `joblib`

You can install them using pip:

pip install pandas matplotlib statsmodels joblib


Data Source

The dataset used is a publicly available weather dataset of daily minimum temperatures from Melbourne, Australia. The data can be accessed at:

[Daily Minimum Temperatures Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv)

How to Run

1. Clone this repository or download the script.
2. Ensure all dependencies are installed.
3. Run the script:

bash
python arima_parallel_forecasting.py


The script will:
- Load and preprocess the dataset.
- Split the data into training and test sets.
- Fit the ARIMA model with parallel execution.
- Display the time series plot and forecast results.

Explanation of the Code

Data Exploration and Plotting

First, the dataset is loaded using `pandas`, and basic descriptive statistics are printed. A line plot is generated to visualize the daily minimum temperatures.

python
df = pd.read_csv(data_url, header=0, index_col=0, parse_dates=True)
plt.plot(df)


 ARIMA Model Fitting and Prediction

The ARIMA model is defined with an order of (5, 1, 0). The script includes a function `fit_arima` that fits the model to the training data and predicts the future values.

python
model = ARIMA(train_data, order=order)
model_fit = model.fit()
predictions = model_fit.forecast(steps=len(test_data))


Parallel Execution

The `joblib` library is used to parallelize the model fitting and prediction process. Multiple runs of the ARIMA model are executed in parallel, and the results are averaged to generate a final prediction.

python
predictions = Parallel(n_jobs=-1)(delayed(fit_arima)(train_data, test_data, order) for _ in range(10))


Visualization

Finally, the actual values and predicted values are plotted together to visualize the accuracy of the model's forecasts.

python
plt.plot(test_data.index, test_data.values, label='Actual')
plt.plot(test_data.index, avg_predictions, color='red', label='Predicted (Average)')


Results

The model is evaluated by comparing the predicted temperature values to the actual test data. The results are displayed in a plot showing both the actual and predicted temperatures.

Execution Time

Parallel execution speeds up the fitting and forecasting process, especially when the model is run multiple times. The script measures the execution time, which is displayed after the parallel execution.


Parallel Execution Time: X seconds


 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For queries or collaboration, contact [MURALI SAI V ]&[SAMEER SK] at [mv8039@srmist.edu.in] &[ss7268@srmist.edu.in].
