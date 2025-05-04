# Stock Price Prediction using LSTM (Long Short-Term Memory)

## Project Overview
This project aims to predict the future stock prices of Microsoft (MSFT) using a deep learning technique called Long Short-Term Memory (LSTM). The goal is to build a predictive model that can forecast stock prices based on historical data, specifically the open, high, low, close prices, and volume of stock traded for each day.

---

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [License](#license)

---

## Installation

Ensure you have Python installed on your machine, and the necessary libraries as outlined below. You can install all the required libraries using `pip`:

### Required Libraries:
- **TensorFlow**: Deep learning framework for building the LSTM model.
- **Pandas**: For data manipulation and processing.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting graphs.
- **Seaborn**: For creating visualizations like heatmaps.
- **Scikit-learn**: For data preprocessing and evaluation metrics.

Dataset
The dataset used in this project is historical stock price data for Microsoft (MSFT), with daily records. 

Columns in the dataset:
Date: The date of the stock price.

Open: The opening price for that day.

High: The highest price reached for that day.

Low: The lowest price reached for that day.

Close: The closing price for that day (used as the target for prediction).

Volume: The number of shares traded on that day.

Name: The stock ticker symbol (MSFT).

Data Source:
You can download the dataset from financial sources like Yahoo Finance or use your own historical data for Microsoft.

Usage
Clone or download this repository to your local machine.

Place the MicrosoftStock.csv dataset in your project directory (or update the file path).

Run the script using Python:

bash
Copy
Edit
python stock_price_prediction.py
Workflow:
Data Preprocessing:

Convert the 'Date' column to a datetime object for easier manipulation.

Drop unnecessary columns like Index and Name.

Normalize numerical columns (Open, High, Low, Close, Volume) using StandardScaler.

Visualization:

Visualize stock price trends and trading volumes over time.

Generate a heatmap to analyze correlations between features.

Model Building:

An LSTM model is created using TensorFlow/Keras with two LSTM layers and a Dense layer.

Dropout regularization is used to avoid overfitting.

Model Training:

The model is trained on 95% of the dataset and tested on the remaining 5%.

It predicts the closing price of Microsoft based on historical data.

Prediction:

After training, the model predicts stock prices for the test set, which is then compared to the actual prices.

Model Evaluation:

Evaluate the model performance using Root Mean Squared Error (RMSE).

Visualize predicted stock prices alongside actual prices for comparison.

Model
The model uses Long Short-Term Memory (LSTM), a type of Recurrent Neural Network (RNN) that excels in capturing dependencies in sequential data. The architecture is as follows:

LSTM Layers: Two layers with 64 units, capturing long-term dependencies in stock prices.

Dense Layer: A fully connected layer with ReLU activation to produce the final predicted stock price.

Dropout Layer: Prevents overfitting by randomly setting input units to 0 during training.

The model is compiled with:

Optimizer: Adam optimizer for efficient training.

Loss Function: Mean Absolute Error (MAE) for regression problems.

Metrics: Root Mean Squared Error (RMSE) to evaluate prediction accuracy.

Results
After training the model, predictions are generated on the test set. The results are visualized in a plot showing both the predicted stock prices and the actual stock prices over time.
