import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import Dropout  # type: ignore # Add this line to import the Dropout layer
from tensorflow.keras.optimizers import Adam  # type: ignore # Add this line to import the Adam optimizer


# Define the Visualization class
class Visualization:
    def __init__(self, df, y_test, y_pred):
        self.df = df
        self.y_test = y_test
        self.y_pred = y_pred

    def plot_data(self):
        # Plot the historical stock prices and moving average
        plt.figure(figsize=(16, 8))
        plt.plot(self.df.index, self.df['4. close'], label='Close Price')
        plt.plot(self.df.index, self.df['moving_avg'], label='Moving Average')
        plt.title('Historical Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_predictions(self):
        # Plot the true prices vs. predicted prices
        plt.figure(figsize=(16, 8))
        plt.plot(self.y_test, label='True Price')
        plt.plot(self.y_pred, label='Predicted Price')
        plt.title('True vs. Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

# Load the dataset from Alpha Vantage API and save it to a CSV file
def load_data():
    # Define the API key and the stock symbol
    api_key = '2EM2U64O7NGI0E62'
    symbol = 'AAPL'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&outputsize=full&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    # print(data)

    # Access the nested dictionary correctly
    time_series_data = data['Weekly Time Series']

    # Convert the data to a pandas DataFrame and save it to a .csv file
    df = pd.DataFrame(time_series_data).transpose()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.to_csv('data.csv')
    return df

# Preprocess the dataset so that it can be used for training the machine learning model
def preprocess_data(df):
    # Feature engineering: calculate the moving average
    df['moving_avg'] = df['4. close'].rolling(window=5).mean()

    # Extract the closing prices
    y = df['4. close'].values.astype(float)

    # Normalization of the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    y = scaler.fit_transform(y.reshape(-1, 1))

    df['4. close'] = scaler.fit_transform(df['4. close'].values.reshape(-1, 1))
    df['moving_avg'] = scaler.fit_transform(df['moving_avg'].values.reshape(-1, 1))

    # Initialize lists to store X and y data
    X, y_data = [], []

    # Create sequences of data with a window size of 60 for each sample
    for i in range(60, len(df)):
        X.append(df.iloc[i-60:i][['4. close', 'moving_avg']].values)
        y_data.append(y[i])

    # Convert lists to numpy arrays
    X = np.array(X)
    y_data = np.array(y_data)

    return X, y_data, scaler


# Create and train the machine learning model
# Create and train the machine learning model
def create_model(X_train, y_train):
    # Create the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))  # type: ignore # Adding dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))  # type: ignore # Adding dropout
    model.add(Dense(units=1))

    # Compile the model with the custom optimizer and learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=200)  # Increase epochs

    return model

# Evaluate the model on the test set
def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Handle missing values
    y_test = np.nan_to_num(y_test)
    y_pred = np.nan_to_num(y_pred)

    return y_pred


# Create main function to run the program
def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Handle missing values
    y_test = np.nan_to_num(y_test)
    y_pred = np.nan_to_num(y_pred)

    return y_pred

def main():
    # Load the data
    df = load_data()

    # Preprocess the data
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Check if the model file exists
    if os.path.exists('model.keras'):
        # Load the model
        model = load_model('model.keras')

        # Make predictions on the test set
        y_pred = evaluate_model(model, X_test, y_test)

        # Invert the scaling for visualization
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

        # Visualize the data and model predictions
        visualization = Visualization(df, y_test, y_pred)
        visualization.plot_data()
        visualization.plot_predictions()
    else:
        # Create and train the model
        model = create_model(X_train, y_train)

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print(f'Test loss: {loss}')

        # Save the model for later use
        model.save('model.keras')

        # Make predictions on the test set
        y_pred = evaluate_model(model, X_test, y_test)

        # Invert the scaling for visualization
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

        # Visualize the data and model predictions
        visualization = Visualization(df, y_test, y_pred)
        visualization.plot_data()
        visualization.plot_predictions()

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()


