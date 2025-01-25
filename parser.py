import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yfinance as yf

class StockTradingML:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def prepare_features(self, data):
        """Calculate technical indicators as features"""
        df = data.copy()

        # Check if there are enough rows to calculate features
        if len(df) < 20:  # Minimum for SMA20
            print("Not enough data to prepare features.")
            return pd.DataFrame()  # Return an empty DataFrame

        # Calculate moving averages
        df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA50'] = df['Close'].rolling(window=50, min_periods=1).mean()

        # Calculate price momentum
        df['Price_Change'] = df['Close'].pct_change().fillna(0)
        df['Price_Change_5'] = df['Close'].pct_change(periods=5).fillna(0)

        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)  # Fill NaN with neutral RSI value

        # Create features array
        features = ['SMA20', 'SMA50', 'Price_Change', 'Price_Change_5', 'RSI']
        X = df[features]

        return X

    def create_labels(self, data, threshold=0.02):
        """Create trading signals (0: Hold, 1: Buy, 2: Sell)"""
        returns = data['Close'].pct_change()
        labels = np.zeros(len(returns))

        # Buy signal if return > threshold
        labels[returns > threshold] = 1
        # Sell signal if return < -threshold
        labels[returns < -threshold] = 2

        return labels[1:]  # Shift labels by 1 to align with future returns

    def train(self, historical_data):
        """Train the model"""
        X = self.prepare_features(historical_data)
        y = self.create_labels(historical_data)

        # Check if there are enough samples
        if X.shape[0] == 0 or y.shape[0] == 0:
            print("Not enough data to train the model.")
            return 0.0  # Return 0.0 or some indication of failure

        X = X.iloc[1:]  # Align X with y by removing the first row of X
        valid_data = ~np.isnan(y)
        X = X[valid_data]
        y = y[valid_data]

        # Check again after filtering
        if X.shape[0] == 0 or y.shape[0] == 0:
            print("After filtering, not enough data to train the model.")
            return 0.0  # Return 0.0 or some indication of failure

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        return self.model.score(X_test_scaled, y_test)

    def predict(self, current_data):
        """Make trading decision for current data"""
        X = self.prepare_features(current_data)

        # Check if features are empty
        if X.empty:
            print("No valid features to predict.")
            return None  # Return None or some indication of failure

        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)
        return prediction[-1]  # Return the most recent prediction

# Example usage:
if __name__ == "__main__":
    # Read penny stocks from CSV
    penny_stocks_df = pd.read_csv("logs/penny_stocks.csv")

    # Initialize the trader
    trader = StockTradingML()

    # Add a header to the file if it doesn't already exist
    with open("logs/stocks.csv", "a") as file:
        file.write("Ticker,Price,Accuracy\n")
        file.flush()
        for index, row in penny_stocks_df.iterrows():
            ticker = row['Ticker']
            print(f"Analyzing {ticker}...")

            # Fetch historical data for the penny stock
            stock = yf.Ticker(ticker)
            historical_data = stock.history(period="1y")  # Get 1 year of data

            # Check if historical data is valid
            if historical_data.empty:
                print(f"No historical data for {ticker}. Skipping...\n")
                continue

            # Train the model and make predictions
            accuracy = trader.train(historical_data)
            print(f"Model accuracy for {ticker}: {accuracy:.2f}")

            decision = trader.predict(historical_data)
            if decision is None:
                print(f"Skipping prediction for {ticker} due to insufficient data.\n")
                continue  # Skip to the next iteration if no valid prediction

            latest_price = historical_data['Close'].iloc[-1]
            print(f"Latest stock price for {ticker}: ${latest_price:.2f}")
            print(f"Trading decision for {ticker}: {['Hold', 'Buy', 'Sell'][int(decision)]}\n")
            if int(decision) == 1 and accuracy == 1 and latest_price < 1:
                file.write(f"{ticker},{latest_price:.2},{accuracy}\n")
                file.flush()