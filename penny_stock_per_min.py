import yfinance as yf
import schedule
import time
import pandas as pd

# Dictionary to store the last price for each ticker
last_prices = {}

def fetch_stock_price(ticker):
    # Fetch the stock data for the given ticker
    stock = yf.Ticker(ticker)
    data = stock.history(period="8d", interval="1m")  # Use interval="1m" for minute-level data
    if not data.empty:
        current_price = data['Close'].iloc[-1]  # Get the last closing price

        # Check if the current price is equal to the last price
        if ticker in last_prices:
            is_equal = current_price == last_prices[ticker]
            if not is_equal:
                print(f"Ticker {ticker} has changed and new price is {current_price}")
        else:
            print(f"First time price for {ticker} is {current_price}.")
        
        # Update the last price
        last_prices[ticker] = current_price
    else:
        print(f"No data available for {ticker}")

def read_tickers_from_csv(file_path):
    # Read the ticker symbols from a CSV file
    df = pd.read_csv(file_path)
    return df['Ticker'].tolist()

# Read ticker symbols from the CSV file
tickers = read_tickers_from_csv('logs/stocks.csv')

# Schedule the fetch_stock_price function to run every minute for each ticker
for ticker in tickers:
    schedule.every(1).minutes.do(fetch_stock_price, ticker)

while True:
    schedule.run_pending()
    time.sleep(1)
