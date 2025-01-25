import os
import requests
import time
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Replace with your API key
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OUTPUT_FILE = "logs/penny_stocks.csv"

def get_all_tickers():
    """
    Fetches all stock tickers from Finnhub API.
    Returns:
        list: A list of stock tickers.
    """
    url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={FINNHUB_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        tickers = [stock["symbol"] for stock in response.json()]
        print(f"Fetched {len(tickers)} tickers.")
        return tickers
    else:
        print(f"Error fetching tickers: {response.status_code}")
        return []

def fetch_penny_stocks(tickers: list = [], price_h_limit: float = 1.0, price_l_limit: float = 0.005):
    """
    Identifies penny stocks from a list of stock tickers by querying their current price.
    Logs each found penny stock to a file.

    Parameters:
        tickers (list): List of stock tickers to analyze.
        price_limit (float): Maximum price to qualify as a penny stock.

    Returns:
        list: A list of penny stock symbols and their prices.
    """
    penny_stocks = []

    # Add a header to the file if it doesn't already exist
    with open(OUTPUT_FILE, "a") as file:
        if file.tell() == 0:  # Check if the file is empty
            file.write("Ticker,Price (USD),Timestamp\n")

    with open(OUTPUT_FILE, "a") as file:
        # Use tqdm for the progress bar
        for ticker in tqdm(tickers, desc="Scanning tickers", unit="ticker"):
            try:
                url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    price = data.get("c", None)  # 'c' is the current price

                    if price is not None and price <= price_h_limit and price >= price_l_limit:
                        penny_stocks.append((ticker, price))
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # Write to file immediately
                        print(f"{ticker},{price},{timestamp}\n")
                        file.write(f"{ticker},{price:},{timestamp}\n")
                        file.flush()  # Ensure data is written to disk
                else:
                    print(f"Error fetching price for {ticker}: {response.status_code}")

                time.sleep(1)  # Throttle requests to avoid rate limits
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

    return penny_stocks

if __name__ == "__main__":
    print("Fetching all tickers...")
    all_tickers = get_all_tickers()

    if all_tickers:
        print("Scanning for penny stocks...")
        penny_stocks = fetch_penny_stocks(tickers=all_tickers, price_h_limit=0.75, price_l_limit=0.005)

        if penny_stocks:
            print("\nPenny stocks found:")
            for ticker, price in penny_stocks:
                print(f"{ticker}: ${price:.2f}")
        else:
            print("\nNo penny stocks found.")
    else:
        print("No tickers to scan.")
