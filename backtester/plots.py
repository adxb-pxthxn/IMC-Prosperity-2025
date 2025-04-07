import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("backtester/resources/round0/prices_round_0_day_-4.csv", sep=';')

# Create a timestamp column from 'day' and 'timestamp' (adjust if real time info is present)
df['time_index'] = df['day'].astype(str) + '_' + df['timestamp'].astype(str)

# Convert to a numeric index if needed
df['time_index'] = range(len(df))

# Get unique products
products = df['product'].unique()

# Plot for each product
for product in products:
    product_df = df[df['product'] == product]

    plt.figure(figsize=(10, 5))
    plt.plot(product_df['time_index'], product_df['bid_price_1'], label='Bid Price 1', marker='o')
    plt.plot(product_df['time_index'], product_df['ask_price_1'], label='Ask Price 1', marker='x')
    plt.plot(product_df['time_index'], product_df['mid_price'], label='Mid Price', linestyle='--', marker='.')
    

    # plt.ylim(-10,10)
    plt.title(f"{product} - Bid/Ask/Mid Prices Over Time")
    plt.xlabel("Time Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
