import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_file = "goat_compare_5m_720d_be20_c1c2c3_p15.csv"


# Load the CSV
df = pd.read_csv(csv_file)

# List of coins
coin_columns = [col for col in df.columns if "equity" in col or "ONDO" in col or "LTC" in col]
# or specify: coin_columns = ["ONDO_USDT_equity", "LTC_USDT_equity"]

plt.figure(figsize=(12,6))
for col in coin_columns:
    plt.plot(df['Date'], df[col], label=col)

plt.title("Multi-Coin Equity Curves")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()