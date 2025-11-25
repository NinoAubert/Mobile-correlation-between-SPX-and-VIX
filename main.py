import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def safe_download(ticker, start, end, max_retries=5):
    for attempt in range(max_retries):
        try:
            print(f"Downloading {ticker} (attempt {attempt+1})...")
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if data.empty:
                raise ValueError("Empty dataset")
            return data
        except Exception as e:
            print(f"Error: {e}. Retrying in 5 sec...")
            time.sleep(5)
    raise RuntimeError(f"Failed to download {ticker} after {max_retries} attempts.")

start = "2010-01-01"
end   = "2020-12-31"

# ---------------------------------------------------
# 1. Download with robust retry
# ---------------------------------------------------
spx = safe_download("^GSPC", start, end)
vix = safe_download("^VIX", start, end)

# Keep Close price
spx_close = spx["Close"]
vix_close = vix["Close"]

# ---------------------------------------------------
# 2. Compute returns
# ---------------------------------------------------
r_spx = np.log(spx_close).diff()
d_vix = vix_close.diff()
r_vix = np.log(vix_close).diff()

df = pd.concat([r_spx, d_vix, r_vix], axis=1)
df.columns = ["r_spx", "d_vix", "r_vix"]
df = df.dropna()

# ---------------------------------------------------
# 3. Correlations
# ---------------------------------------------------
rho_diff = df["r_spx"].corr(df["d_vix"])
rho_log  = df["r_spx"].corr(df["r_vix"])

print("Corrélation (SPX returns, ΔVIX) :", round(rho_diff, 4))
print("Corrélation (SPX returns, log VIX returns) :", round(rho_log, 4))

# ---------------------------------------------------
# 4. Rolling 180-day correlation
# ---------------------------------------------------
rolling_corr = df["r_spx"].rolling(180).corr(df["d_vix"])

plt.figure(figsize=(12,6))
plt.plot(rolling_corr)
plt.axhline(0, color="black", linewidth=1)
plt.plot(rolling_corr, color='black')
plt.title("Corrélation mobile SPX / ΔVIX (fenêtre 180 jours)")
plt.grid(True)
plt.show()
