import marimo

__generated_with = "0.12.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Options analysis""")
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    from scipy.stats import norm
    return math, norm, np, pd, plt


@app.cell
def _(pd):
    d1,d2,d3=pd.read_csv('round3-data/prices_round_3_day_0.csv',sep=";"),pd.read_csv('round3-data/prices_round_3_day_1.csv',sep=";"),pd.read_csv('round3-data/prices_round_3_day_2.csv',sep=";")

    d1['day']=0
    d2['day']=1
    d3['day']=2
    return d1, d2, d3


@app.cell
def _(d1, d2, d3, pd):
    combined=pd.concat([d1,d2,d3],ignore_index=True)
    combined.loc[combined['day'] == 1, 'timestamp'] += 1000000
    combined.loc[combined['day'] == 2, 'timestamp'] += 2000000
    combined
    return (combined,)


@app.cell
def _(math, np, pd):
    def get_popular_price(row, bid_ask: str) -> int:
        best_price = -1
        max_volume = -1

        for i in range(1, 4):
            volume = getattr(row, f"{bid_ask}_volume_{i}")
            if math.isnan(volume):
                break

            if volume > max_volume:
                best_price = getattr(row, f"{bid_ask}_price_{i}")
                max_volume = volume

        return best_price

    def get_product_prices(prices: pd.DataFrame, product: str) -> np.ndarray:
        prices = prices[prices['product'] == product]

        mid_prices = []
        for row in prices.itertuples():
            popular_buy_price = get_popular_price(row, "bid")
            popular_sell_price = get_popular_price(row, "ask")
            mid_prices.append((popular_buy_price + popular_sell_price) / 2)

        return np.array(mid_prices)
    return get_popular_price, get_product_prices


@app.cell
def _(combined, get_product_prices):
    rock=get_product_prices(combined,'VOLCANIC_ROCK')
    r1=get_product_prices(combined,'VOLCANIC_ROCK_VOUCHER_9500')
    r2=get_product_prices(combined,'VOLCANIC_ROCK_VOUCHER_9750')
    r3=get_product_prices(combined,'VOLCANIC_ROCK_VOUCHER_10000')
    r4=get_product_prices(combined,'VOLCANIC_ROCK_VOUCHER_10250')
    r5=get_product_prices(combined,'VOLCANIC_ROCK_VOUCHER_10500')
    return r1, r2, r3, r4, r5, rock


@app.cell
def _(plt, r1, r2, r3, r4, r5):
    plt.plot(r1,label='950')
    plt.plot(r2,label='9750')
    plt.plot(r3,label='10000')
    plt.plot(r4,label='10250')
    plt.plot(r5,label='10500')

    plt.legend()

    plt.gca()
    return


@app.cell
def _(norm, np, plt):
    # --- Black-Scholes Pricing ---
    def bs_price_call(S, K, T, sigma):
        if T <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * norm.cdf(d2)

    # --- Manual Binary Search for IV ---
    def implied_volatility(S, K, T, market_price, option_type='call', tol=1e-6, max_iter=100):
        low, high = 1e-6, 5.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            price = bs_price_call(S, K, T, mid)
            if abs(price - market_price) < tol:
                return mid
            if price > market_price:
                high = mid
            else:
                low = mid
        return np.nan  # If not converged

    # --- Parabola fit using numpy.polyfit ---
    def fit_parabola(m_vals, v_vals):
        # Fit v = a*m^2 + b*m + c
        coeffs = np.polyfit(m_vals, v_vals, 2)
        return coeffs  # [a, b, c]

    # --- Smile Processing ---
    def process_smile(Sts, Ks, TTEs, Vts):
        smiles = []
        base_ivs = []

        for St, K_list, TTE, V_list in zip(Sts, Ks, TTEs, Vts):
            m_vals = []
            v_vals = []

            for K, Vt in zip(K_list, V_list):
                m = np.log(K / St) / np.sqrt(TTE)
                iv = implied_volatility(St, K, TTE, Vt)
                if not np.isnan(iv):
                    m_vals.append(m)
                    v_vals.append(iv)

            if len(m_vals) >= 3:
                m_vals = np.array(m_vals)
                v_vals = np.array(v_vals)

                coeffs = fit_parabola(m_vals, v_vals)
                a, b, c = coeffs
                fitted = a * m_vals**2 + b * m_vals + c
                base_iv = c  # since m=0 -> v = c

                # Store and optionally plot
                smiles.append((m_vals, v_vals, fitted))
                base_ivs.append(base_iv)

                plt.plot(m_vals, v_vals, 'o', label='Observed')
                plt.plot(np.sort(m_vals), a*np.sort(m_vals)**2 + b*np.sort(m_vals) + c, '-', label='Fitted')
                plt.axvline(0, color='gray', linestyle='--')
                plt.title("Volatility Smile")
                plt.xlabel("m_t")
                plt.ylabel("Implied Volatility")
                plt.legend()
                plt.show()

        return smiles, base_ivs
    return bs_price_call, fit_parabola, implied_volatility, process_smile


@app.cell
def _(np):
    def rough_iv(S, K, T, C):
        # Rough IV for calls using a shortcut formula
        with np.errstate(divide='ignore', invalid='ignore'):
            approx_iv = C / (0.4 * S * np.sqrt(T))
            approx_iv = np.clip(approx_iv, 0, 5)  # clip outliers
        return approx_iv

    def process_smile_q(Sts, Ks, TTEs, Vts):
        smiles = []
        base_ivs = []

        for S, K_list, T, V_list in zip(Sts, Ks, TTEs, Vts):
            S = float(S)
            K_array = np.array(K_list)
            V_array = np.array(V_list)

            if np.any(np.isnan(V_array)) or S <= 0 or T <= 0:
                smiles.append(None)
                base_ivs.append(np.nan)
                continue

            m = np.log(K_array / S) / np.sqrt(T)
            iv = rough_iv(S, K_array, T, V_array)

            valid = ~np.isnan(m) & ~np.isnan(iv)
            if np.sum(valid) < 3:
                smiles.append(None)
                base_ivs.append(np.nan)
                continue

            # Fit quadratic v(m) = a m^2 + b m + c
            coeffs = np.polyfit(m[valid], iv[valid], 2)
            smiles.append(coeffs.tolist())

            # Base IV = v(m=0) = c
            base_ivs.append(coeffs[2])

        return smiles, base_ivs



    return process_smile_q, rough_iv


@app.cell
def _(process_smile_q, r1, r2, r3, r4, r5, rock):
    # Underlying price time series
    Sts = rock

    # All strike levels
    Ks = [[9500, 9750, 10000, 10250, 10500]] * len(rock)

    # Time to expiry for each t (assuming constant for now, e.g., 10 trading days left)
    TTEs = [246 / 365] * len(rock)  # Change if needed

    # All voucher prices
    Vts = list(zip(r1, r2, r3, r4, r5))  # Each row = [r1_t, r2_t, ..., r5_t]

    # Run the smile fitting
    smiles, base_ivs = process_smile_q(Sts, Ks, TTEs, Vts)

    # Now base_ivs is your time series of fitted base implied volatilities
    return Ks, Sts, TTEs, Vts, base_ivs, smiles


@app.cell
def _(smiles):
    smiles
    return


@app.cell
def _(base_ivs):
    base_ivs
    return


if __name__ == "__main__":
    app.run()
