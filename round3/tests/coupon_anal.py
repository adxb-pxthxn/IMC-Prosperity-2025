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
    from statistics import NormalDist
    from scipy.stats import norm
    from scipy.optimize import curve_fit
    return NormalDist, curve_fit, math, norm, np, pd, plt


@app.cell
def _(pd):
    d1,d2,d3=pd.read_csv('round3-data/prices_round_3_day_0.csv',sep=";"),pd.read_csv('round3-data/prices_round_3_day_1.csv',sep=";"),pd.read_csv('round3-data/prices_round_3_day_2.csv',sep=";")
    d4=pd.read_csv('round3-data/prices_round_4_day_3.csv',sep=';')
    d5=pd.read_csv('round3-data/prices_round_5_day_4.csv',sep=";")

    d1['day']=0
    d2['day']=1
    d3['day']=2
    d4['day']=3
    d5['day']=4
    return d1, d2, d3, d4, d5


@app.cell
def _(d3, d4, d5, pd):
    combined=pd.concat([d3,d4,d5],ignore_index=True)
    combined.loc[combined['day'] == 3, 'timestamp'] += 1000000
    combined.loc[combined['day'] == 4, 'timestamp'] += 2000000
    # combined.loc[combined['day'] == 3, 'timestamp'] += 3000000
    # combined.loc[combined['day'] == 4, 'timestamp'] += 4000000
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
def _(combined):
    print(combined.head())
    return


@app.cell
def _():
    # # --- Black-Scholes Pricing ---
    # def bs_price_call(S, K, T, sigma):
    #     if T <= 0:
    #         return max(S - K, 0)
    #     d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    #     d2 = d1 - sigma * np.sqrt(T)
    #     return S * norm.cdf(d1) - K * norm.cdf(d2)

    # # --- Manual Binary Search for IV ---
    # def implied_volatility(S, K, T, market_price, option_type='call', tol=1e-6, max_iter=100):
    #     low, high = 1e-6, 5.0
    #     for _ in range(max_iter):
    #         mid = (low + high) / 2
    #         price = bs_price_call(S, K, T, mid)
    #         if abs(price - market_price) < tol:
    #             return mid
    #         if price > market_price:
    #             high = mid
    #         else:
    #             low = mid
    #     return np.nan  # If not converged

    # # --- Parabola fit using numpy.polyfit ---
    # def fit_parabola(m_vals, v_vals):
    #     # Fit v = a*m^2 + b*m + c
    #     coeffs = np.polyfit(m_vals, v_vals, 2)
    #     return coeffs  # [a, b, c]

    # # --- Smile Processing ---
    # def process_smile(Sts, Ks, TTEs, Vts):
    #     smiles = []
    #     base_ivs = []

    #     for St, K_list, TTE, V_list in zip(Sts, Ks, TTEs, Vts):
    #         m_vals = []
    #         v_vals = []

    #         for K, Vt in zip(K_list, V_list):
    #             m = np.log(K / St) / np.sqrt(TTE)
    #             iv = implied_volatility(St, K, TTE, Vt)
    #             if not np.isnan(iv):
    #                 m_vals.append(m)
    #                 v_vals.append(iv)

    #         if len(m_vals) >= 3:
    #             m_vals = np.array(m_vals)
    #             v_vals = np.array(v_vals)

    #             coeffs = fit_parabola(m_vals, v_vals)
    #             a, b, c = coeffs
    #             fitted = a * m_vals**2 + b * m_vals + c
    #             base_iv = c  # since m=0 -> v = c

    #             # Store and optionally plot
    #             smiles.append((m_vals, v_vals, fitted))
    #             base_ivs.append(base_iv)

    #             plt.plot(m_vals, v_vals, 'o', label='Observed')
    #             plt.plot(np.sort(m_vals), a*np.sort(m_vals)**2 + b*np.sort(m_vals) + c, '-', label='Fitted')
    #             plt.axvline(0, color='gray', linestyle='--')
    #             plt.title("Volatility Smile")
    #             plt.xlabel("m_t")
    #             plt.ylabel("Implied Volatility")
    #             plt.legend()
    #             plt.show()

    #     return smiles, base_ivs
    return


@app.cell
def _():
    # def rough_iv(S, K, T, C):
    #     # Make all inputs arrays for safe elementwise ops
    #     S = np.asarray(S)
    #     K = np.asarray(K)
    #     T = np.asarray(T)
    #     C = np.asarray(C)

    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         approx_iv = C / (0.4 * S * np.sqrt(T))
    #         approx_iv = np.clip(approx_iv, 0, 5)  # reasonable bounds

    #     return approx_iv  # shape same as C

    # def process_smile_q(Sts, Ks, TTEs, Vts):
    #     smiles = []
    #     base_ivs = []

    #     for S, K_list, T, V_list in zip(Sts, Ks, TTEs, Vts):
    #         S = float(S)
    #         K_array = np.array(K_list)
    #         V_array = np.array(V_list)

    #         if np.any(np.isnan(V_array)) or S <= 0 or T <= 0:
    #             smiles.append(None)
    #             base_ivs.append(np.nan)
    #             continue

    #         m = np.log(K_array / S) / np.sqrt(T)
    #         iv = rough_iv(S, K_array, T, V_array)

    #         if np.isscalar(iv):  # safety check in case scalar still happens
    #             iv = np.full_like(m, np.nan)

    #         valid = ~np.isnan(m) & ~np.isnan(iv)
    #         if np.sum(valid) < 3:
    #             smiles.append(None)
    #             base_ivs.append(np.nan)
    #             continue

    #         coeffs = np.polyfit(m[valid], iv[valid], 2)
    #         smiles.append(coeffs.tolist())
    #         base_ivs.append(coeffs[2])  # v(0) = base IV

    #     return smiles, base_ivs


    # def process_smile_qf(Sts, Ks, TTEs, Vts):
    #     smiles = []
    #     base_ivs = []

    #     def quad(m, a, b, c):
    #         """Quadratic function for curve fitting."""
    #         return a * m**2 + b * m + c

    #     for S, K_list, T, V_list in zip(Sts, Ks, TTEs, Vts):
    #         S = float(S)
    #         K_array = np.array(K_list)
    #         V_array = np.array(V_list)

    #         if np.any(np.isnan(V_array)) or S <= 0 or T <= 0:
    #             smiles.append(None)
    #             base_ivs.append(np.nan)
    #             continue

    #         m = np.log(K_array / S) / np.sqrt(T)
    #         iv = C / (0.4 * S * np.sqrt(T))  # Rough estimate of implied volatility

    #         if np.isscalar(iv):  # Safety check in case scalar still happens
    #             iv = np.full_like(m, np.nan)

    #         valid = ~np.isnan(m) & ~np.isnan(iv)
    #         if np.sum(valid) < 3:
    #             smiles.append(None)
    #             base_ivs.append(np.nan)
    #             continue

    #         # Normalize m before fitting
    #         m_mean = np.mean(m[valid])
    #         m_std = np.std(m[valid]) + 1e-6  # Avoid divide by 0
    #         m_scaled = (m[valid] - m_mean) / m_std

    #         # Fit a quadratic to the valid points
    #         try:
    #             popt, _ = curve_fit(quad, m_scaled, iv[valid])
    #             coeffs = popt
    #             base_ivs.append(coeffs[2])  # Base IV is the constant term (c) in the fit
    #         except:
    #             coeffs = [np.nan, np.nan, np.nan]
    #             base_ivs.append(np.nan)

    #         smiles.append(coeffs.tolist())

    #         # Plot the result for diagnostics
    #         x_plot = np.linspace(m_scaled.min(), m_scaled.max(), 100)
    #         y_fit = quad(x_plot, *coeffs)

    #         plt.scatter(m[valid], iv[valid], label="Data", alpha=0.5)
    #         plt.plot(x_plot * m_std + m_mean, y_fit, label="Quadratic Fit", color='red')
    #         plt.title(f"Smile fit | S={S:.2f}, T={T:.2f}")
    #         plt.xlabel("Moneyness")
    #         plt.ylabel("Rough IV")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()

    #     return smiles, base_ivs
    return


@app.cell
def _():
    # def process_smile_qfi(Sts, Ks, TTEs, Vts):
    #     smiles = []
    #     base_ivs = []

    #     def quad(m, a, b, c):
    #         """Quadratic function for curve fitting."""
    #         return a * m**2 + b * m + c

    #     for S, K_list, T, V_list in zip(Sts, Ks, TTEs, Vts):
    #         S = float(S)
    #         K_array = np.array(K_list)
    #         V_array = np.array(V_list)

    #         if np.any(np.isnan(V_array)) or S <= 0 or T <= 0:
    #             smiles.append(None)
    #             base_ivs.append(np.nan)
    #             continue

    #         m = np.log(K_array / S) / np.sqrt(T)
    #         # Now, V_array represents the option price (C)
    #         iv = V_array / (0.4 * S * np.sqrt(T))  # Rough estimate of implied volatility

    #         if np.isscalar(iv):  # Safety check in case scalar still happens
    #             iv = np.full_like(m, np.nan)

    #         valid = ~np.isnan(m) & ~np.isnan(iv)
    #         if np.sum(valid) < 3:
    #             smiles.append(None)
    #             base_ivs.append(np.nan)
    #             continue

    #         # Normalize m before fitting
    #         m_mean = np.mean(m[valid])
    #         m_std = np.std(m[valid]) + 1e-6  # Avoid divide by 0
    #         m_scaled = (m[valid] - m_mean) / m_std

    #         # Fit a quadratic to the valid points
    #         try:
    #             popt, _ = curve_fit(quad, m_scaled, iv[valid])
    #             coeffs = popt
    #             base_ivs.append(coeffs[2])  # Base IV is the constant term (c) in the fit
    #         except:
    #             coeffs = [np.nan, np.nan, np.nan]
    #             base_ivs.append(np.nan)

    #         smiles.append(coeffs)

    #         # Plot the result for diagnostics
    #         x_plot = np.linspace(m_scaled.min(), m_scaled.max(), 100)
    #         y_fit = quad(x_plot, *coeffs)

    #         plt.scatter(m[valid], iv[valid], label="Data", alpha=0.5)
    #         plt.plot(x_plot * m_std + m_mean, y_fit, label="Quadratic Fit", color='red')
    #         plt.title(f"Smile fit | S={S:.2f}, T={T:.2f}")
    #         plt.xlabel("Moneyness")
    #         plt.ylabel("Rough IV")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()

    #     return smiles, base_ivs
    return


@app.cell
def _():
    # # Underlying price time series
    # Sts = rock

    # # All strike levels
    # Ks = [[9500, 9750, 10000, 10250, 10500]] * len(rock)

    # # Time to expiry for each t (assuming constant for now, e.g., 10 trading days left)
    # TTEs = [4 / 252] * len(rock)  # Change if needed

    # # All voucher prices
    # Vts = zip(r1,r2,r3,r4,r5)  # Each row = [r1_t, r2_t, ..., r5_t]

    # # Run the smile fitting
    # smiles, base_ivs = process_smile_q(Sts, Ks, TTEs, Vts)

    # # Now base_ivs is your time series of fitted base implied volatilities
    return


@app.cell
def _():
    # smiles[-10]
    return


@app.cell
def _():
    # np.array(smiles)[-1]
    return


@app.cell
def _():
    # plt.plot(base_ivs)
    return


@app.cell
def _():
    # np.array(base_ivs).mean()
    return


@app.cell
def _():
    def black_scholes_call_price(S, K, T, r, sigma):
        from scipy.stats import norm
        from numpy import log, sqrt, exp
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

    # def implied_volatility(S, K, T, C, r=0.0, tol=1e-5, max_iter=1000):
    #     sigma = 0.2
    #     for _ in range(max_iter):
    #         price = black_scholes_call_price(S, K, T, r, sigma)
    #         vega = S * np.sqrt(T) * np.exp(-0.5 * ((np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T)))**2) / np.sqrt(2 * np.pi)
    #         if vega == 0:
    #             break
    #         diff = price - C
    #         if abs(diff) < tol:
    #             return sigma
    #         sigma -= diff / vega
    #     return sigma

    # def smile_curve(m, a, b, c):
    #     return a * m**2 + b * m + c  # parabolic fit

    # def fit_smile(S_t: float, K_list, V_t_list, TTE: float):
    #     m_t = [np.log(K / S_t) / np.sqrt(TTE) for K in K_list]
    #     v_t = [implied_volatility(S_t, K, TTE, V) for K, V in zip(K_list, V_t_list)]

    #     # Fit parabola
    #     popt, _ = curve_fit(smile_curve, m_t, v_t)
    #     m_vals = np.linspace(min(m_t), max(m_t), 100)
    #     v_fit = smile_curve(m_vals, *popt)
    #     base_iv = smile_curve(0, *popt)

    #     # Plot
    #     plt.figure(figsize=(8, 4))
    #     plt.scatter(m_t, v_t, label='Observed IVs')
    #     plt.plot(m_vals, v_fit, 'r--', label='Fitted smile')
    #     plt.axvline(0, color='gray', linestyle=':', linewidth=1)
    #     plt.title('Implied Volatility Smile')
    #     plt.xlabel('m_t (log-moneyness / sqrt(TTE))')
    #     plt.ylabel('v_t (Implied Volatility)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    #     return np.array(m_t), np.array(v_t), base_iv
    return (black_scholes_call_price,)


@app.cell
def _():
    # S_t = 10000
    # K_list = [9500, 9750, 10000, 10250, 10500]
    # V_t_list = [540, 360, 250, 160, 90]
    # TTE = 3 / 365  # 4 days to expiry

    # m_t, v_t, base_iv = fit_smile(S_t, K_list, V_t_list, TTE)
    # print(f"Base IV (m_t=0): {base_iv:.4f}")
    return


@app.cell
def _():
    # m_t,v_t,base_iv
    return


@app.cell
def _(NormalDist, combined, math, np):
    df=combined
    # assume your DataFrame is called `df`
    # and has columns: day, timestamp, product, mid_price, …

    # 1) Keep only the voucher rows
    mask = df['product'].str.contains('VOLCANIC_ROCK_VOUCHER')
    df_v = df[mask].copy()

    # 2) Extract the strike K (e.g. “VOLCANIC_ROCK_VOUCHER_10500” → 10500)
    df_v['K'] = df_v['product'].str.split('_').str[-1].astype(int)

    # 3) Bring in the underlying price S_t by merging with the base‐asset rows
    #    (you must have, in the same df, a row where product == 'VOLCANIC_ROCK')
    df_base = (
        df[df['product'] == 'VOLCANIC_ROCK']
          [['timestamp','mid_price']]
          .rename(columns={'mid_price':'S'})
    )
    df_v = df_v.merge(df_base, on='timestamp', how='left')

    # 4) Compute time‐to‐expiry (in years). 
    #    If “day” counts up 0,1,2,3 and expiry is after day=3, then:
    expiry_day = 7
    df_v['TTE'] = (8 - df_v['day']) / 365

    # 5) Compute normalized log‐moneyness
    df_v['m_t'] = np.log(df_v['K'] / df_v['S']) / np.sqrt(df_v['TTE'])

    # 6) Define Black‑Scholes call price and an implied‑vol solver
    def bs_call_price(S, K, T, r, sigma):
        N = NormalDist().cdf
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        return S*N(d1) - K*math.exp(-r*T)*N(d2)

    def implied_vol(market_price, S, K, T, r=0., tol=1e-6, max_iter=10000):
        low, high = 1e-6, 5.0
        for _ in range(max_iter):
            mid = 0.5*(low + high)
            price = bs_call_price(S, K, T, r, mid)
            # if our price is too high, vol is too big
            if price > market_price:
                high = mid
            else:
                low = mid
            if high - low < tol:
                break
        return 0.5*(low + high)

    # 7) Back out implied vols from mid_price → v_t
    df_v['v_t'] = df_v.apply(
        lambda r: implied_vol(r['mid_price'], r['S'], r['K'], r['TTE']),
        axis=1
    )

    # now df_v has columns day, timestamp, product, mid_price, K, S, TTE, m_t, v_t
    print(df_v.head())
    return bs_call_price, df, df_base, df_v, expiry_day, implied_vol, mask


@app.cell
def _(df_v):
    df_v['K'].unique()
    return


@app.cell
def _(df_v, np, plt):
    # Fit the parabola to the full data (or you can fit per-K too if needed)
    df_temp=df_v[df_v['v_t']>0.12]
    coeffs = np.polyfit(df_temp['m_t'], df_temp['v_t'], 2)
    fitted_values = np.polyval(coeffs, df_v['m_t'])

    # Define strike-to-color mapping
    mapping = {
        9500: 'red',
        9750: 'orange',
        10000: 'gold',
        10250: 'green',
        10500: 'cyan',
    }

    # Start plotting
    plt.figure(figsize=(10, 6))

    # Scatter each strike's data with its color
    for K, group in df_v.groupby('K'):
        color = mapping.get(K, 'gray')  # fallback color
        plt.plot(group['m_t'], group['v_t'], '.', label=f"K={K}", color=color)

    # Plot the fitted parabola (over full m_t range)
    m_t_range = np.linspace(df_v['m_t'].min(), df_v['m_t'].max(), 200)
    fitted_curve = np.polyval(coeffs, m_t_range)
    plt.plot(m_t_range, fitted_curve, label='Fitted Parabola', color="white", linewidth=2)
    plt.plot(m_t_range, fitted_curve+0.02, label='Fitted Parabola High', color="blue", linewidth=1)
    plt.plot(m_t_range, fitted_curve-0.02, label='Fitted Parabola Low', color="blue", linewidth=1)

    # Final plot setup
    plt.xlabel('m_t (log-moneyness)')
    plt.ylabel('v_t (Implied Vol)')
    plt.title('Parabolic Fit to the Smile')
    plt.legend()
    plt.grid(True)
    plt.show()
    return (
        K,
        coeffs,
        color,
        df_temp,
        fitted_curve,
        fitted_values,
        group,
        m_t_range,
        mapping,
    )


@app.cell
def _(coeffs):
    coeffs
    return


@app.cell
def _(df, plt):
    plt.plot(df[df['product']=="VOLCANIC_ROCK"]['mid_price'])
    return


@app.cell
def _(black_scholes_call_price, coeffs, df_v, np, plt):
    r = 0
    S = df_v['S'].iloc[0] if 'S' in df_v else 10000  

    # Predicted price using fitted parabola
    df_v['fitted_sigma'] = np.polyval(coeffs, df_v['m_t'])
    df_v['predicted_price'] = df_v.apply(lambda row: black_scholes_call_price(
        S=row['S'],
        K=row['K'],
        T=row['TTE'],
        r=r,
        sigma=row['fitted_sigma']
    ), axis=1)

    # Ensure you have actual prices in df_v['mid_price']
    # Plot Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(df_v['mid_price'], df_v['predicted_price'], alpha=1, color='skyblue', edgecolors='k')


    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Predicted vs Actual Option Prices')
    plt.grid(True)
    # plt.legend()
    plt.show()
    return S, r


@app.cell
def _(plt):
    def plot_predict(df):
        plt.title(f'Predicted vs actual for k={df['K'].iloc[0]}')
    
        plt.plot(df['timestamp'],df['predicted_price'],'-', color='red',label='predicted')
        plt.plot(df['timestamp'],df['mid_price'],'-', color='blue',label='actual')
        plt.legend()
        return plt.gca()   
    return (plot_predict,)


@app.cell
def _(plt):
    def plot_residue(df):
        plt.title('Predicted vs Actual Residue')
    
        residue = df['predicted_price'] - df['mid_price']
        plt.plot(df['timestamp'], residue, ".", color='red', label='residue')
    
        # Compute and plot mean residue as a dotted line
        mean_residue = residue.mean()
        plt.axhline(mean_residue, color='blue', linestyle='--', label=f'Mean Residue: {mean_residue:.4f}')
    
        plt.legend()
        return plt.gca()
    return (plot_residue,)


@app.cell
def _(df_v, plot_predict):
    plot_predict(df=df_v[df_v['product']=='VOLCANIC_ROCK_VOUCHER_9750'])
    return


@app.cell
def _(df_v, plot_residue):
    plot_residue(df=df_v[df_v['product']=='VOLCANIC_ROCK_VOUCHER_9500'])
    return


@app.cell
def _(df_v, plot_residue):
    plot_residue(df=df_v[df_v['product']=='VOLCANIC_ROCK_VOUCHER_9750'])
    return


@app.cell
def _(df_v, plot_residue):
    plot_residue(df=df_v[df_v['product']=='VOLCANIC_ROCK_VOUCHER_10000'])
    return


@app.cell
def _(df_v, plot_residue):
    plot_residue(df=df_v[df_v['product']=='VOLCANIC_ROCK_VOUCHER_10250'])
    return


@app.cell
def _(df_v, plot_residue):
    plot_residue(df=df_v[df_v['product']=='VOLCANIC_ROCK_VOUCHER_10500'])
    return


@app.cell
def _(df_v, plot_predict):
    plot_predict(df=df_v[df_v['product']=='VOLCANIC_ROCK_VOUCHER_10500'])
    return


@app.cell
def _(df_v, plot_predict):
    plot_predict(df=df_v[df_v['product']=='VOLCANIC_ROCK_VOUCHER_10250'])
    return


@app.cell
def _(df_v, plot_predict):
    plot_predict(df=df_v[df_v['product']=='VOLCANIC_ROCK_VOUCHER_9750'])
    return


@app.cell
def _(df_v, plot_predict):
    plot_predict(df=df_v[df_v['product']=='VOLCANIC_ROCK_VOUCHER_10000'])
    return


if __name__ == "__main__":
    app.run()
