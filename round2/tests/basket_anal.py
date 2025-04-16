import marimo

__generated_with = "0.12.7"
app = marimo.App(
    width="full",
    app_title="Analysis",
    layout_file="layouts/basket_anal.grid.json",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis of Basket Data""")
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    from numpy.polynomial.polynomial import Polynomial
    import math
    return Polynomial, curve_fit, math, np, pd, plt


@app.cell
def _(pd):
    d1=pd.read_csv('round2-data/prices_round_2_day_-1.csv',sep=";")
    d2=pd.read_csv('round2-data/prices_round_2_day_0.csv',sep=";")
    d3=pd.read_csv('round2-data/prices_round_2_day_1.csv',sep=";")
    return d1, d2, d3


@app.cell
def _(d1, d2, d3, pd):
    combined=pd.concat([d1,d2,d3],ignore_index=True)
    combined.loc[combined['day'] == 0, 'timestamp'] += 1000000
    combined.loc[combined['day'] == 1, 'timestamp'] += 2000000
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
        prices = prices[prices["product"] == product]

        mid_prices = []
        for row in prices.itertuples():
            popular_buy_price = get_popular_price(row, "bid")
            popular_sell_price = get_popular_price(row, "ask")
            mid_prices.append((popular_buy_price + popular_sell_price) / 2)

        return np.array(mid_prices)

    return get_popular_price, get_product_prices


@app.cell
def _(combined, get_product_prices, plt):
    basket = get_product_prices(combined,'PICNIC_BASKET1')
    dj = get_product_prices(combined,'DJEMBES')
    cros = get_product_prices(combined,'CROISSANTS')
    jam = get_product_prices(combined,'JAMS')

    basket_add =dj + cros * 6 + jam*3


    plt.plot(basket,label='Actual',alpha=0.5)
    plt.plot(basket_add,label='Theo',linestyle="--",color='red',linewidth=2)


    plt.legend()
    plt.show()


    return basket, basket_add, cros, dj, jam


@app.cell
def _(cros, dj, jam, plt):
    plt.plot(cros,label='Cros',linestyle="--",linewidth=0.5)
    plt.show()
    plt.plot(dj,label='dj',linestyle="--",linewidth=0.5)
    plt.show()
    plt.plot(jam,label='jam',linestyle="--",linewidth=0.5)
    plt.show()
    return


@app.cell
def _(combined, cros, jam, plt):
    basket2 = combined[combined['product'] == 'PICNIC_BASKET2']['mid_price'].values

    basket2_add = cros * 4 + jam*2


    plt.plot(basket2,label='Actual',alpha=0.5)
    plt.plot(basket2_add,label='Theo',linestyle="--",color='red',linewidth=2)
    plt.ylim([min(basket2.min(), basket2_add.min()), max(basket2.max(), basket2_add.max())])
    plt.legend()
    plt.show()
    return basket2, basket2_add


@app.cell
def _(diff, diff2):
    (diff.mean(),diff.std()),(diff2.mean(),diff2.std())
    (diff.mean(),diff.std()),(diff2.mean(),diff2.std())
    return


@app.cell
def _(basket, basket_add, np, plt):
    diff = np.array(basket) - np.array(basket_add)
    plt.plot(diff, label='Residual (Actual - Theo)')
    plt.hlines(diff.mean(), xmin=0, xmax=len(diff)-1, colors='red', linestyles='--', label='Mean Residual')
    plt.legend()
    plt.show()
    print(diff.mean())
    return (diff,)


@app.cell
def _(basket2, basket2_add, plt):
    diff2 = basket2 - basket2_add
    plt.plot(diff2, label='Residual (Actual - Theo)')
    plt.hlines(diff2.mean(), xmin=0, xmax=len(diff2)-1, colors='red', linestyles='--', label='Mean Residual')
    plt.legend()
    plt.show()
    print(diff2.mean())
    return (diff2,)


@app.cell(hide_code=True)
def _(mo):
    add1=mo.ui.slider(start=30,stop=100,step=0.1)
    add1
    return (add1,)


@app.cell(hide_code=True)
def _(add1, mo):
    mo.md(f"{add1.value}")
    return


@app.cell
def _(add1, basket, basket_add, cros, dj, jam, mo, plt):
    @mo.cache
    def _():
        basket_predict=dj + cros * 6 + jam*3 +add1.value
        plt.plot(basket,label='Actual',alpha=0.5)
        plt.plot(basket_predict,label='Theo',linestyle="--",color='red',linewidth=2)

        plt.ylim([min(basket.min(), basket_add.min()), max(basket.max(), basket_add.max())])
        plt.legend()
        return plt.gca()


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    add2=mo.ui.slider(start=30,stop=100,step=0.1)
    add2
    return (add2,)


@app.cell(hide_code=True)
def _(add2, mo):
    mo.md(f"{add2.value}")
    return


@app.cell
def _(add2, basket2, basket2_add, cros, jam, mo, plt):
    @mo.cache
    def _():
        basket2_predict= cros * 4 + jam*2 +add2.value
        plt.plot(basket2,label='Actual',alpha=0.5)
        plt.plot(basket2_predict,label='Theo',linestyle="--",color='red',linewidth=2)

        plt.ylim([min(basket2.min(), basket2_add.min()), max(basket2.max(), basket2_add.max())])
        plt.legend()
        return plt.gca()


    _()
    return


@app.cell
def _(basket2, basket2_add, np, plt):
    def _():
        # Your existing residual
        diff = basket2 - basket2_add

        # Create x values
        x = np.linspace(0, 2 * np.pi, len(diff))

        # Generate sine wave (match amplitude & shift if you want)
        sine_wave = (diff.std())* np.sin(2.2*x+75) + diff.mean()-20  # scaled & shifted

        # Plot
        plt.plot(diff, label='Residual (Actual - Theo)', color='blue')
        plt.plot(sine_wave, label='Sine Approx', linestyle='--', color='orange')
        plt.title('Residual vs. Sine Wave')
        plt.legend()

        return plt.show()


    _()
    return


@app.cell
def _(curve_fit, diff, np, plt):
    x = np.linspace(0, 2 * np.pi, len(diff)) 
    def sine_wave(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    print(diff.std())
    # 3. Initial guess for parameters [Amplitude, Frequency, Phase, Offset]
    guess = [170, 2.2, -150, diff.mean()]

    # 4. Curve fitting
    params, _ = curve_fit(sine_wave, x, diff, p0=guess)
    A, B, C, D = params

    print(params)
    # 5. Generate fitted sine wave
    x = np.linspace(0, 2 * np.pi, len(diff)+2000) 
    fitted_wave = sine_wave(x, A, B, C, D)
    # 6. Plot actual vs fitted sine wave
    plt.plot(diff, label='Residual', color='blue')
    plt.plot(fitted_wave, label='Fitted Sine Wave', linestyle='--', color='orange')
    plt.title('Sine Wave Fitting')
    plt.legend()
    plt.grid(True)
    plt.show()
    return A, B, C, D, fitted_wave, guess, params, sine_wave, x


@app.cell
def _(Polynomial, diff, np, plt):
    def _():
        x = np.arange(len(diff))
        coeffs = Polynomial.fit(x, diff, deg=5)  # Try deg=2,3,4...
        print(coeffs)
        plt.plot(diff, label='Residual')
        plt.plot(coeffs(x), label='Poly Fit', linestyle='--')
        plt.legend()
        return plt.show()




    _()
    return


@app.cell
def _(mo):
    buffer=mo.ui.slider(start=0,stop=250,step=1)
    buffer
    return (buffer,)


@app.cell
def _(mo):
    window=mo.ui.slider(start=100,stop=10000,step=100)
    window
    return (window,)


@app.cell
def _(buffer, mo, window):
    mo.md(f'{buffer.value} buffer, {window.value} window')
    return


@app.cell
def _(EWM, diff2, np, plt):
    # Create a new EWM instance
    ewm = EWM(alpha=1 / (2401))  # match your window size

    # Calculate the signal
    ema_values = []
    signal_values = []

    for d in diff2:
        ema_val = ewm.update(d)
        ema_values.append(ema_val)
        signal_values.append(ema_val - d)

    # Plot signal
    plt.plot(signal_values, label='Signal (EMA - diff)', color='purple')
    plt.axhline(np.array(signal_values).mean(), color='gray', linestyle='--', linewidth=1)
    plt.legend()
    plt.title("Signal = EMA - diff")
    plt.show()
    return d, ema_val, ema_values, ewm, signal_values


@app.cell
def _(basket, dj, np):
    a = np.array(dj) 
    b = np.array(basket) 

    corr = np.corrcoef(a, b)[0, 1]
    print(f"Correlation: {corr:.4f}")
    return a, b, corr


app._unparsable_cell(
    r"""
    def _():
        def _():
            a = np.array(jam) 
            b = np.array(basket) 

            corr = np.corrcoef(a, b)[0, 1]
        return _()
            return print(f\"Correlation: {corr:.4f}\")


    _()
    """,
    name="_"
)


@app.cell
def _(buffer, diff2, mo, np, plt, window):
    class EWM:
        def __init__(self,alpha=0.001):
            self.alpha=alpha
            self.value=None
        def update(self,price):
            if self.value is None:
                self.value = price 
            else:
                self.value = self.alpha * price + (1 - self.alpha) * self.value
            return self.value

    def ema(data, window):
        alpha = 2 / (window + 1)
        ema_values = np.zeros_like(data)
        ema_values[0] = data[0]  # initialize
        for t in range(1, len(data)):
            ema_values[t] = alpha * data[t] + (1 - alpha) * ema_values[t - 1]
        return ema_values

    ema_10 = ema(diff2, window=window.value
                )
    @mo.cache
    def _():
        plt.plot(diff2, label='Original',color='white', linestyle="--",alpha=0.5)
        plt.plot(ema_10, label='EMA (window=3000)', color='blue')

        # Add shaded buffer around the EMA line
        plt.fill_between(
            np.arange(len(ema_10)),
            ema_10 - buffer.value,
            ema_10 + buffer.value,
            color='white',
            alpha=0.5,
            label='±10 buffer'
        )

        plt.legend()
        plt.title("Exponential Moving Average with ±10 Buffer")
        return plt.gca()

    _()
    return EWM, ema, ema_10


@app.cell
def _(basket, cros, np):
    def _():
        a = np.array(cros) 
        b = np.array(basket) 

        corr = np.corrcoef(a, b)[0, 1]
        return print(f"Correlation: {corr:.4f}")


    _()
    return


@app.cell
def _(basket, diff, np):
    def _():
        def _():
            a = np.array(basket) 
            b = np.array(diff) 

            corr = np.corrcoef(a, b)[0, 1]
            return print(f"Correlation: {corr:.4f}")
        return _()


    _()
    return


if __name__ == "__main__":
    app.run()
