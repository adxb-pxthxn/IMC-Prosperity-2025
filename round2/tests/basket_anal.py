import marimo

__generated_with = "0.12.7"
app = marimo.App(width="full", app_title="Analysis")


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
    return Polynomial, curve_fit, np, pd, plt


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
def _(combined, plt):


    basket = combined[combined['product'] == 'PICNIC_BASKET1']['mid_price'].values
    dj = combined[combined['product'] == 'DJEMBES']['mid_price'].values
    cros = combined[combined['product'] == 'CROISSANTS']['mid_price'].values
    jam = combined[combined['product'] == 'JAMS']['mid_price'].values

    basket_add =dj + cros * 6 + jam*3


    plt.plot(basket,label='Actual',alpha=0.5)
    plt.plot(basket_add,label='Theo',linestyle="--",color='red',linewidth=2)

    plt.ylim([min(basket.min(), basket_add.min()), max(basket.max(), basket_add.max())])
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
def _(basket, basket_add, plt):
    diff = basket - basket_add
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
        coeffs = Polynomial.fit(x, diff, deg=4)  # Try deg=2,3,4...

        plt.plot(diff, label='Residual')
        plt.plot(coeffs(x), label='Poly Fit', linestyle='--')
        plt.legend()
        return plt.show()


    _()
    return


if __name__ == "__main__":
    app.run()
