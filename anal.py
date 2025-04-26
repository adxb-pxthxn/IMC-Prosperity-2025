import marimo

__generated_with = "0.12.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import math
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from randomcolor import RandomColor
    from typing import Any
    return Any, RandomColor, go, make_subplots, math, np, pd, px


@app.cell
def _(pd):
    from pathlib import Path

    DATA_ROOT = Path('data/')

    def get_prices(round_num: int, day_num: int) -> pd.DataFrame:
        return pd.read_csv(DATA_ROOT / f"round{round_num}" / f"prices_round_{round_num}_day_{day_num}.csv", sep=";")

    def get_trades(round_num: int, day_num: int) -> pd.DataFrame:

        file = DATA_ROOT / f"round{round_num}" / f"trades_round_{round_num}_day_{day_num}.csv"
        if file.is_file():
            return pd.read_csv(file, sep=";")
        raise ValueError(f"Cannot find trades data for round {round_num} day {day_num}")
    return DATA_ROOT, Path, get_prices, get_trades


@app.cell
def _(Any, math, np, pd):
    def get_popular_price(row: Any, bid_ask: str) -> int:
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
            mid_price = (popular_buy_price + popular_sell_price) / 2

            if mid_price < 500 and product == "COCONUT_COUPON":
                mid_price = mid_prices[-1]

            mid_prices.append(mid_price)

        return np.array(mid_prices)
    return get_popular_price, get_product_prices


@app.cell
def _(
    RandomColor,
    get_prices,
    get_product_prices,
    get_trades,
    go,
    make_subplots,
    pd,
):

    days = [ [5, [2, 3, 4]]]

    traders = [
        "Amir",
        "Ayumi",
        "Ari",
        "Anika",
        "Boris",
        "Bashir",
        "Bonnie",
        "Blue",
        "Sanjay",
        "Sami",
        "Sierra",
        "Santiago",
        "Mikhail",
        "Mina",
        "Morgan",
        "Manuel",
        "Carlos",
        "Candice",
        "Carson",
        "Cristiano"
    ]
    traders = set()

    for round_num, day_nums in days:
        for day_num in day_nums:
            trades = get_trades(round_num, day_num)
            traders = traders.union(trades["buyer"])
            traders = traders.union(trades["seller"])

    traders = sorted(traders)
    traders


    combinations = set()

    for round_num, day_nums in days:
        for day_num in day_nums:
            trades = get_trades(round_num, day_num)
            for row in trades.itertuples():
                combinations.add((row.buyer, row.seller))

    sorted(combinations)

    for round_num, day_nums in days:
        prices = pd.DataFrame()
        trades = pd.DataFrame()

        for day_num in day_nums:
            if len(prices) == 0:
                timestamp_offset = 0
            else:
                timestamp_offset = int(prices["timestamp"].tail(1).iloc[0])

            day_prices = get_prices(round_num, day_num)
            day_trades = get_trades(round_num, day_num)

            day_prices["timestamp"] += timestamp_offset
            day_trades["timestamp"] += timestamp_offset

            prices = pd.concat([prices, day_prices])
            trades = pd.concat([trades, day_trades])

        for product in sorted(prices["product"].unique()):
            product_prices = get_product_prices(prices, product)
            product_trades = trades[trades["symbol"] == product]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=prices[prices["product"] == product]["timestamp"], y=product_prices, name="Price", line={"color": "gray"}))

            colors = RandomColor(seed=0)

            for trader in traders:
                for side in ["buyer", "seller"]:
                    trader_trades = product_trades[product_trades[side] == trader]
                    fig.add_trace(go.Scatter(
                        x=trader_trades["timestamp"],
                        y=trader_trades["price"],
                        mode="markers",
                        name=f"{side} = {trader}",
                        visible="legendonly",
                        line={"color": colors.generate()[0]},
                    ))

            fig.update_layout(title_text=f"Round {round_num} - {product}")
            fig.show()
    for round_num, day_nums in days:
        prices = pd.DataFrame()
        trades = pd.DataFrame()

        for day_num in day_nums:
            if len(prices) == 0:
                timestamp_offset = 0
            else:
                timestamp_offset = int(prices["timestamp"].tail(1).iloc[0])

            day_prices = get_prices(round_num, day_num)
            day_trades = get_trades(round_num, day_num)

            day_prices["timestamp"] += timestamp_offset
            day_trades["timestamp"] += timestamp_offset

            prices = pd.concat([prices, day_prices])
            trades = pd.concat([trades, day_trades])

        for product in sorted(prices["product"].unique()):
            product_prices = get_product_prices(prices, product)
            product_trades = trades[trades["symbol"] == product]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=prices[prices["product"] == product]["timestamp"], y=product_prices, name="Price", line={"color": "gray"}))

            colors = RandomColor(seed=0)

            for buyer in traders:
                for seller in traders:
                    trader_trades = product_trades[(product_trades["buyer"] == buyer) & (product_trades["seller"] == seller)]
                    if len(trader_trades) < 2 or len(trader_trades) > 100:
                        continue

                    fig.add_trace(go.Scatter(
                        x=trader_trades["timestamp"],
                        y=trader_trades["price"],
                        mode="markers",
                        name=f"{buyer}/{seller}",
                        visible="legendonly",
                        line={"color": colors.generate()[0]},
                    ))

            fig.update_layout(title_text=f"Round {round_num} - {product}")
            fig.show()
    return (
        buyer,
        colors,
        combinations,
        day_num,
        day_nums,
        day_prices,
        day_trades,
        days,
        fig,
        prices,
        product,
        product_prices,
        product_trades,
        round_num,
        row,
        seller,
        side,
        timestamp_offset,
        trader,
        trader_trades,
        traders,
        trades,
    )


if __name__ == "__main__":
    app.run()
