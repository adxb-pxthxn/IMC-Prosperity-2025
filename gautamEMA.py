from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math

# storing string as const to avoid typos
SUBMISSION = "SUBMISSION"
RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"

PRODUCTS = [
    RAINFOREST_RESIN,
    KELP,
]

DEFAULT_PRICES = {
    RAINFOREST_RESIN : 10_000,
    KELP : 2000,
}

class Trader:

    def __init__(self, hyperparam=0.11) -> None:

        print("Initializing Trader...")

        self.position_limit = {
            RAINFOREST_RESIN: 20,
            KELP: 20,
        }

        self.round = 0

        # Values to compute pnl
        self.cash = 0
        # positions can be obtained from state.position

        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None

        self.ema_param = hyperparam

    # utils
    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)

    def get_mid_price(self, product, state: TradingState):

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price

        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask) / 2

    def get_value_on_product(self, product, state: TradingState):
        """
        Returns the amount of MONEY currently held on the product.
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)

    def update_pnl(self, state: TradingState):
        """
        Updates the pnl.
        """

        def update_cash():
            # Update cash
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        # Trade was already analyzed
                        continue

                    if trade.buyer == SUBMISSION:
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == SUBMISSION:
                        self.cash += trade.quantity * trade.price

        def get_value_on_positions():
            value = 0
            for product in state.position:
                value += self.get_value_on_product(product, state)
            return value

        # Update cash
        update_cash()
        return self.cash + get_value_on_positions()

    def update_ema_prices(self, state: TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1 - self.ema_param) * self.ema_prices[product]

    # Algorithm logic
    def RAINFOREST_RESIN_strategy(self, state: TradingState):
        """
        Returns a list of orders with trades of RAINFOREST_RESIN.

        Comment: Mudar depois. Separar estrategia por produto assume que
        cada produto eh tradado independentemente
        """

        position_RAINFOREST_RESIN = self.get_position(RAINFOREST_RESIN, state)

        bid_volume = self.position_limit[RAINFOREST_RESIN] - position_RAINFOREST_RESIN
        ask_volume = - self.position_limit[RAINFOREST_RESIN] - position_RAINFOREST_RESIN

        orders = []
        orders.append(Order(RAINFOREST_RESIN, DEFAULT_PRICES[RAINFOREST_RESIN] - 1, bid_volume))
        orders.append(Order(RAINFOREST_RESIN, DEFAULT_PRICES[RAINFOREST_RESIN] + 1, ask_volume))

        return orders

    def KELP_strategy(self, state: TradingState):
        """
        Returns a list of orders with trades of KELP.

        Comment: Mudar depois. Separar estrategia por produto assume que
        cada produto eh tradado independentemente
        """

        position_KELP = self.get_position(KELP, state)

        bid_volume = self.position_limit[KELP] - position_KELP
        ask_volume = - self.position_limit[KELP] - position_KELP

        orders = []

        if position_KELP == 0:
            # Not long nor short
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP] - 1), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP] + 1), ask_volume))

        if position_KELP > 0:
            # Long position
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP] - 2), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP]), ask_volume))

        if position_KELP < 0:
            # Short position
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP]), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP] + 2), ask_volume))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)

        print(f"Log round {self.round}")

        print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    print(trade)

        print(f"\tCash {self.cash}")
        for product in PRODUCTS:
            print(
                f"\tProduct {product}, Position {self.get_position(product, state)}, Midprice {self.get_mid_price(product, state)}, Value {self.get_value_on_product(product, state)}, EMA {self.ema_prices[product]}")
        print(f"\tPnL {pnl}")

        # Initialize the method output dict as an empty dict
        result = {}

        # BANANA STRATEGY
        try:
            result[KELP] = self.KELP_strategy(state)
        except Exception as e:
            print("Error in KELP strategy")
            print(e)

        print("+---------------------------------+")

        return result, 1, "EMA_KELP_Resin"