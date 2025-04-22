import json
from abc import abstractmethod
from collections import deque
import math
from statistics import NormalDist
import numpy as np
from typing import List, Any, TypeAlias, Tuple

from datamodel import TradingState, Symbol, Order, Listing, OrderDepth, Trade, Observation, ProsperityEncoder, \
    ConversionObservation

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


# =======================
# NOTE: THIS IS BOILERPLATE
# region boilerplate
# =======================


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()

# =======================
# NOTE: BOILERPLATE END
# endregion
# =======================


LIMITS = {
    "CROISSANT": 250,
    "JAMS": 350,
    "DJEMBE": 60,
    "KELP": 50,
    "RAINFOREST_RESIN": 50,
    "SQUID_INK": 50,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75
}


# === Base Strategy Class === #
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.conversions = None
        self.orders = None
        self.symbol = symbol
        self.limit = LIMITS[self.symbol]

    @abstractmethod
    def act(self, state: TradingState, traderObject) -> None:  # To be overridden by asset specific action
        raise NotImplementedError()

    def run(self, state: TradingState, traderObject) -> tuple[list[Any], list[Any]]:  # Boiler-plate run method,
        self.orders = []  # just for reference no logic
        self.conversions = []

        self.act(state, traderObject=traderObject)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:  # Default buy function
        self.orders.append(Order(self.symbol, round(price), quantity))

    def sell(self, price: int, quantity: int) -> None:  # Default sell function
        self.orders.append(Order(self.symbol, round(price), -quantity))

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.sell_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position

        self.buy(price, to_buy)
        return to_buy

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.buy_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position

        self.sell(price, to_sell)
        return to_sell
    def convert(self, amount: int) -> None:  # Default convert function for coupons and macrons
        self.conversions += amount

    # === MAKE SURE TO OVERRIDE IF NOT USING PERSISTENT DATA === #
    def save(self) -> JSON:  # Base save function to return dictionary of values,
        return {  # makes key data persistent for next call to run
            "last_price": getattr(self, "last_price", None),
            "window": list(self.window)
        }

    # === MAKE SURE TO OVERRIDE IF NOT USING PERSISTENT DATA === #
    def load(self, data: JSON) -> None:  # Base load function to access persistent values,
        self.last_price = data.get("last_price", None)
        self.window = deque(data.get("window", []))


# === Base Market Making Class === #
class MarketMaking(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque()  # window to keep track of prev ticks actions
        self.window_size = 10  # window size to be optimised

    @abstractmethod
    def get_fair_price(self, order: OrderDepth, traderObject) -> int:  # To be overridden in asset specific class
        raise NotImplementedError

    def act(self, state: TradingState, traderObject) -> None:  # base market making trading logic
        order_depth = state.order_depths[self.symbol]
        fair_price = self.get_fair_price(order_depth, traderObject)

        if fair_price is None:  # if no fair price is given, likely as order book is empty
            return  # immediately return

        # sort each side of the order book #
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # get current position and set current position limits #
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # append a boolean based on if we are at position limit #
        self.window.append(abs(position) == self.limit)  # appends true if we are at position limit
        if len(self.window) > self.window_size:
            self.window.popleft()  # remove left-most when window size goes over

        #  flag for passively executing, occurs when the buffer window is full,
        #  at least half the indices in window are 1 meaning at least half the
        #  ticks in the buffer window did not lead to trades and last tick was not traded
        soft_act = (
                len(self.window) == self.window_size and
                sum(self.window) >= self.window_size / 2 and
                self.window[-1]
        )

        #  flag for aggresive executing, occurs when the buffer window is full,
        #  and trades were not made for every tick in the window
        hard_act = (
                len(self.window) == self.window_size and
                all(self.window)
        )

        # calculate max buy and min sell prices, if inventory less than limit/2
        # else if inventory is more than halfway full, trade at fair_price, this
        # means likely no profit will be made but prevents overextending on asset
        max_buy_price = fair_price - 1 if position > self.limit * 0.5 else fair_price
        min_sell_price = fair_price + 1 if position < self.limit * -0.5 else fair_price

        # for all orders in sell orders
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:  # buy from profitable sell orders
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        # if we should buy aggressively
        if to_buy > 0 and hard_act:
            quantity = to_buy
            self.buy(fair_price, quantity)  # buy at fair price
            to_buy -= quantity

        # if we should buy passively
        if to_buy > 0 and soft_act:
            quantity = to_buy
            self.buy(fair_price - 2, quantity)  # buy under fair price
            to_buy -= quantity

        # if we can buy but there is no market skew
        if to_buy > 0:
            # find the price with most buy orders
            common_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, common_buy_price + 1)  # buy at or above popular price
            self.buy(price, to_buy)

        # same principal for buy orders
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_act:
            quantity = to_sell
            self.sell(fair_price, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_act:
            quantity = to_sell
            self.sell(fair_price + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return {
            "window": list(self.window)
        }

    def load(self, data: JSON) -> None:
        self.window = deque(data.get("window", []))


# === Base Mean Reversion Class === #
class MeanReversion(MarketMaking):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)

    def get_fair_price(self, order: OrderDepth, traderObject) -> int:  # To be overridden in asset specific class
        raise NotImplementedError

    def act(self, state, traderObject):
        # tracks how much we have bought/sold this tick
        buy_volume = 0
        sell_volume = 0

        # standard attributes
        order_depth = state.order_depths[self.symbol]
        fair = self.get_fair_price(order_depth, traderObject)
        position = state.position.get(self.symbol, 0)
        position_limit = self.limit
        sell_orders = order_depth.sell_orders
        buy_orders = order_depth.buy_orders

        try:  # attempts to iteratively get best ask price in the book, if none found trades at fair
            best_ask_fair = min([p for p in sell_orders.keys() if p > fair], default=fair + 1)
        except ValueError:
            best_ask_fair = fair

        try:  # attempts to iteratively get best bid price in the book, if none found trades at fair
            best_bid_fair = max([p for p in buy_orders.keys() if p < fair], default=fair - 1)
        except ValueError:
            best_bid_fair = fair

        # try to fulfill the best sell orders #
        if sell_orders:
            best_ask = min(sell_orders.keys())  # most favourable ask order in the book
            best_ask_amount = -sell_orders[best_ask]
            if best_ask < fair:  # if price is less that fair
                quantity = min(best_ask_amount, position_limit - position)  # fulfill either till limit or full order
                if quantity > 0:
                    self.buy(best_ask, quantity)
                    buy_volume += quantity  # update how much we have bought this tick

        # try to fulfill the best sell orders #
        if buy_orders:
            best_bid = max(buy_orders.keys())  # most favourable ask order in the book
            best_bid_amount = buy_orders[best_bid]
            if best_bid > fair:
                quant = min(best_bid_amount, position_limit + position)  # fulfill either till limit or full order
                if quant > 0:
                    self.sell(best_bid, quant)
                    sell_volume += quant  # update how much we have sold this tick

        # if additional trades can be made, trade on a lesser spread #
        buy_quant = position_limit - (position + buy_volume)
        if buy_quant > 0:
            self.buy(best_bid_fair + 1, buy_quant)

        sell_quant = position_limit + (position - sell_volume)
        if sell_quant > 0:
            self.sell(best_ask_fair - 1, sell_quant)


# === Rainforest Resin Trading Strategy Class === #
class ResinStrategy(MeanReversion):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.ewm = EWM(0.15)  # set exponential weighted moving average alpha for helper
        self.last_price = 10000  # base last price is 10k as that is practically the fair value

    def get_mid_price(self, order, traderObject):
        # if order book has depth on both sides #
        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            self.last_price = (best_bid + best_ask) / 2.0
            return self.last_price  # update the last_price to be mid-price

    def get_fair_price(self, order, traderObject):  # fair-price is the EWMA again from the helper
        return self.ewm.update(self.get_mid_price(order, traderObject))

    # === ACTUAL ACTION IS THE SAME AS MEAN REVERSION === #

    def save(self) -> JSON:
        return {
            "last_price": getattr(self, "last_price", None),
        }

    def load(self, data: JSON) -> None:
        self.last_price = data.get("last_price", None)


# === Kelp Trading Strategy Class === #
class KelpStrategy(MarketMaking):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.last_price = None  # set to none by default

    def get_fair_price(self, order_depth: OrderDepth, traderObject) -> float:
        # if there are orders on both side of the book
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())  # current market best ask
            best_bid = max(order_depth.buy_orders.keys())  # current market best bid

            # filtering logic for bid and ask prices #
            # to calculate a reliable mid-price we filter out
            # asks and bids which fall below the adverse_volume
            # this removes noise from outliers
            adverse_volume = 15

            filtered_ask = [  # constructs a list of ask prices with enough volume
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= adverse_volume
            ]
            filtered_bid = [  # constructs a list of bid prices with enough volume
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= adverse_volume
            ]

            # pick the best prices for ask/bid out of this filtered set
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            # if any filtered results were empty meaning the book is imbalanced/sparse
            if mm_ask is None or mm_bid is None:
                if traderObject.get("KELP_last_price", None) is None:  # if last price was also None
                    best_price = (best_ask + best_bid) / 2  # use best ask and best bid for mid
                else:
                    best_price = traderObject["KELP_last_price"]  # if not then use the last price
            else:
                best_price = (mm_ask + mm_bid) / 2  # else we use the filtered data for mid price

            # if we have a valid last price
            if self.last_price is not None:
                last_returns = (best_price - self.last_price) / self.last_price
                pred_returns = last_returns * -0.229
                fair = best_price + (best_price * pred_returns)
            else:
                fair = best_price

            self.last_price = best_price
            return fair
        else:
            return None

    def save(self) -> JSON:
        return {
            "last_price": getattr(self, "last_price", None),
            "window": list(self.window)
        }

    def load(self, data: JSON) -> None:
        self.last_price = data.get("last_price", None)
        self.window = deque(data.get("window", []))


# === Squid Ink Trading Strategy Class === #
class SquidInkStrategy(MarketMaking):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.last_price = None
        self.price_history = []
        self.tickCount = 0
        self.last_trade_time = -1000
        self.window_size = 75

    def compute_volatility(self, prices: list[float]) -> float:
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

    def should_quote(
            self,
            order_depth: OrderDepth,
            vol: float,
            mm_ask: float,
            mm_bid: float
    ) -> bool:
        if mm_ask is None or mm_bid is None:
            return False
        if len(order_depth.sell_orders) < 2 or len(order_depth.buy_orders) < 2:
            return False
        if vol > 0.02:
            return False

        self.tickCount += 1

        cooldown_ms = 15
        if self.tickCount < self.last_trade_time + cooldown_ms:
            return False
        self.last_trade_time = self.tickCount
        return True

    def get_fair_price(self, order_depth: OrderDepth, traderObject) -> float:
        traderObject = traderObject
        min_volume = 15

        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return None

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        filtered_ask = [
            price for price, volume in order_depth.sell_orders.items()
            if abs(volume) >= min_volume
        ]
        filtered_bid = [
            price for price, volume in order_depth.buy_orders.items()
            if abs(volume) >= min_volume
        ]

        mm_ask = min(filtered_ask) if filtered_ask else None
        mm_bid = max(filtered_bid) if filtered_bid else None

        if mm_ask is None or mm_bid is None:
            mid_price = traderObject.get("SQUID_last_price", (best_ask + best_bid) / 2)
        else:
            mid_price = (mm_ask + mm_bid) / 2

        self.price_history.append(mid_price)
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)

        vol = self.compute_volatility(self.price_history)
        vol = max(vol, 0.0001)

        alpha = max(0.05, 0.3 - vol * 10)
        beta = max(0.05, 0.25 - vol * 10)
        spread_padding = vol * 1000

        historical_mean = sum(self.price_history) / len(self.price_history)
        deviation = mid_price - historical_mean

        if self.last_price:
            ret = (mid_price - self.last_price) / self.last_price
        else:
            ret = 0.0

        if (deviation > 0 and ret < 0) or (deviation < 0 and ret > 0):
            reversion_adjustment = -alpha * deviation
        else:
            reversion_adjustment = 0.0

        momentum_adjustment = -ret * beta * mid_price

        if not self.should_quote(order_depth, vol, mm_ask, mm_bid):
            return None

        fair_price = mid_price + reversion_adjustment + momentum_adjustment - spread_padding

        self.last_price = mid_price
        return fair_price


# === Basket 1 Trading Strategy Class === #
class Basket1Strat(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self.window = deque()
        self.params = [-20, -50]
        self.ewm = EWM(1 / 4500)


    def get_mid_price(self, order, traderObject=None):
        
        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            return (best_bid + best_ask) / 2.0


    def act(self, state, traderObject):
        order_depth = state.order_depths[self.symbol]
        # position=state.position.get(self.symbol, 0)

        basket = self.get_mid_price(order_depth)

        cros, jams = self.get_mid_price(state.order_depths['CROISSANTS']), self.get_mid_price(
            state.order_depths['JAMS'])

        diff = basket - (4 * cros + 2 * jams)

        signal = self.ewm.update(diff) - diff

        # Order book for trading GIFT_BASKET
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        threshold = [80, -75]

        if signal > threshold[0]:
            self.go_long(state)
        elif signal < threshold[1]:
            self.go_short(state)


# === Basket 2 Trading Strategy Class === #
class Basket2Strat(Strategy):

    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self.window = deque()
        self.params = [0.99, 0.98, 0.98, 0.98]
        self.ewm = EWM(1 / 4500)

    def get_mid_price(self, order, traderObject=None):

        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            return (best_bid + best_ask) / 2.0

    def act(self, state, traderObject):

        order_depth = state.order_depths[self.symbol]

        basket = self.get_mid_price(order_depth)

        cros, jams = self.get_mid_price(state.order_depths['CROISSANTS']), self.get_mid_price(
            state.order_depths['JAMS'])

        diff = basket - (4 * cros + 2 * jams)

        signal = self.ewm.update(diff) - diff

        # Order book for trading GIFT_BASKET
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return


        threshold = [40, -40]

        if signal > threshold[0]:
            self.go_long(state)
        elif signal < threshold[1]:
            self.go_short(state)



# === Macrons Trading MR Strategy Class === #
class MacaronMeanReversionStrategy(MeanReversion):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.last_fair = None  # optional: for debugging/logging

    def get_fair_price(self, order: OrderDepth, traderObject) -> int:
        try:
            obs: ConversionObservation = traderObject["conversionObservations"][self.symbol]
        except (KeyError, TypeError):
            # fallback fair price if observation is missing
            return None  # or whatever default makes sense to you

        fair = (
                188.4 +
                61.5 * obs.transportFees -
                62.5 * obs.exportTariff -
                52.1 * obs.importTariff +
                4.97 * obs.sugarPrice -
                3.31 * obs.sunlightIndex
        )
        self.last_fair = fair
        return round(fair)

class MacaronSignalsStrategy(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.CSI = 46
        self.means = {
            "sugarPrice": 216.73,
            "sunlightIndex": 60.82,
            "importTariff": -3.49,
            "exportTariff": 9.67,
            "transportFees": 1.42
        }
        self.stds = {
            "sugarPrice": 14.28,
            "sunlightIndex": 5.44,
            "importTariff": 0.47,
            "exportTariff": 0.61,
            "transportFees": 0.52
        }
        self.weights = np.array([-0.382, 0.191, -1.094, 0.386, 1.171])
        self.bias = -2.712
        self.longSignal = 0.55
        self.shortSignal = 0.45
        self.last_price = None
        self.conversions = 0

    def get_signal(self, obs):
        x = np.array([
            (obs.sugarPrice - self.means["sugarPrice"]) / self.stds["sugarPrice"],
            (obs.sunlightIndex - self.means["sunlightIndex"]) / self.stds["sunlightIndex"],
            (obs.importTariff - self.means["importTariff"]) / self.stds["importTariff"],
            (obs.exportTariff - self.means["exportTariff"]) / self.stds["exportTariff"],
            (obs.transportFees - self.means["transportFees"]) / self.stds["transportFees"],
        ])
        z = np.dot(self.weights, x) + self.bias
        return 1 / (1 + np.exp(-z))

    def act(self, state: TradingState, traderObject):
        self.conversions = 0
        order_depth = state.order_depths[self.symbol]
        obs = traderObject["conversionObservations"].get(self.symbol)
        if obs is None:
            return

        position = state.position.get(self.symbol, 0)
        limit = self.limit
        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)

        # Update mid-price for profit-taking
        if best_bid is not None and best_ask is not None:
            self.last_price = (best_bid + best_ask) / 2

        pristine_obs = traderObject["conversionObservations"].get(self.symbol)
        if pristine_obs:
            pristine_total_cost = pristine_obs.askPrice + pristine_obs.importTariff + pristine_obs.transportFees
        else:
            pristine_total_cost = float('inf')  # fallback to prevent blocking logic

        # --- PANIC MODE (buy aggressively from Pristine or market) ---
        if obs.sunlightIndex < self.CSI:
            if position < limit:
                # Prefer Pristine if cheaper
                buy_amount = min(limit - position, 10)
                if pristine_total_cost < (best_ask if best_ask is not None else float('inf')):
                    self.convert(buy_amount)
                    position += buy_amount
                else:
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        ask_volume = -order_depth.sell_orders[ask_price]
                        quantity = min(ask_volume, limit - position)
                        if quantity <= 0:
                            break
                        self.buy(ask_price, quantity)
                        position += quantity
            # Always try to convert what we hold
            if position > 0:
                self.convert(min(position, 10))

        else:
            # --- SIGNAL MODE ---
            signal = self.get_signal(obs)
            if signal > self.longSignal and position < limit:
                buy_amount = min(limit - position, 10)
                if pristine_total_cost < (best_ask if best_ask is not None else float('inf')):
                    self.convert(buy_amount)
                elif best_ask is not None:
                    volume = min(limit - position, abs(order_depth.sell_orders[best_ask]))
                    self.buy(best_ask, volume)
            elif signal < self.shortSignal and best_bid is not None and position > -limit:
                volume = min(limit + position, order_depth.buy_orders[best_bid])
                self.sell(best_bid, volume)

        # --- PROFIT-TAKE MODE ---
        if position > 0 and pristine_obs:
            pristine_bid = pristine_obs.bidPrice
            net_conversion_value = pristine_bid - pristine_obs.exportTariff - pristine_obs.transportFees

            market_bid_ok = best_bid is not None and best_bid >= net_conversion_value
            pristine_better_than_fair = self.last_price is not None and net_conversion_value > self.last_price + 20

            if market_bid_ok:
                volume = min(position, order_depth.buy_orders[best_bid])
                self.sell(best_bid, volume)
            elif pristine_better_than_fair:
                self.convert(min(position, 10))

    def run(self, state: TradingState, traderObject):
        self.orders = []
        self.act(state, traderObject)
        return self.orders, self.conversions

    def convert(self, amount: int) -> None:
        self.conversions = int(amount)

    def save(self) -> JSON:
        return {"last_price": getattr(self, "last_price", None)}

    def load(self, data: JSON) -> None:
        self.last_price = data.get("last_price", None)

class EWM:
    def __init__(self, alpha=0.002):
        self.alpha = alpha
        self.value = None

    def update(self, price):
        if self.value is None:
            self.value = price
        else:
            self.value = self.alpha * price + (1 - self.alpha) * self.value
        return self.value


class EWMAbs:

    def __init__(self, price_alpha=0.0015, dev_alpha=0.002, long_alpha=0.0001):
        self.price_ema = EWM(alpha=price_alpha)
        self.long_ema = EWM(alpha=long_alpha)
        self.deviation_ema = EWM(alpha=dev_alpha)

    def update(self, price):
        ema_price = self.price_ema.update(price)
        long_ema = self.long_ema.update(price)

        abs_deviation = abs(price - ema_price)

        ema_abs_deviation = self.deviation_ema.update(abs_deviation)

        return ema_price, ema_abs_deviation, long_ema


def rough_iv(S, K, T, C):
    approx_iv = C / (0.2 * S * np.sqrt(T))
    approx_iv = np.clip(approx_iv, 0.01, 2.0)
    return approx_iv


class CouponStrategy(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self.voucher_strikes = [9500, 9750, 10000, 10250, 10500]
        self.smile_window = deque(maxlen=100)
        self.window=[]


    def run(self, state: TradingState, traderObject) -> tuple[list[Any], list[Any]]:
        self.orders = []
        self.conversions = []

        under_order=self.act(state, traderObject=traderObject)
        return self.orders, self.conversions,under_order
        
    def get_sigma(self, S, K, T):
        coeffs = [0.28486795, 0.01185442, 0.14024405]

        m = math.log(K / S) / math.sqrt(T)

        sigma=np.polyval(coeffs,m)

        return sigma
        

    


    def get_greeks(self,S, K, T, r, sigma):
        """
        Black‑Scholes call price, delta and vega using only numpy + statistics.NormalDist
        """
        # edge cases: immediate expiry or zero vol
        if T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0.0)
            # delta is 1 if in‑the‑money, else 0
            delta = 1.0 if S > K else 0.0
            return intrinsic, delta, 0.0

        # compute d1, d2
        sqrtT = np.sqrt(T)
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT

        # use the standard normal CDF / PDF from statistics.NormalDist
        std_norm = NormalDist()  
        Nd1 = std_norm.cdf(d1)
        nd1 = std_norm.pdf(d1)

        # price, delta, vega
        price = S * Nd1 - K * np.exp(-r*T) * std_norm.cdf(d2)
        delta = Nd1
        vega  = S * nd1 * sqrtT

        return price, delta, vega

    def get_mid_price(self, order, traderObject=None):
        
        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            return (best_bid + best_ask) / 2.0



    def act(self, state, traderObject) -> None:
        base_asset = "VOLCANIC_ROCK"
        if base_asset not in state.order_depths:
            return

        for sym in [base_asset, self.symbol]:
            if sym not in state.order_depths or not (
                state.order_depths[sym].buy_orders and state.order_depths[sym].sell_orders
            ):
                return

        rock = self.get_mid_price(state.order_depths[base_asset])
        coupon = self.get_mid_price(state.order_depths[self.symbol])
        if rock is None or coupon is None:
            return

        # timestamp=state.timestamp
        # mill=1_000_000
        K = int(self.symbol.split('_')[-1])
        T = 4/365

        r = 0

        sigma=self.get_sigma(rock,K,T)

        fair,delta,vega=self.get_greeks(rock,K,T,r,sigma)






        order_depth = state.order_depths[self.symbol]
        buy_price = max(order_depth.buy_orders)
        sell_price = min(order_depth.sell_orders)
        buy_vol = order_depth.buy_orders[buy_price]
        sell_vol = order_depth.sell_orders[sell_price]


        hedge_order = []
        residual = fair-coupon


        # thresh={
        #     9500:1.5,
        #     9750:2,
        #     10000:2,
        #     10250:2,
        #     10500:1.5

        # }
        threshold = 2

        


        hedge_thresh=0.75

        # logger.print(f"Residual:{residual}")

        if residual < -threshold :
            short=self.go_short(state)
            
            hedge = int(min(delta * abs(buy_vol),delta* abs(short)))

            if hedge > 0:
                hedge_order.append(Order(base_asset, round(rock),round( hedge*hedge_thresh)))

    
        elif residual >threshold:
            long=self.go_long(state)

            hedge = int(min(delta * abs(sell_vol), delta*abs(long)))

            if hedge > 0:
                hedge_order.append(Order(base_asset, round(rock), -round(hedge*hedge_thresh)))

        return hedge_order



class Trader:
    def __init__(self) -> None:
        self.strategies = {}
        self.strategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, LIMITS[symbol]) for symbol, clazz in {
            "KELP": KelpStrategy,
            "RAINFOREST_RESIN": ResinStrategy,
            "SQUID_INK": SquidInkStrategy,
            "PICNIC_BASKET1": Basket1Strat,
            "PICNIC_BASKET2": Basket2Strat,
      
            # "MAGNIFICENT_MACARONS": MacaronSignalsStrategy
        }.items()}
        self.couponStrategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, LIMITS[symbol]) for symbol, clazz in {
            "VOLCANIC_ROCK_VOUCHER_9500":CouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_9750":CouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_10500":CouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_10250":CouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_10000":CouponStrategy,
        }.items()}
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if (
                    symbol in state.order_depths
                    and len(state.order_depths[symbol].buy_orders) > 0
                    and len(state.order_depths[symbol].sell_orders) > 0
            ):
                trader_object = old_trader_data.get(symbol, {})

                # Inject conversionObservations safely
                if hasattr(state.observations, "conversionObservations"):
                    trader_object["conversionObservations"] = (
                            state.observations.conversionObservations or {}
                    )

                strategy_orders, strategy_conversions = strategy.run(
                    state, traderObject=trader_object
                )

                orders[symbol] = strategy_orders
                conversions += int(strategy_conversions or 0)
        #trade coupons
        rock_orders=[]
        for symbol, strategy in self.couponStrategies.items():
                if symbol in old_trader_data:
                    strategy.load(old_trader_data[symbol])

                if (symbol in state.order_depths and
                        len(state.order_depths[symbol].buy_orders) > 0 and
                        len(state.order_depths[symbol].sell_orders) > 0
                ):
                    strategy_orders, strategy_conversions,under_trade = strategy.run(state,
                                                                        traderObject=old_trader_data.get(symbol, {}), )

                    orders[symbol] = strategy_orders
                    rock_orders.extend(under_trade)
                    
                    conversions += sum(strategy_conversions)

                new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        orders['VOLCANIC_ROCK']=rock_orders

        new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
