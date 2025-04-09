import json
from abc import abstractmethod
from collections import deque
from typing import List, Any, TypeAlias, Tuple
import jsonpickle

from datamodel import TradingState, Symbol, Order, Listing, OrderDepth, Trade, Observation, ProsperityEncoder

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
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


# =======================
# NOTE: BOILERPLATE END
# endregion
# =======================

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState, traderObject) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState, traderObject) -> tuple[list[Any], list[Any]]:
        self.orders = []
        self.conversions = []

        self.act(state, traderObject=traderObject)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, round(price), quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, round(price), -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return {
            "last_price": getattr(self, "last_price", None),
            "window": list(self.window)
        }

    def load(self, data: JSON) -> None:
        self.last_price = data.get("last_price", None)
        self.window = deque(data.get("window", []))


class MarketMaking(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_fair_price(self, order: OrderDepth, traderObject) -> int:
        raise NotImplementedError

    def act(self, state: TradingState, traderObject) -> None:
        order_depth = state.order_depths[self.symbol]

        if self.symbol is "KELP":
            fair_price = self.get_fair_price(order_depth, traderObject)
        else:
            fair_price = self.get_fair_price(order_depth, traderObject=None)

        # get and sort each side of the order book #
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # get current position and set current position limits #
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # append a boolean based on if we are at position limit #
        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()  # remove left-most when size goes over

        #  flag for passively executing, occurs when the buffer window is full,
        #  at least half the indexs in window are 1 meaning at least half the
        # ticks in the buffer window trades were executed, prev ticker also 1
        soft_liquidate = (
                len(self.window) == self.window_size and
                sum(self.window) >= self.window_size / 2 and
                self.window[-1]
        )

        #  flag for aggresive executing, occurs when the buffer window is full,
        #  and trades were made for every tick within the buffer window
        hard_liquidate = (
                len(self.window) == self.window_size and
                all(self.window)
        )

        # calculate max buy and min sell prices, if inventory less than limit/2
        max_buy_price = fair_price - 1 if position > self.limit * 0.5 else fair_price
        min_sell_price = fair_price + 1 if position < self.limit * -0.5 else fair_price

        # for all orders in sell orders
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:  # fulfil profitable sell orders
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        # if can buy aggressively
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy
            self.buy(fair_price, quantity)  # buy at fair price
            to_buy -= quantity

        # if can buy passively
        if to_buy > 0 and soft_liquidate:
            quantity = to_buy
            self.buy(fair_price - 2, quantity)  # buy under fair price
            to_buy -= quantity

        # if can buy but no skew
        if to_buy > 0:
            # find the price with most buy orders
            common_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, common_buy_price + 1)  # buy at or above popular price
            self.buy(price, to_buy)

        # same principal
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell
            self.sell(fair_price, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell
            self.sell(fair_price + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)


class KelpStrategy(MarketMaking):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.last_price = None

    def get_fair_price(self, order_depth: OrderDepth, traderObject) -> float:
        traderObject = traderObject

        # if there are orders on both side of the book
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())  # current market best ask
            best_bid = max(order_depth.buy_orders.keys())  # current market best bid

            # === filtering logic for bid and ask prices === #
            # to calculate a realiable mid price both
            # we filter out asks and bids which fall below the
            # adverse_volume, this removes noise from outliers
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= 15  # make this a PARAM later
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= 15
            ]

            # pick the best prices for ask/bid out of this filtered set
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            # if any filtered results were empty meaning the book is imbalanced/sparse
            if mm_ask is None or mm_bid is None:
                if traderObject.get("KELP_last_price", None) is None:  # if last price was also None
                    mmmid_price = (best_ask + best_bid) / 2  # use best ask and best bid for mid
                else:
                    mmmid_price = traderObject["KELP_last_price"]  # if not then use the last price
            else:
                mmmid_price = (mm_ask + mm_bid) / 2  # else we use the filtered data for mid price

            # if we have a valid last price
            if self.last_price is not None:
                last_returns = (mmmid_price - self.last_price) / self.last_price
                pred_returns = last_returns * -0.229
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            self.last_price = mmmid_price
            return fair

        return None

    def save(self) -> JSON:
        return {
            "last_price": getattr(self, "last_price", None),
            "window": list(self.window)
        }

    def load(self, data: JSON) -> None:
        self.last_price = data.get("last_price", None)
        self.window = deque(data.get("window", []))


class ResinStrategy(MarketMaking):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.last_price = None

    def get_fair_price(self, state: TradingState, traderObject) -> int:
        traderObject = traderObject
        return 10_000


class Trader:
    def __init__(self) -> None:
        limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
        }

        self.strategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "KELP": KelpStrategy,
            "RAINFOREST_RESIN": ResinStrategy
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if (symbol in state.order_depths and
                    len(state.order_depths[symbol].buy_orders) > 0 and
                    len(state.order_depths[symbol].sell_orders) > 0
            ):
                strategy_orders, strategy_conversions = strategy.run(state,
                                                                     traderObject=old_trader_data.get(symbol, {}), )

                orders[symbol] = strategy_orders
                conversions += sum(strategy_conversions)

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
