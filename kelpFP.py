import json
from typing import Dict, List, Any
from datamodel import Order, OrderDepth, TradingState, Symbol, Listing, Trade, Observation, ProsperityEncoder
import numpy as np

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

class Trader:
    PARAMS = {
        "MAX_POSITION": 50,
        "FAIR_WMA_WINDOW": 12,
        "FAIR_BOOK_WEIGHT": 0.4,
        "FAIR_POSITION_BIAS": 0.01,
        "ZSCORE_WINDOW": 15,
        "ZSCORE_THRESHOLD": 0.5,
        "STD_SMOOTHING": 0.3,
        "LADDER_LEVELS": 4,
        "BASE_ORDER_SIZE": 10,
    }

    def __init__(self):
        self.mid_history: Dict[Symbol, List[float]] = {}
        self.std_cache: Dict[Symbol, float] = {}
        self.fair_cache: Dict[Symbol, float] = {}

    def weighted_moving_average(self, values: List[float]) -> float:
        if not values:
            return 0
        weights = np.arange(1, len(values) + 1)
        return np.dot(values, weights) / weights.sum()

    def compute_fair_price(self, product: Symbol, best_bid: int, best_ask: int, bid_vol: int, ask_vol: int, position: int) -> float:
        mid = (best_bid + best_ask) / 2
        book_mid = (best_bid * bid_vol + best_ask * ask_vol) / (bid_vol + ask_vol)

        history = self.mid_history.get(product, [])
        history.append(mid)
        if len(history) > 50:
            history.pop(0)
        self.mid_history[product] = history

        wma = self.weighted_moving_average(history[-self.PARAMS["FAIR_WMA_WINDOW"]:])
        fair = (
            self.PARAMS["FAIR_BOOK_WEIGHT"] * book_mid +
            (1 - self.PARAMS["FAIR_BOOK_WEIGHT"]) * wma
        )

        bias = -position * self.PARAMS["FAIR_POSITION_BIAS"]
        fair += bias

        self.fair_cache[product] = fair
        return fair

    def compute_zscore(self, product: Symbol, mid: float) -> float:
        history = self.mid_history[product]
        if len(history) < self.PARAMS["ZSCORE_WINDOW"]:
            return 0.0

        recent = np.array(history[-self.PARAMS["ZSCORE_WINDOW"]:])
        mean = np.mean(recent)
        std = np.std(recent)

        # Smooth std
        last_std = self.std_cache.get(product, std)
        smoothed_std = self.PARAMS["STD_SMOOTHING"] * std + (1 - self.PARAMS["STD_SMOOTHING"]) * last_std
        self.std_cache[product] = smoothed_std

        if smoothed_std < 1e-6:
            return 0.0

        return (mid - mean) / smoothed_std

    def mm_resin_mean_reversion(self, product: Symbol, order_depth: OrderDepth, position: int) -> List[Order]:
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_vol = abs(order_depth.buy_orders[best_bid])
        ask_vol = abs(order_depth.sell_orders[best_ask])
        mid = (best_bid + best_ask) / 2

        fair = self.compute_fair_price(product, best_bid, best_ask, bid_vol, ask_vol, position)
        z = self.compute_zscore(product, mid)

        orders = []
        max_pos = self.PARAMS["MAX_POSITION"]

        if abs(z) < self.PARAMS["ZSCORE_THRESHOLD"]:
            return orders  # No signal

        for level in range(1, self.PARAMS["LADDER_LEVELS"] + 1):
            offset = level
            confidence = min(abs(z), 3)
            size = max(1, int(self.PARAMS["BASE_ORDER_SIZE"] * (1 - level / (self.PARAMS["LADDER_LEVELS"] + 1)) * (confidence / 3)))

            if z < -self.PARAMS["ZSCORE_THRESHOLD"] and position < max_pos:
                orders.append(Order(product, int(fair - offset), size))
            if z > self.PARAMS["ZSCORE_THRESHOLD"] and position > -max_pos:
                orders.append(Order(product, int(fair + offset), -size))

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0
        trader_data = ""

        for product, order_depth in state.order_depths.items():
            if product != "RAINFOREST_RESIN":
                result[product] = []
                continue

            position = state.position.get(product, 0)
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            result[product] = self.mm_resin_mean_reversion(product, order_depth, position)

        return result, conversions, trader_data


