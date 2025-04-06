import json
from typing import Dict, List, Any, Tuple
from datamodel import Order, OrderDepth, TradingState, Symbol, Listing, Trade, Observation, ProsperityEncoder
import numpy as np
import math

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
        "BOOK_WEIGHT_EMA_ALPHA": 0.3,
        "POSITION_BIAS_COEFF": 0.01,
        "KELP_LADDER_DEPTH": 6,
        "KELP_BASE_SIZE": 8,
        "KELP_FILLER_SIZE": 2,
        "KELP_FILLER_SPREAD": 3,
        "RESIN_LADDER_DEPTH": 8,
        "RESIN_BASE_SIZE": 10,
        "RESIN_FILLER_SIZE": 2,
        "RESIN_FILLER_SPREAD": 4,
        "REGRESSION_WINDOW": 8,
        "MAX_REGRESSION_SHIFT": 1.5,
    }

    def __init__(self):
        self.price_history: Dict[Symbol, List[float]] = {}
        self.book_fair: Dict[Symbol, float] = {}

    def get_fair_price(self, symbol: Symbol, best_bid: int, best_ask: int, bid_vol: int, ask_vol: int) -> float:
        mid = (best_bid + best_ask) / 2
        book_based = (best_bid * bid_vol + best_ask * ask_vol) / (bid_vol + ask_vol)
        prev = self.book_fair.get(symbol, mid)
        smoothed = self.PARAMS["BOOK_WEIGHT_EMA_ALPHA"] * book_based + (1 - self.PARAMS["BOOK_WEIGHT_EMA_ALPHA"]) * prev
        self.book_fair[symbol] = smoothed

        history = self.price_history.get(symbol, [])
        history.append(mid)
        if len(history) > 50:
            history.pop(0)
        self.price_history[symbol] = history

        ema = np.mean(history[-10:]) if len(history) >= 10 else mid
        fair = 0.4 * smoothed + 0.3 * mid + 0.3 * ema
        return fair

    def get_position_bias(self, position: int) -> float:
        max_pos = self.PARAMS["MAX_POSITION"]
        return -position * self.PARAMS["POSITION_BIAS_COEFF"] * (1 + abs(position / max_pos))

    def regression_fair_price(self, history: List[float]) -> float:
        N = self.PARAMS["REGRESSION_WINDOW"]
        if len(history) < N:
            return 0.0
        y = history[-N:]
        x = list(range(N))
        coef = np.polyfit(x, y, deg=1)
        slope = coef[0]
        return max(-self.PARAMS["MAX_REGRESSION_SHIFT"], min(slope, self.PARAMS["MAX_REGRESSION_SHIFT"]))

    def taker_logic(self, product: str, fair: float, best_bid: int, best_ask: int,
                    spread: int, position: int, history: List[float],
                    order_depth: OrderDepth) -> List[Order]:
        orders = []
        max_pos = self.PARAMS["MAX_POSITION"]

        # Reversion
        if best_bid < fair - 2 and position < max_pos:
            orders.append(Order(product, best_bid, 3))
        if best_ask > fair + 2 and position > -max_pos:
            orders.append(Order(product, best_ask, -3))

        # Momentum
        if len(history) >= 4:
            deltas = [history[-i] - history[-i-1] for i in range(1, 4)]
            slope = sum(deltas)
            if all(d > 0 for d in deltas) and fair > best_ask and position < max_pos:
                size = int(2 + min(3, abs(slope) * 10))
                orders.append(Order(product, best_ask + 1, size))
            elif all(d < 0 for d in deltas) and fair < best_bid and position > -max_pos:
                size = int(2 + min(3, abs(slope) * 10))
                orders.append(Order(product, best_bid - 1, -size))

        # Imbalance
        bid_volume = sum(abs(v) for v in order_depth.buy_orders.values())
        ask_volume = sum(abs(v) for v in order_depth.sell_orders.values())
        if bid_volume > 3 * ask_volume and position < max_pos:
            strength = math.log(bid_volume / max(1, ask_volume))
            size = int(2 + min(3, strength))
            orders.append(Order(product, best_ask, size))
        elif ask_volume > 3 * bid_volume and position > -max_pos:
            strength = math.log(ask_volume / max(1, bid_volume))
            size = int(2 + min(3, strength))
            orders.append(Order(product, best_bid, -size))

        return orders

    def mm_kelp(self, product: str, order_depth: OrderDepth, position: int) -> List[Order]:
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_vol = abs(order_depth.buy_orders[best_bid])
        ask_vol = abs(order_depth.sell_orders[best_ask])
        spread = best_ask - best_bid

        fair = self.get_fair_price(product, best_bid, best_ask, bid_vol, ask_vol)
        fair += self.get_position_bias(position)
        history = self.price_history[product]

        orders = []
        for level in range(1, self.PARAMS["KELP_LADDER_DEPTH"] + 1):
            offset = level
            size = max(1, int(self.PARAMS["KELP_BASE_SIZE"] * (1 - abs(position / self.PARAMS["MAX_POSITION"])) * (1 - level / (self.PARAMS["KELP_LADDER_DEPTH"] + 1))))
            if position < self.PARAMS["MAX_POSITION"]:
                orders.append(Order(product, int(fair - offset), size))
            if position > -self.PARAMS["MAX_POSITION"]:
                orders.append(Order(product, int(fair + offset), -size))

        if spread >= self.PARAMS["KELP_FILLER_SPREAD"]:
            if position < self.PARAMS["MAX_POSITION"]:
                orders.append(Order(product, best_bid + 1, self.PARAMS["KELP_FILLER_SIZE"]))
            if position > -self.PARAMS["MAX_POSITION"]:
                orders.append(Order(product, best_ask - 1, -self.PARAMS["KELP_FILLER_SIZE"]))

        orders += self.taker_logic(product, fair, best_bid, best_ask, spread, position, history, order_depth)
        return orders

    def mm_resin(self, product: str, order_depth: OrderDepth, position: int) -> List[Order]:
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_vol = abs(order_depth.buy_orders[best_bid])
        ask_vol = abs(order_depth.sell_orders[best_ask])
        spread = best_ask - best_bid

        fair = self.get_fair_price(product, best_bid, best_ask, bid_vol, ask_vol)
        fair += self.get_position_bias(position)

        history = self.price_history[product]
        regression_shift = self.regression_fair_price(history)
        fair += regression_shift

        orders = []
        for level in range(1, self.PARAMS["RESIN_LADDER_DEPTH"] + 1):
            offset = level + 1
            size = max(1, int(self.PARAMS["RESIN_BASE_SIZE"] * (1 - abs(position / self.PARAMS["MAX_POSITION"])) * (1 - level / (self.PARAMS["RESIN_LADDER_DEPTH"] + 1))))
            if position < self.PARAMS["MAX_POSITION"]:
                orders.append(Order(product, int(fair - offset), size))
            if position > -self.PARAMS["MAX_POSITION"]:
                orders.append(Order(product, int(fair + offset), -size))

        if spread >= self.PARAMS["RESIN_FILLER_SPREAD"]:
            if position < self.PARAMS["MAX_POSITION"]:
                orders.append(Order(product, best_bid + 1, self.PARAMS["RESIN_FILLER_SIZE"]))
            if position > -self.PARAMS["MAX_POSITION"]:
                orders.append(Order(product, best_ask - 1, -self.PARAMS["RESIN_FILLER_SIZE"]))

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            if product == "KELP":
                result[product] = self.mm_kelp(product, order_depth, position)
            elif product == "RAINFOREST_RESIN":
                result[product] = self.mm_resin(product, order_depth, position)
            else:
                result[product] = []

        return result, conversions, trader_data