import json
from typing import Dict, List, Any, Tuple
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
        "RESIN_LADDER_BASE_OFFSET": 2,
        "RESIN_VOL_THRESHOLDS": [1.0, 2.0, 3.0],
        "RESIN_LADDER_DEPTHS": [2, 3, 4, 5],
        "RESIN_WINDOW": 10,
        "RESIN_WMA_WINDOW": 20,
        "KELP_VOL_CANCEL_THRESHOLD": 3.0,
        "KELP_PRICE_DEV_CANCEL": 2,
        "KELP_LADDER_LEVELS": 4,
        "KELP_TAKE_THRESHOLD": 1.0,
        "KELP_CLEAR_THRESHOLD": 45,
        "KELP_CLEAR_SIZE": 10,
        "EMA_FAST_ALPHA": 0.6,
        "EMA_SLOW_ALPHA": 0.3,
        "EMA_FAST_WINDOW": 3,
        "EMA_SLOW_WINDOW": 10,
        "CLEAR_SLOPE_DIVISOR": 1.0
    }

    def __init__(self):
        self.mid_price_history: Dict[Symbol, List[float]] = {}
        self.ema_fast: Dict[Symbol, float] = {}
        self.ema_slow: Dict[Symbol, float] = {}

    def update_ema(self, symbol: Symbol, price: float, alpha: float, which: str):
        if which == "fast":
            prev = self.ema_fast.get(symbol, price)
            self.ema_fast[symbol] = alpha * price + (1 - alpha) * prev
            return self.ema_fast[symbol]
        else:
            prev = self.ema_slow.get(symbol, price)
            self.ema_slow[symbol] = alpha * price + (1 - alpha) * prev
            return self.ema_slow[symbol]

    def weighted_moving_average(self, data: List[float]) -> float:
        weights = list(range(1, len(data) + 1))
        return sum(x * w for x, w in zip(data, weights)) / sum(weights)

    def get_ladder_depth(self, volatility: float) -> int:
        thresholds = self.PARAMS["RESIN_VOL_THRESHOLDS"]
        depths = self.PARAMS["RESIN_LADDER_DEPTHS"]
        for i, t in enumerate(thresholds):
            if volatility <= t:
                return depths[i]
        return depths[-1]

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""
        max_pos = self.PARAMS["MAX_POSITION"]

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            orders: List[Order] = []

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2

            # Update mid price history
            history = self.mid_price_history.get(product, [])
            history.append(mid_price)
            if len(history) > 50:
                history.pop(0)
            self.mid_price_history[product] = history

            # === RESIN Logic ===
            if product == "RAINFOREST_RESIN":
                fair_price = self.weighted_moving_average(history)
                volatility = np.std(history[-self.PARAMS["RESIN_WINDOW"]:])
                ladder_depth = self.get_ladder_depth(volatility)
                base_offset = self.PARAMS["RESIN_LADDER_BASE_OFFSET"]

                for level in range(1, ladder_depth + 1):
                    offset = base_offset + (level - 1)
                    size = max(1, int(10 * (1 - abs(position / max_pos))))
                    if position < max_pos:
                        orders.append(Order(product, int(fair_price - offset), size))
                    if position > -max_pos:
                        orders.append(Order(product, int(fair_price + offset), -size))

                if spread >= 2:
                    if position < max_pos:
                        orders.append(Order(product, int(fair_price - 1), 3))
                    if position > -max_pos:
                        orders.append(Order(product, int(fair_price + 1), -3))

            # === KELP Logic with Optimized EMA + Blended Fair Price ===
            elif product == "KELP":
                if len(history) < max(self.PARAMS["EMA_SLOW_WINDOW"], self.PARAMS["EMA_FAST_WINDOW"]):
                    result[product] = []
                    continue

                ema_fast = self.update_ema(product, mid_price, self.PARAMS["EMA_FAST_ALPHA"], "fast")
                ema_slow = self.update_ema(product, mid_price, self.PARAMS["EMA_SLOW_ALPHA"], "slow")

                # Blended fair price: more weight to fast for early movement
                fair_price = 0.6 * ema_fast + 0.4 * ema_slow
                slope = ema_fast - ema_slow
                volatility = np.std(history[-10:])
                momentum_score = slope / volatility if volatility > 0 else 0

                if volatility > self.PARAMS["KELP_VOL_CANCEL_THRESHOLD"] or abs(mid_price - fair_price) > self.PARAMS["KELP_PRICE_DEV_CANCEL"]:
                    result[product] = []
                    continue

                # Market Making
                levels = self.PARAMS["KELP_LADDER_LEVELS"]
                base_offset = 1 if spread <= 3 else 2
                size_multiplier = 1 + min(1.5, abs(momentum_score))

                for level in range(1, levels + 1):
                    offset = base_offset + level - 1
                    size = max(1, int(8 * size_multiplier * (1 - abs(position / max_pos))))
                    if position < max_pos:
                        orders.append(Order(product, int(fair_price - offset), size))
                    if position > -max_pos:
                        orders.append(Order(product, int(fair_price + offset), -size))

                # Market Taking
                if abs(momentum_score) > 1:
                    take_size = max(1, int(4 * size_multiplier * (1 - abs(position / max_pos))))
                    if position < max_pos and fair_price > best_ask + self.PARAMS["KELP_TAKE_THRESHOLD"]:
                        orders.append(Order(product, best_ask, take_size))
                    if position > -max_pos and fair_price < best_bid - self.PARAMS["KELP_TAKE_THRESHOLD"]:
                        orders.append(Order(product, best_bid, -take_size))

                # Smart Aggressive Clearing
                if abs(position) >= self.PARAMS["KELP_CLEAR_THRESHOLD"]:
                    clear_strength = min(1.0, abs(slope) / self.PARAMS["CLEAR_SLOPE_DIVISOR"])
                    clear_size = max(1, int(clear_strength * self.PARAMS["KELP_CLEAR_SIZE"]))
                    if position > 0 and slope < 0 and best_bid >= fair_price - 2:
                        orders.append(Order(product, best_bid, -clear_size))
                    elif position < 0 and slope > 0 and best_ask <= fair_price + 2:
                        orders.append(Order(product, best_ask, clear_size))

            result[product] = orders

        return result, conversions, trader_data