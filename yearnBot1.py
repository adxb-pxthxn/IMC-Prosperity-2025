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
        "RESIN_LADDER_BASE_OFFSET": 2,
        "RESIN_VOL_THRESHOLDS": [1.0, 2.0, 3.0],
        "RESIN_LADDER_DEPTHS": [2, 3, 4, 5],
        "RESIN_WINDOW": 10,
        "RESIN_WMA_WINDOW": 20,
        "KELP_VOL_CANCEL_THRESHOLD": 3.0,
        "KELP_PRICE_DEV_CANCEL": 2,
        "KELP_LADDER_LEVELS": 4,
        "KELP_TAKE_THRESHOLD": 1.0,
        "KELP_CLEAR_THRESHOLD": 40,
        "KELP_CLEAR_SIZE": 10,
        "RESIN_TIGHT_MM_OFFSET": 1,
        "RESIN_TIGHT_MM_SIZE": 3,
    }

    def __init__(self):
        self.mid_price_history = {}

    def weighted_moving_average(self, data: list[float]) -> float:
        weights = list(range(1, len(data) + 1))
        return sum(x * w for x, w in zip(data, weights)) / sum(weights)

    def get_ladder_depth(self, volatility: float) -> int:
        thresholds = self.PARAMS["RESIN_VOL_THRESHOLDS"]
        depths = self.PARAMS["RESIN_LADDER_DEPTHS"]
        for i, t in enumerate(thresholds):
            if volatility <= t:
                return depths[i]
        return depths[-1]

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""
        max_pos = self.PARAMS["MAX_POSITION"]

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            orders: list[Order] = []

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2

            history = self.mid_price_history.get(product, [])
            history.append(mid_price)
            if len(history) > 20:
                history.pop(0)
            self.mid_price_history[product] = history

            if product == "RAINFOREST_RESIN":
                fair_price = self.weighted_moving_average(history)
                volatility = np.std(history[-self.PARAMS["RESIN_WINDOW"]:])
                ladder_depth = self.get_ladder_depth(volatility)
                base_offset = self.PARAMS["RESIN_LADDER_BASE_OFFSET"]

                # Ladder orders
                for level in range(1, ladder_depth + 1):
                    offset = base_offset + (level - 1)
                    size = max(1, int(10 * (1 - abs(position / max_pos))))
                    if position < max_pos:
                        orders.append(Order(product, int(fair_price - offset), size))
                    if position > -max_pos:
                        orders.append(Order(product, int(fair_price + offset), -size))

                # Micro market making at ±1 offset
                tight_offset = self.PARAMS["RESIN_TIGHT_MM_OFFSET"]
                tight_size = self.PARAMS["RESIN_TIGHT_MM_SIZE"]
                if spread >= 2:
                    if position < max_pos:
                        orders.append(Order(product, int(fair_price - tight_offset), tight_size))
                    if position > -max_pos:
                        orders.append(Order(product, int(fair_price + tight_offset), -tight_size))

            elif product == "KELP":
                if len(history) < 6:
                    result[product] = []
                    continue

                wma_now = self.weighted_moving_average(history[-6:])
                wma_prev = self.weighted_moving_average(history[-7:-1])
                predicted_price = wma_now + (wma_now - wma_prev)
                slope = wma_now - wma_prev

                volatility = np.std(history[-10:])
                if volatility > self.PARAMS["KELP_VOL_CANCEL_THRESHOLD"] or abs(mid_price - predicted_price) > self.PARAMS["KELP_PRICE_DEV_CANCEL"]:
                    result[product] = []
                    continue

                levels = self.PARAMS["KELP_LADDER_LEVELS"]
                base_offset = 1 if spread <= 3 else 2
                for level in range(1, levels + 1):
                    offset = base_offset + level - 1
                    size = max(1, int(10 * (1 - abs(position / max_pos))))
                    if position < max_pos:
                        orders.append(Order(product, int(predicted_price - offset), size))
                    if position > -max_pos:
                        orders.append(Order(product, int(predicted_price + offset), -size))

                take_thresh = self.PARAMS["KELP_TAKE_THRESHOLD"]
                take_size = max(1, int(5 * (1 - abs(position / max_pos))))
                if position < max_pos and predicted_price > best_ask + take_thresh:
                    orders.append(Order(product, best_ask, take_size))
                if position > -max_pos and predicted_price < best_bid - take_thresh:
                    orders.append(Order(product, best_bid, -take_size))

                if position >= self.PARAMS["KELP_CLEAR_THRESHOLD"] and slope < 0:
                    orders.append(Order(product, best_bid, -self.PARAMS["KELP_CLEAR_SIZE"]))
                elif position <= -self.PARAMS["KELP_CLEAR_THRESHOLD"] and slope > 0:
                    orders.append(Order(product, best_ask, self.PARAMS["KELP_CLEAR_SIZE"]))

            result[product] = orders

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data