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
    def __init__(self):
        self.mid_price_history = {}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""
        max_pos = 50

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            orders: List[Order] = []

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2

            # === Update mid-price history ===
            history = self.mid_price_history.get(product, [])
            history.append(mid_price)
            if len(history) > 15:
                history.pop(0)
            self.mid_price_history[product] = history

            # === RAINFOREST_RESIN: Mean-Reversion Laddering ===
            if product == "RAINFOREST_RESIN":
                fair_price = sum(history) / len(history)
                for level in range(1, 4):
                    offset = 2 + (level - 1)
                    size = max(1, int(10 * (1 - abs(position / max_pos))))
                    if position < max_pos:
                        orders.append(Order(product, int(fair_price - offset), size))
                    if position > -max_pos:
                        orders.append(Order(product, int(fair_price + offset), -size))

            # === KELP: Directional Market Making ===
            elif product == "KELP":
                fair_price = sum(history) / len(history)
                momentum = 0
                if len(history) >= 5:
                    momentum = mid_price - sum(history[:-1]) / len(history[:-1])

                # Adjust quoting offsets based on momentum
                base_offset = 2
                if momentum > 0.5:
                    bid_offset = base_offset + 1
                    ask_offset = base_offset - 1
                elif momentum < -0.5:
                    bid_offset = base_offset - 1
                    ask_offset = base_offset + 1
                else:
                    bid_offset = ask_offset = base_offset

                # Position-aware size
                size = max(1, int(10 * (1 - abs(position / max_pos))))

                # Place skewed passive orders
                if position < max_pos:
                    orders.append(Order(product, int(fair_price - bid_offset), size))
                if position > -max_pos:
                    orders.append(Order(product, int(fair_price + ask_offset), -size))

            result[product] = orders

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
