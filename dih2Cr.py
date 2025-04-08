from typing import Dict, List,Any
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import math
import numpy as np


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

class Trader:
    def __init__(self):
        self.mid_price_history = {}
        self.last_price = {}
        self.macd_history = {}

    # === RESIN Mean-Reversion Ladder (unchanged) ===
    def track_mid_price(self, product: str, mid_price: float, window: int = 20) -> float:
        history = self.mid_price_history.get(product, [])
        history.append(mid_price)
        if len(history) > window:
            history.pop(0)
        self.mid_price_history[product] = history
        return sum(history) / len(history)

    def adjust_order_size(self, position: int, max_position: int = 50, base_size: int = 10) -> int:
        return max(1, int(base_size * (1 - abs(position / max_position))))

    def place_resin_ladder_orders(self, product: str, fair_price: float, position: int) -> List[Order]:
        orders = []
        for level in range(1, 4):  # 3 levels
            offset = 2 + (level - 1) # This is beautiful
            size = self.adjust_order_size(position, base_size=10)
            if position < 50:
                orders.append(Order(product, int(fair_price - offset), size))
            if position > -50:
                orders.append(Order(product, int(fair_price + offset), -size))
        return orders

    # === KELP Liquidity Sniper ===
    def kelp_liquidity_sniper(self, product: str, order_depth: OrderDepth, position: int) -> List[Order]:
        orders = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        spread = best_ask - best_bid

        if spread >= 4:
            # inside spread levels
            buy_price = best_bid + 1
            sell_price = best_ask - 1

            base_size = 10
            if position < 50:
                orders.append(Order(product, buy_price, base_size))
            if position > -50:
                orders.append(Order(product, sell_price, -base_size))

        return orders
    
    def calculate_ema(self, values: List[float], window: int) -> float:
        if not values or len(values) < window:
            return sum(values) / len(values) if values else 0

        k = 2 / (window + 1)
        ema = values[0]
        for price in values[1:]:
            ema = price * k + ema * (1 - k)
        return ema
    
    def calculate_macd(self, product: str, short_window=6, long_window=13, signal_window=5):
        prices = self.mid_price_history.get(product, [])
        if len(prices) < long_window: 
            return None, None # if there isn't enough data yet for long window calculation
        
        short_ema = self.calculate_ema(prices, short_window)
        long_ema = self.calculate_ema(prices, long_window)

        macd = short_ema - long_ema

        macd_hist = self.macd_history.get(product, [])
        macd_hist.append(macd)

        if len(macd_hist) > signal_window:
            macd_hist.pop(0)
        self.macd_history[product] = macd_hist

        signal_line = self.calculate_ema(macd_hist, signal_window)
        return macd, signal_line

    
    # Calculating squid ink momentum
    def squid_ink_momentum(self, product: str, order_depth: OrderDepth, position: int) -> List[Order]:
        
        macd, signal = self.calculate_macd(product)
        if macd is None or signal is None:
            return []
        orders = []

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)

        base_size = 3 # Currently arbitrarily chosen.

        threshold = 0.1  # Experiment with this

        # Detect momentum shift
        if len(self.macd_history[product]) >= 2:
            prev_macd = self.macd_history[product][-2]
            prev_signal = self.calculate_ema(self.macd_history[product][:-1], 7)

            bullish_cross = prev_macd < prev_signal and macd > signal
            bearish_cross = prev_macd > prev_signal and macd < signal

            if bullish_cross and position < 50:
                orders.append(Order(product, best_ask, base_size))

            elif bearish_cross and position > -50:
                orders.append(Order(product, best_bid, base_size))
        # These exist otherwise
        elif macd - signal > threshold and position < 50:
            orders.append(Order(product, best_ask, base_size))
        elif signal - macd > threshold and position > -50:
            orders.append(Order(product, best_bid, base_size))
        
        return orders

    

    # === Main Bot ===
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths:
            if product != "SQUID_INK":
                continue
            order_depth = state.order_depths[product]
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2
            position = state.position.get(product, 0)

            if product == "RAINFOREST_RESIN":
                fair_price = self.track_mid_price(product, mid_price, window=20)
                orders = self.place_resin_ladder_orders(product, fair_price, position)

            elif product == "KELP":
                orders = self.kelp_liquidity_sniper(product, order_depth, position)
            if product == "SQUID_INK":
                self.track_mid_price(product, mid_price, window=13)
                orders = self.squid_ink_momentum(product, order_depth, position)


            result[product] = orders

        traderData = "KELP_Sniper_RESIN_Ladder_INK_Sniper"
        conversions = 1

        logger.flush(state, result, conversions, traderData) #this is necessary for visualiser
        return result, conversions, traderData
