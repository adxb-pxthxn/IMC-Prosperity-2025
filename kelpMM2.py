import json
from typing import Dict, List, Any, Tuple
from datamodel import Order, OrderDepth, TradingState, Symbol, Listing, Trade, Observation, ProsperityEncoder
import numpy as np

ENABLE_LOGGING = True

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
    # === Configs ===
    ENABLE_LOGGING = True  # Flip this OFF for prod
    MAX_POSITION = 50
    BASE_ORDER_SIZE = 8
    EMA_ALPHA = 0.2
    LADDER_LEVELS = 3

    def __init__(self):
        self.ema_fair: Dict[Symbol, float] = {}
        self.mid_price_history: Dict[Symbol, List[float]] = {}

    # === Technicals ===

    def calculate_ema(self, product: str, mid_price: float) -> float:
        prev = self.ema_fair.get(product, mid_price)
        ema = self.EMA_ALPHA * mid_price + (1 - self.EMA_ALPHA) * prev
        self.ema_fair[product] = ema
        return ema

    def calculate_slope(self, history: List[float]) -> float:
        if len(history) < 5:
            return 0.0
        return (history[-1] - history[-5]) / 4

    def calculate_imbalance(self, order_depth: OrderDepth) -> float:
        bid_vol = sum(abs(v) for v in order_depth.buy_orders.values())
        ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def get_signal_strength(self, slope: float, imbalance: float) -> float:
        # Combines slope + imbalance into one number
        return slope * 100 + imbalance * 50

    # === Trade Signal Logic ===

    def should_market_take(self, mid_prices: List[float], slope: float, imbalance: float, direction: str) -> bool:
        if len(mid_prices) < 4:
            return False

        # Only need 2 of last 3 ticks going in the same direction
        recent_deltas = [mid_prices[-i] - mid_prices[-i - 1] for i in range(1, 4)]
        direction_ok = sum(
            (delta > 0 if direction == "up" else delta < 0) for delta in recent_deltas
        ) >= 2

        if direction == "up":
            return direction_ok and slope > 0.1 and imbalance > 0.1
        elif direction == "down":
            return direction_ok and slope < -0.1 and imbalance < -0.1
        return False

    # === Passive Quote Builder with Directional Skew ===

    def directional_passives(
        self, product: str, fair: float, position: int, signal_strength: float, slope: float, imbalance: float
    ) -> List[Order]:
        orders = []
        # Stronger signal = quotes closer to mid
        skew = int(signal_strength / 10)

        for level in range(1, self.LADDER_LEVELS + 1):
            size = max(1, int(self.BASE_ORDER_SIZE * (1 - abs(position) / self.MAX_POSITION)))

            # Start with basic ladder around fair
            bid_price = int(fair) - level
            ask_price = int(fair) + level

            # Push bids closer in bullish trend, asks closer in bearish
            if slope > 0 and imbalance > 0:
                bid_price += skew
            if slope < 0 and imbalance < 0:
                ask_price -= skew

            if position < self.MAX_POSITION:
                orders.append(Order(product, bid_price, size))
            if position > -self.MAX_POSITION:
                orders.append(Order(product, ask_price, -size))

        return orders

    # === Main Run ===

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0
        trader_data = ""

        for product, order_depth in state.order_depths.items():
            if product != "KELP":
                continue

            position = state.position.get(product, 0)
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2

            # Track EMA + history
            ema = self.calculate_ema(product, mid_price)
            self.mid_price_history.setdefault(product, []).append(mid_price)
            history = self.mid_price_history[product]

            slope = self.calculate_slope(history)
            imbalance = self.calculate_imbalance(order_depth)
            signal_strength = self.get_signal_strength(slope, imbalance)

            # Decide if we go for a taker or passive quoting
            orders = []
            trade_size = max(1, int(self.BASE_ORDER_SIZE * (1 - abs(position) / self.MAX_POSITION)))
            action = "PASSIVE"

            take_long = self.should_market_take(history, slope, imbalance, "up")
            take_short = self.should_market_take(history, slope, imbalance, "down")

            if take_long and position < self.MAX_POSITION:
                orders.append(Order(product, best_ask + 1, trade_size))  # market buy
                action = "TAKE_LONG"
            elif take_short and position > -self.MAX_POSITION:
                orders.append(Order(product, best_bid - 1, -trade_size))  # market sell
                action = "TAKE_SHORT"
            else:
                orders.extend(
                    self.directional_passives(product, ema, position, signal_strength, slope, imbalance)
                )

            result[product] = orders

            # === Logging: only if enabled ===
            if self.ENABLE_LOGGING:
                strength_label = (
                    "HIGH" if abs(signal_strength) > 12 else
                    "MEDIUM" if abs(signal_strength) > 6 else
                    "LOW"
                )
                logger.print(
                    f"[{state.timestamp}] | mid={mid_price:.2f} ema={ema:.2f} slope={slope:.3f} "
                    f"imb={imbalance:.2f} pos={position} signal={strength_label} action={action} "
                    f"pnl={state.observations.plainValueObservations.get(product, 0)}"
                )

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
