from typing import Dict, List, Any, Tuple
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
    # hybrid strategy constants
    MAX_POSITION = 50
    BASE_ORDER_SIZE = 6
    ATR_PERIOD = 20
    SLOPE_PERIOD = 5
    SLOPE_THRESHOLD = 0.5  # price ticks per tick
    EMA_ALPHA = 0.2

    def __init__(self):
        self.price_history: List[float] = []
        self.ema = None

    def calculate_ema(self, price: float) -> float:
        if self.ema is None:
            self.ema = price
        else:
            self.ema = self.EMA_ALPHA * price + (1 - self.EMA_ALPHA) * self.ema
        return self.ema

    def calculate_slope(self) -> float:
        if len(self.price_history) < self.SLOPE_PERIOD + 1:
            return 0.0
        return (self.price_history[-1] - self.price_history[-self.SLOPE_PERIOD - 1]) / self.SLOPE_PERIOD

    def calculate_atr(self) -> float:
        if len(self.price_history) < self.ATR_PERIOD:
            return 1.0
        recent = self.price_history[-self.ATR_PERIOD:]
        return max(recent) - min(recent)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        product = "KELP"
        order_depth = state.order_depths[product]
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2

        self.price_history.append(mid_price)
        ema = self.calculate_ema(mid_price)
        slope = self.calculate_slope()
        atr = self.calculate_atr()

        orders = []

        # size scales with volatility
        size = max(1, int(self.BASE_ORDER_SIZE / atr))

        # skew quoting depending on slope
        if slope > self.SLOPE_THRESHOLD:
            # price trending up → favor buys
            bid_price = int(ema)
            ask_price = int(ema + 2)  # further out to avoid getting hit
            if state.position.get(product, 0) < self.MAX_POSITION:
                orders.append(Order(product, bid_price, size))

        elif slope < -self.SLOPE_THRESHOLD:
            # price trending down → favor sells
            bid_price = int(ema - 2)  # pull back bid
            ask_price = int(ema)
            if state.position.get(product, 0) > -self.MAX_POSITION:
                orders.append(Order(product, ask_price, -size))

        else:
            # flat market → quote both sides normally
            bid_price = int(ema - 1)
            ask_price = int(ema + 1)
            pos = state.position.get(product, 0)
            if pos < self.MAX_POSITION:
                orders.append(Order(product, bid_price, size))
            if pos > -self.MAX_POSITION:
                orders.append(Order(product, ask_price, -size))

        result[product] = orders
        return result, conversions, ""
### logger.flush(state, result, conversions, trader_data) ###
### with open("kelp_trade_log.txt", "a") as f:
### f.write(trader_data)
