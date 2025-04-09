from typing import Dict, List, Any
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
        self.mid_price_history = {}

    def rain_order(self, order_depth: OrderDepth, fair=10000, position=0, position_limit=50):

        orders = []

        buy_volume = 0
        sell_volume = 0

        sell_orders = order_depth.sell_orders
        buy_orders = order_depth.buy_orders
        try:
            best_ask_fair = min([p for p in sell_orders.keys() if p > fair + 1], default=fair + 2)
        except ValueError:
            best_ask_fair = fair + 1

        try:
            best_bid_fair = max([p for p in buy_orders.keys() if p < fair + 1], default=fair - 2)
        except ValueError:
            best_bid_fair = fair - 1

        if sell_orders:
            best_ask = min(sell_orders.keys())
            best_ask_ammount = -sell_orders[best_ask]
            if best_ask < fair:
                quant = min(best_ask_ammount, position_limit - position)
                if quant > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quant))
                    buy_volume += quant
        if buy_orders:
            best_bid = max(buy_orders.keys())
            best_bid_amount = buy_orders[best_bid]
            if best_bid > fair:
                quant = min(best_bid_amount, position_limit + position)
                if quant > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -quant))
                    sell_volume += quant

        buy_quant = position_limit - (position + buy_volume)
        if buy_quant > 0:
            orders.append(Order("RAINFOREST_RESIN", best_bid_fair + 1, buy_quant))

        sell_quant = position_limit + (position - sell_volume)
        if sell_quant > 0:
            orders.append(Order("RAINFOREST_RESIN", best_ask_fair - 1, -sell_quant))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        if 'RAINFOREST_RESIN' in state.order_depths:
            rain_position = state.position.get('RAINFOREST_RESIN', 0)
            rain_orders = self.rain_order(state.order_depths['RAINFOREST_RESIN'], position=rain_position)
            result['RAINFOREST_RESIN'] = rain_orders

        # if 'KELP' in state.order_depths:

        #     kelp_position = state.position.get('RAINFOREST_RESIN',0)
        #     fair_price = self.track_mid_price('KELP', ((max(state.order_depths.buy_orders.keys())+min(state.order_depths.sell_orders.keys()))/2), window=10)
        #     rain_orders=self.kelp_order(state.order_depths['RAINFOREST_RESIN'],fair_priceposition=kelp_position)
        #     result['KELP']=rain_orders

        traderData = "MM_MeanReversion_Ladder"
        conversions = 1

        logger.flush(state, result, conversions, traderData)  # this is necessary for visualiser
        return result, conversions, traderData
