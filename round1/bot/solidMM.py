from typing import Dict, List, Any, Tuple
import json

import jsonpickle

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
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"


# ===================================================================================================================
# Strategies used:
# Market Making: For both resin and kelp we passively market make around a fair price. For resin this can be fixed to
# 10000. For kelp the fair price is calculated every tick using the KELP_fair_value function.
#
# Market Taking: Market taking is an approach which aggresively hits the market either buying or selling when there is
# favourable prices available.
#
# Position Clearing; This pairs well with market taking, this layer reduces or clears our current position held if it
# is risky.
# ===================================================================================================================

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,  # default spread for MM
        "clear_width": 0,
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within 2 ticks joining instead of pennying against
        "default_edge": 4,  # if no options to join post quote 4 ticks
        "soft_position_limit": 30,  # soft position limit, pretty much skews inventory to follow this
        "TRADE": False  # flag to trade or not, for iso testing strats for each asset
    },
    Product.KELP: {
        "take_width": 1,  # default spread for MM
        "clear_width": 0,  # how close the BP to our FP to clear, 0 means it must be fair
        "prevent_adverse": True,  # flag to prevent trades when there is too much depth at best price
        "adverse_volume": 15,  # this is the volume threshold to toggle prevent_adverse
        "reversion_beta": -0.229,  # mean reversion coefficient pretty much our confidence on it dropping to mean
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 1,  # Joins when someone is 1 tick from FP, keeps us at the top of the book
        "default_edge": 1,  # When no chances for joining and pennying just quote 1 tick from FP
        "TRADE": True  # flag to trade or not, for iso testing strats for each asset

    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RAINFOREST_RESIN: 20, Product.KELP: 20}  # self-imposed position limit

    # === function to get optimal orders === #
    def take_best_orders(
            self,
            product: str,
            fair_value: int,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:  # if order book has sell orders
            best_ask = min(order_depth.sell_orders.keys())  # best ask is the lowest sell order
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]  # check order book depth for the best ask

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:  # if prevent_adverse is off
                if best_ask <= fair_value - take_width:  # if best ask is within take width
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))  # buy
                        buy_order_volume += quantity  # update the buy order volume
                        order_depth.sell_orders[best_ask] += quantity  # add the quantity to sell orders
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:  # if order book has buy orders
            best_bid = max(order_depth.buy_orders.keys())  # best bid is the highest buy order
            best_bid_amount = order_depth.buy_orders[best_bid]  # best volume is the full depth of the best bid

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:  # if prevent_adverse is off
                if best_bid >= fair_value + take_width:  # if best bid is within take width
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))  # sell
                        sell_order_volume += quantity  # update the sell order volume
                        order_depth.buy_orders[best_bid] -= quantity  # take the quantity from buy orders
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    # === function for calculating prices to place orders at === #
    def make_orders(
            self,
            product,
            order_depth: OrderDepth,
            fair_value: float,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            disregard_edge: float,  # disregard trades within this edge for pennying or joining
            join_edge: float,  # join trades within this edge
            default_edge: float,  # default edge to request if there are no levels to penny or join
            manage_position: bool = False,
            soft_position_limit: int = 0,
            # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        # list of current sell orders higher than fair
        # here the disregard edge is used to treat
        # orders within the set edge as essentially fair
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        # list of current buy orders lower than fair
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        # take best of the best if they are present
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)  # place buy orders fixed edge above fair
        if best_ask_above_fair is not None:  # if other orders are above fair
            if abs(best_ask_above_fair - fair_value) <= join_edge:  # if within join
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)  # place sell orders fixed edge below fair
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:  # if within join
                bid = best_bid_below_fair # join
            else:
                bid = best_bid_below_fair + 1 # penny

        # inventory management flag to keep volume close to soft limit
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        # call market make
        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    # === function for placing MM orders === #
    def market_make(
            self,
            product: str,
            orders: List[Order],
            bid: int,
            ask: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)  # how many orders can be placed
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    # === function for clearing a position === #
    def clear_position_order(
            self,
            product: str,
            fair_value: float,
            width: int,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> tuple[int, int]:
        # predict the position if all the orders placed during the take go through
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)  # calculate the fair exit price
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)  # calculate quantities to clear
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        # if our current position is long
        if position_after_take > 0:
            # aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)  # dont sell more than you have
            sent_quantity = min(sell_quantity, clear_quantity)  # decide which quantity to sell
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))  # sell
                sell_order_volume += abs(sent_quantity)  # log volume change

        # if our current position is short
        if position_after_take < 0:
            # aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))  # dont buy more than needed to cover short
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))  # buy
                buy_order_volume += abs(sent_quantity)  # log volume change

        return buy_order_volume, sell_order_volume

    def take_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            take_width: float,
            position: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    # === function for clearing orders === #
    def clear_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            clear_width: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        # clears both sides of the order book
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    # === function for calculating Kelp fair price === #
    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
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
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.KELP]["adverse_volume"]
            ]

            # pick the best prices for ask/bid out of this filtered set
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            # if any filtered results were empty meaning the book is imbalanced/sparse
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:  # if last price was also None
                    mmmid_price = (best_ask + best_bid) / 2  # use best ask and best bid for mid
                else:
                    mmmid_price = traderObject["KELP_last_price"]  # if not then use the last price
            else:
                mmmid_price = (mm_ask + mm_bid) / 2  # else we use the filtered data for mid price

            # if we have a valid last price
            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price  # calculate the percentage change
                pred_returns = (
                        last_returns * self.params[Product.KELP]["reversion_beta"]  # predict next return
                )
                fair = mmmid_price + (mmmid_price * pred_returns)  # new fair price is calculated
            else:
                fair = mmmid_price  # else we just use mid-price as the fair price
            traderObject["KELP_last_price"] = mmmid_price  # update last price
            return fair
        return None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            RR_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )

            # get best orders for resin
            RR_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    RR_position,
                )
            )

            # clear orders if required
            RR_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    RR_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )

            # make orders for resin
            RR_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                RR_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                    RR_take_orders + RR_clear_orders + RR_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            # get fair price and volume data
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )

            # take best kelp orders
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )

            # clear orders
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )

            # make orders
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                    KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
