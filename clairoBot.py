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
        "fair_value": 10000,               # long-term stable fair price to anchor to
        "take_width": 1,                  # unused in resin for now
        "clear_width": 0,                 # unused in resin for now

        # === signal-driven MR logic ===
        "std_threshold": 1.0,             # how volatile it needs to be to trigger MR
        "momentum_threshold": 0.0,        # how directional it needs to be to stay out of static
        "mr_base_edge": 5,                # base MR quote offset from fair
        "mr_std_multiplier": 1.2,         # scale edge size based on volatility
        "min_mr_edge": 1,                 # quote re-aggression floor

        # === static MM behavior ===
        "default_edge": 4,                # distance from fair to quote in MM (static mode)
        "disregard_edge": 1,              # don't quote too close to book levels
        "join_edge": 2,                   # how close we need to be to join vs penny

        # === inventory biasing ===
        "soft_position_limit": 30,        # where we start skewing quotes based on inventory
        "skew_adjust": 1,                 # how much to nudge quotes when leaning

        "stale_requote_aggression": 1,    # ticks to boost bid/ask after stale
        "stale_aggression_threshold": 5,  # how many ticks without fills before boost
        "market_sweep_threshold": 15,     # volume change threshold to detect sweeps
        "partial_fill_tracking": True,
        "fallback_edge_boost": 1 ,         # increase quoting width in fallback mode

        # === mode + fallback dynamics ===
        "max_mr_failures": 3,             # how many bad MR entries before forcing static
        "TRADE": True                     # toggle trading on/off
    },
    Product.KELP: {
        "take_width": 1,                  # default spread for MM
        "clear_width": 0,                 # how close the BP to our FP to clear, 0 means it must be fair
        "prevent_adverse": True,          # flag to prevent trades when there is too much depth at best price
        "adverse_volume": 15,             # this is the volume threshold to toggle prevent_adverse
        "reversion_beta": -0.229,         # mean reversion coefficient pretty much our confidence on it dropping to mean
        "disregard_edge": 1,              # disregards orders for joining or pennying within this value from fair
        "join_edge": 1,                   # joins when someone is 1 tick from FP, keeps us at the top of the book
        "default_edge": 1,                # when no chances for joining and pennying just quote 1 tick from FP
        "TRADE": False                    # flag to trade or not, for iso testing strats for each asset

    },
}

class Trader:
    DEBUG_LOGGING = True
    def __init__(self, params=None):
        self.current_tick = None
        if params is None:
            params = PARAMS
        self.params = params
        self.mid_price_history = []
        self.LIMIT = {Product.RAINFOREST_RESIN: 20, Product.KELP: 20}  # self-imposed position limit
        self.resin_mid_history = []  # for storing mid prices over time

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
                bid = best_bid_below_fair  # join
            else:
                bid = best_bid_below_fair + 1  # penny

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
            if mm_ask is None or mm_bid is None:
                if traderObject.get("KELP_last_price", None) is None:  # if last price was also None
                    mmmid_price = (best_ask + best_bid) / 2  # use best ask and best bid for mid
                else:
                    mmmid_price = traderObject["KELP_last_price"]  # if not then use the last price
            else:
                mmmid_price = (mm_ask + mm_bid) / 2  # else we use the filtered data for mid price

            # if we have a valid last price
            if traderObject.get("KELP_last_price", None) is not None:
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

    def rain_order(self, order_depth: OrderDepth, position: int) -> list[Order]:
        product = Product.RAINFOREST_RESIN
        params = self.params[product]
        orders: list[Order] = []

        self.update_resin_mid(order_depth)

        # === init state vars if missing ===
        if not hasattr(self, "resin_stale_ticks"):
            self.resin_stale_ticks = 0
            self.resin_last_position = position
            self.resin_last_mode = "mean_reversion"
            self.resin_reaggression_ticks = 0
            self.resin_mr_failures = 0
            self.resin_no_fill_ticks = 0
            self.resin_refresh_ticks = 0

        # === check if we're getting filled ===
        filled = position != self.resin_last_position
        if not filled:
            self.resin_stale_ticks += 1
            self.resin_reaggression_ticks += 1
            self.resin_no_fill_ticks += 1
            self.resin_refresh_ticks += 1
        else:
            self.resin_stale_ticks = 0
            self.resin_reaggression_ticks = 0
            self.resin_no_fill_ticks = 0
            self.resin_refresh_ticks = 0
            self.resin_mr_failures = max(0, self.resin_mr_failures - 1)
        self.resin_last_position = position

        # === pull params ===
        fair = params["fair_value"]
        std_threshold = params.get("std_threshold", 1.0)
        momentum_threshold = params.get("momentum_threshold", 0.0)
        base_mr_edge = params.get("mr_base_edge", 5)
        std_mult = params.get("mr_std_multiplier", 1.2)
        skew_adjust = params.get("skew_adjust", 1)
        max_mr_failures = params.get("max_mr_failures", 3)
        min_edge = params.get("min_mr_edge", 1)
        default_edge = params.get("default_edge", 4)
        disregard_edge = params.get("disregard_edge", 1)
        join_edge = params.get("join_edge", 2)
        soft_position_limit = params.get("soft_position_limit", 30)

        # new fallback+aggression tuning
        stale_aggression_threshold = params.get("stale_aggression_threshold", 5)
        stale_requote_aggression = params.get("stale_requote_aggression", 1)
        fallback_edge_boost = params.get("fallback_edge_boost", 1)

        # === signal gen ===
        mean = self.calculate_mean()
        std = self.calculate_std()
        momentum = self.calculate_momentum()
        buffer_full = len(self.resin_mid_history) >= 20

        # === mode control ===
        low_volatility = std < std_threshold
        no_momentum = abs(momentum) <= momentum_threshold
        weak_signal = low_volatility and no_momentum
        force_static = self.resin_stale_ticks > 20 and weak_signal
        conditions_favor_mr = std > std_threshold and abs(momentum) > momentum_threshold

        if self.resin_mr_failures >= max_mr_failures:
            conditions_favor_mr = False

        mode = "mean_reversion"
        if weak_signal or force_static or not conditions_favor_mr:
            mode = "static"
            if self.resin_last_mode == "mean_reversion":
                self.resin_mr_failures += 1
        elif self.resin_last_mode == "static" and conditions_favor_mr:
            self.resin_mr_failures = 0

        if self.DEBUG_LOGGING and mode != self.resin_last_mode:
            logger.print(f"Mode transition: {self.resin_last_mode} → {mode}")
        self.resin_last_mode = mode

        # === quote logic ===
        buy_order_volume = 0
        sell_order_volume = 0

        if mode == "static":
            # fallback to MM but increase edge slightly if we've been stuck
            adjusted_edge = default_edge
            if self.resin_stale_ticks > stale_aggression_threshold:
                adjusted_edge += fallback_edge_boost

            orders, buy_order_volume, sell_order_volume = self.make_orders(
                product,
                order_depth,
                fair,
                position,
                buy_order_volume,
                sell_order_volume,
                disregard_edge,
                join_edge,
                adjusted_edge,
                manage_position=True,
                soft_position_limit=soft_position_limit,
            )
        else:
            # MR mode with re-aggression + vol/momentum-based adjustment
            edge = base_mr_edge + std * std_mult
            if std < 1.5:
                edge *= 0.8
            if buffer_full:
                edge *= 0.9
            if self.resin_reaggression_ticks > 0:
                edge = max(min_edge, edge - self.resin_reaggression_ticks)

            # If we're stale for too long, add edge aggression
            if self.resin_stale_ticks > stale_aggression_threshold:
                edge = max(min_edge, edge - stale_requote_aggression)

            skew = -skew_adjust if momentum > 0 else skew_adjust
            bid = round(fair - edge + skew)
            ask = round(fair + edge + skew)
            if position > 0: ask -= 1
            if position < 0: bid += 1

            limit = self.LIMIT[product]
            orders.append(Order(product, bid, limit))
            orders.append(Order(product, ask, -limit))

        # === logging ===
        if self.DEBUG_LOGGING and self.current_tick % 100 == 0:
            logger.print(
                f"RESIN tick: {self.current_tick} | mode: {mode} | fair: {fair} | mean: {round(mean)} | "
                f"std: {round(std, 2)} | momentum: {momentum} | stale_ticks: {self.resin_stale_ticks} | "
                f"reaggression_ticks: {self.resin_reaggression_ticks} | mr_failures: {self.resin_mr_failures}"
            )
            logger.print(f"Quotes: {[(o.price, o.quantity) for o in orders]}")

        return orders

    # === Updates midprice history for resin every tick ===
    def update_resin_mid(self, order_depth: OrderDepth):
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2
            self.resin_mid_history.append(mid)
            self.resin_mid_history = self.resin_mid_history[-20:]  # keep history max 20 long

    # === gets the average mid price ===
    def calculate_mean(self) -> float:
        if not self.resin_mid_history:
            return self.params[Product.RAINFOREST_RESIN]["fair_value"]
        return sum(self.resin_mid_history) / len(self.resin_mid_history)

    # === std dev of mid prices to detect quiet market ===
    def calculate_std(self) -> float:
        if len(self.resin_mid_history) < 2:
            return 0.0
        mean = self.calculate_mean()
        return (sum((x - mean) ** 2 for x in self.resin_mid_history) / len(self.resin_mid_history)) ** 0.5

    # === basic directional momentum signal from midprice trend ===
    def calculate_momentum(self) -> int:
        hist = self.resin_mid_history
        if len(hist) < 3:
            return 0
        return 1 if hist[-1] > hist[0] else -1 if hist[-1] < hist[0] else 0

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        self.current_tick = state.timestamp

        # === RAINFOREST_RESIN block === #
        if (
                Product.RAINFOREST_RESIN in self.params and
                Product.RAINFOREST_RESIN in state.order_depths and
                self.params[Product.RAINFOREST_RESIN].get("TRADE", True)
        ):
            result[Product.RAINFOREST_RESIN] = self.rain_order(
                state.order_depths[Product.RAINFOREST_RESIN],
                state.position.get(Product.RAINFOREST_RESIN, 0)
            )


        # === KELP block === #
        if (
                Product.KELP in self.params and
                Product.KELP in state.order_depths and
                self.params[Product.KELP].get("TRADE", True)
        ):
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
