import csv
import json
import os
from abc import abstractmethod
from collections import deque
import numpy as np
from typing import List, Any, TypeAlias, Tuple
import jsonpickle


from datamodel import TradingState, Symbol, Order, Listing, OrderDepth, Trade, Observation, ProsperityEncoder

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


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

position_limit=50

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState, traderObject) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState, traderObject) -> tuple[list[Any], list[Any]]:
        self.orders = []
        self.conversions = []

        self.act(state, traderObject=traderObject)
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, round(price), quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, round(price), -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return {
            "last_price": getattr(self, "last_price", None),
            "window": list(self.window)
        }

    def load(self, data: JSON) -> None:
        self.last_price = data.get("last_price", None)
        self.window = deque(data.get("window", []))


class MarketMaking(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_fair_price(self, order: OrderDepth, traderObject) -> int:
        raise NotImplementedError

    def act(self, state: TradingState, traderObject) -> None:
        order_depth = state.order_depths[self.symbol]

        
        fair_price = self.get_fair_price(order_depth, traderObject)

        # get and sort each side of the order book #
        order_depth = state.order_depths[self.symbol]
        buy_orders = order_depth.buy_orders.items()
        sell_orders = order_depth.sell_orders.items()

        # get current position and set current position limits #
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # append a boolean based on if we are at position limit #
        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()  # remove left-most when size goes over

        #  flag for passively executing, occurs when the buffer window is full,
        #  at least half the indexs in window are 1 meaning at least half the
        # ticks in the buffer window trades were executed, prev ticker also 1
        soft_liquidate = (
                len(self.window) == self.window_size and
                sum(self.window) >= self.window_size / 2 and
                self.window[-1]
        )

        #  flag for aggresive executing, occurs when the buffer window is full,
        #  and trades were made for every tick within the buffer window
        hard_liquidate = (
                len(self.window) == self.window_size and
                all(self.window)
        )

        # calculate max buy and min sell prices, if inventory less than limit/2
        max_buy_price = fair_price - 1 if position > self.limit * 0.5 else fair_price
        min_sell_price = fair_price + 1 if position < self.limit * -0.5 else fair_price

        # for all orders in sell orders
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:  # fulfil profitable sell orders
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        # if can buy aggressively
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy
            self.buy(fair_price, quantity)  # buy at fair price
            to_buy -= quantity

        # if can buy passively
        if to_buy > 0 and soft_liquidate:
            quantity = to_buy
            self.buy(fair_price - 2, quantity)  # buy under fair price
            to_buy -= quantity

        # if can buy but no skew
        if to_buy > 0:
            # find the price with most buy orders
            common_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, common_buy_price + 1)  # buy at or above popular price
            self.buy(price, to_buy)

        # same principal
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell
            self.sell(fair_price, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell
            self.sell(fair_price + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

class MeanReversion(MarketMaking):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
    
    def act(self,state,traderObject):

        buy_volume=0
        sell_volume=0
        order_depth = state.order_depths[self.symbol]
        
        fair= self.get_fair_price(order_depth, traderObject)

        # get and sort each side of the order book #
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # get current position and set current position limits #
        position = state.position.get(self.symbol, 0)

        sell_orders=order_depth.sell_orders
        buy_orders=order_depth.buy_orders
        try:
            best_ask_fair = min([p for p in sell_orders.keys() if p > fair], default=fair+1)
        except ValueError:
            best_ask_fair = fair
            
        try:
            best_bid_fair = max([p for p in buy_orders.keys() if p < fair], default=fair-1)
        except ValueError:
            best_bid_fair = fair

        if sell_orders:
            best_ask=min(sell_orders.keys())
            best_ask_amount=-sell_orders[best_ask]
            if best_ask<fair:
                quant=min(best_ask_amount,position_limit-position)
                if quant>0:
                    self.buy(best_ask,quant)
                    buy_volume+=quant
        if buy_orders:
            best_bid = max(buy_orders.keys())
            best_bid_amount = buy_orders[best_bid]
            if best_bid > fair:
                quant = min(best_bid_amount, position_limit + position)
                if quant > 0:
                    self.sell(best_bid,quant)
                    sell_volume += quant
        
        buy_quant=position_limit-(position+buy_volume)
        if buy_quant>0:
            self.buy(best_bid_fair+1,buy_quant)


        sell_quant=position_limit+(position-sell_volume)
        if sell_quant>0:
            self.sell(best_ask_fair - 1,sell_quant)


class KelpStrategy(MarketMaking):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.last_price = None

    def get_fair_price(self, order_depth: OrderDepth, traderObject) -> float:
        traderObject = traderObject

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
                if abs(order_depth.sell_orders[price]) >= 15  # make this a PARAM later
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= 15
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
            if self.last_price is not None:
                last_returns = (mmmid_price - self.last_price) / self.last_price
                pred_returns = last_returns * -0.229
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            self.last_price = mmmid_price
            return fair

        return None

class ResinStrategy(MeanReversion):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.ewm=EWM(0.15)
        self.last_price = 10_000

    def get_mid_price(self, order, traderObject):
        
        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            self.last_price = (best_bid + best_ask) / 2.0
            return self.last_price
        else:
            return None 
    
    def get_fair_price(self, order, traderObject):
        return self.ewm.update(self.get_mid_price(order,traderObject))

    def save(self) -> JSON:
        return {
            "last_price": getattr(self, "last_price", None),
        }

    def load(self, data: JSON) -> None:
        self.last_price = data.get("last_price", None)


class EWM:
    def __init__(self,alpha=0.002):
        self.alpha=alpha
        self.value=None
    def update(self,price):
        if self.value is None:
            self.value = price 
        else:
            self.value = self.alpha * price + (1 - self.alpha) * self.value
        return self.value
    
class EWMAbs:
    def __init__(self, price_alpha=0.0015, dev_alpha=0.002,long_alpha=0.0001):
        self.price_ema = EWM(alpha=price_alpha) 
        self.long_ema=EWM(alpha=long_alpha)
        self.deviation_ema = EWM(alpha=dev_alpha) 

    def update(self, price):

        ema_price = self.price_ema.update(price)
        long_ema=self.long_ema.update(price)


        abs_deviation = abs(price - ema_price)

        ema_abs_deviation = self.deviation_ema.update(abs_deviation)

        return ema_price, ema_abs_deviation,long_ema


class InkStrategy(MarketMaking):
    def __init__(self, symbol, limit,t=0.4,alpha1=0.2,alpha2=0.25,alpha3=0.01):
        super().__init__(symbol, limit)
        self.emv=EWMAbs(alpha1,alpha2,alpha3)
        self.threshold=t
        self.price_history = []
        self.resin_history = []
        self.lambda_ = 0.94  # Decay factor for EWMA risk-free rate estimation
        self.window_size = 100
        self.trade_log = []
        self.log_filename = f"{symbol}_trades_log.csv"
        self.logged_header = False


    def store_price_data(self, squid_ink_price, resin_price):
        if squid_ink_price is not None:
            self.price_history.append(squid_ink_price)
        if resin_price is not None:
            self.resin_history.append(resin_price)

        self.price_history = self.price_history[-self.window_size:]
        self.resin_history = self.resin_history[-self.window_size:]

    def calculate_realized_volatility(self, prices):
        if len(prices) < 2:
            return 0
        log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
        return np.sqrt(np.sum(log_returns**2))  # Intraday realized volatility

    def estimate_risk_free_rate(self):
        if len(self.resin_history) < 2:
            return 0
        log_returns = np.log(np.array(self.resin_history[1:]) / np.array(self.resin_history[:-1]))
        ewma_r = np.mean(log_returns) * self.window_size  # Annualized, but could be shorter term
        return ewma_r * self.lambda_ + (1 - self.lambda_) * log_returns[-1]

    def estimate_time_to_maturity(self):
        trading_intervals = min(len(self.price_history), self.window_size)
        return 1 / trading_intervals if trading_intervals > 0 else 1 / self.window_size

    def get_fair_price(self, order: OrderDepth, traderObject):
        est_price = self.emv.update(self.get_mid_price(order,traderObject))
        if not order.sell_orders or not order.buy_orders:
            return est_price

        best_ask = min(order.sell_orders.keys())
        best_bid = max(order.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        vol = self.calculate_realized_volatility(self.price_history)
        r = self.estimate_risk_free_rate()
        T = self.estimate_time_to_maturity()

        expected_price = mid_price * np.exp(r * T + 0.5 * vol**2 * T)

        self.store_price_data(mid_price, traderObject.get("RAINFOREST_RESIN_last_price", None))
        return expected_price

    def buy(self, price: int, quantity: int) -> None:
        super().buy(price, quantity)
        self.log_trade("BUY", price, quantity)

    def sell(self, price: int, quantity: int) -> None:
        super().sell(price, quantity)
        self.log_trade("SELL", price, quantity)

    def log_trade(self, side: str, price: float, quantity: int):
        best_bid = max(self.last_order_depth.buy_orders.keys()) if self.last_order_depth.buy_orders else None
        best_ask = min(self.last_order_depth.sell_orders.keys()) if self.last_order_depth.sell_orders else None

        entry = {
            "timestamp": self.current_timestamp,
            "position": self.last_position,
            "side": side,
            "price": round(price),
            "quantity": quantity,
            "best_bid": best_bid,
            "best_ask": best_ask
        }

        self.trade_log.append(entry)

    def act(self, state: TradingState, traderObject) -> None:
        self.current_timestamp = state.timestamp
        self.last_position = state.position.get(self.symbol, 0)
        self.last_order_depth = state.order_depths[self.symbol]
        super().act(state, traderObject)

    def get_mid_price(self, order, traderObject):
        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            return (best_bid + best_ask) / 2.0
        else:
            # Fallback or default, if one side of the order book is missing
            return None 
        
    def save(self) -> JSON:
        # Write to CSV
        if self.trade_log:
            write_header = not os.path.isfile(self.log_filename)
            with open(self.log_filename, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.trade_log[0].keys())
                if write_header:
                    writer.writeheader()
                writer.writerows(self.trade_log)
            self.trade_log.clear()  # clear after writing

        return {
            "price_history": self.price_history,
            "resin_history": self.resin_history,
        }
    def load(self, data: JSON) -> None:
        self.price_history = list(data.get("price_history", []))
        self.resin_history = list(data.get("resin_history", []))


class Trader:
    def __init__(self) -> None:
        limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK":50
        }
        self.strategies={}
        self.strategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
        "KELP": KelpStrategy,
            "RAINFOREST_RESIN": ResinStrategy,
            "SQUID_INK":InkStrategy
            
        }.items()}
        
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if (symbol in state.order_depths and
                    len(state.order_depths[symbol].buy_orders) > 0 and
                    len(state.order_depths[symbol].sell_orders) > 0
            ):
                strategy_orders, strategy_conversions = strategy.run(state,
                                                                     traderObject=old_trader_data.get(symbol, {}), )

                orders[symbol] = strategy_orders
                conversions += sum(strategy_conversions)

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
    
