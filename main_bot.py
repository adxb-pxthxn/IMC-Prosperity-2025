import json
from abc import abstractmethod
from collections import deque
import numpy as np
from typing import List, Any, TypeAlias, Tuple
import jsonpickle
import numpy.random as random

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

LIMITS = {
    "CROISSANT": 250,
    "JAMS": 350,
    "DJEMBE": 60,
    "KELP": 50,
    "RAINFOREST_RESIN": 50,
    "SQUID_INK": 50,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "DJEMBES":60,
    "MAGNIFICENT_MACARONS": 75

}

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = LIMITS[self.symbol]

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
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

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

        if (fair_price is None):
            return
        
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

    def act(self, state, traderObject):

        buy_volume = 0
        sell_volume = 0
        order_depth = state.order_depths[self.symbol]

        fair = self.get_fair_price(order_depth, traderObject)

        # get and sort each side of the order book #
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # get current position and set current position limits #
        position = state.position.get(self.symbol, 0)
        position_limit = self.limit

        sell_orders = order_depth.sell_orders
        buy_orders = order_depth.buy_orders
        try:
            best_ask_fair = min([p for p in sell_orders.keys() if p > fair], default=fair + 1)
        except ValueError:
            best_ask_fair = fair

        try:
            best_bid_fair = max([p for p in buy_orders.keys() if p < fair], default=fair - 1)
        except ValueError:
            best_bid_fair = fair

        if sell_orders:
            best_ask = min(sell_orders.keys())
            best_ask_amount = -sell_orders[best_ask]
            if best_ask < fair:
                quant = min(best_ask_amount, position_limit - position)
                if quant > 0:
                    self.buy(best_ask, quant)
                    buy_volume += quant
        if buy_orders:
            best_bid = max(buy_orders.keys())
            best_bid_amount = buy_orders[best_bid]
            if best_bid > fair:
                quant = min(best_bid_amount, position_limit + position)
                if quant > 0:
                    self.sell(best_bid, quant)
                    sell_volume += quant

        buy_quant = position_limit - (position + buy_volume)
        if buy_quant > 0:
            self.buy(best_bid_fair + 1, buy_quant)

        sell_quant = position_limit + (position - sell_volume)
        if sell_quant > 0:
            self.sell(best_ask_fair - 1, sell_quant)


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

    def save(self) -> JSON:
        return {
            "last_price": getattr(self, "last_price", None),
            "window": list(self.window)
        }

    def load(self, data: JSON) -> None:
        self.last_price = data.get("last_price", None)
        self.window = deque(data.get("window", []))


class ResinStrategy(MeanReversion):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.ewm = EWM(0.15)
        self.last_price = 10000

    def get_mid_price(self, order, traderObject):

        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            self.last_price = (best_bid + best_ask) / 2.0
            return self.last_price
        else:
            return None

    def get_fair_price(self, order, traderObject):
        return self.ewm.update(self.get_mid_price(order, traderObject))

    def save(self) -> JSON:
        return {
            "last_price": getattr(self, "last_price", None),
        }

    def load(self, data: JSON) -> None:
        self.last_price = data.get("last_price", None)


class SquidInkStrategy(MarketMaking):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.last_price = None
        self.price_history = []
        self.tickCount = 0
        self.last_trade_time = -1000
        self.window_size = 75

    def compute_volatility(self, prices: list[float]) -> float:
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

    def should_quote(
        self,
        order_depth: OrderDepth,
        vol: float,
        mm_ask: float,
        mm_bid: float
    ) -> bool:
        if mm_ask is None or mm_bid is None:
            return False
        if len(order_depth.sell_orders) < 2 or len(order_depth.buy_orders) < 2:
            return False
        
        self.tickCount += 1

        cooldown_ms = (-40)/(1+pow(2.7128, abs(4*vol))) + 20
        if self.tickCount < self.last_trade_time + cooldown_ms:
            return False
        self.last_trade_time = self.tickCount
        return True

    def get_fair_price(self, order_depth: OrderDepth, traderObject) -> float:
        traderObject = traderObject
        min_volume = 15

        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return None

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        filtered_ask = [
            price for price, volume in order_depth.sell_orders.items()
            if abs(volume) >= min_volume
        ]
        filtered_bid = [
            price for price, volume in order_depth.buy_orders.items()
            if abs(volume) >= min_volume
        ]

        mm_ask = min(filtered_ask) if filtered_ask else None
        mm_bid = max(filtered_bid) if filtered_bid else None

        if mm_ask is None or mm_bid is None:
            mid_price = traderObject.get("SQUID_last_price", (best_ask + best_bid) / 2)
        else:
            mid_price = (mm_ask + mm_bid) / 2

        self.price_history.append(mid_price)
        if len(self.price_history) > self.window_size:
            self.price_history.pop(0)

        vol = self.compute_volatility(self.price_history)
        vol = max(vol, 0.0001)

        alpha = max(0.05, 0.3 - vol * 10)
        beta = max(0.05, 0.25 - vol * 10)
        spread_padding = vol * 1000

        historical_mean = sum(self.price_history) / len(self.price_history)
        deviation = mid_price - historical_mean

        if self.last_price:
            ret = (mid_price - self.last_price) / self.last_price
        else:
            ret = 0.0

        if (deviation > 0 and ret < 0) or (deviation < 0 and ret > 0):
            reversion_adjustment = -alpha * deviation
        else:
            reversion_adjustment = 0.0

        momentum_adjustment = -ret * beta * mid_price


        if not self.should_quote(order_depth, vol, mm_ask, mm_bid):
            return None

        fair_price = mid_price + reversion_adjustment + momentum_adjustment - spread_padding

        self.last_price = mid_price
        return fair_price


class JamStrategy(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.prices = deque(maxlen=20) 
        self.alpha = 0.1
        self.b1=[]
        self.b2=[]
    def run(self, state, traderObject):
        return super().run(state, traderObject),self.b1,self.b2

    def update_signal(self, price: float) -> float:
        self.prices.append(price)
        if len(self.prices) < 2:
            return 0
        return price - np.mean(self.prices) 

    def get_mid_price(self, order: OrderDepth):
        if order.buy_orders and order.sell_orders:
            return (max(order.buy_orders) + min(order.sell_orders)) / 2
        return None

    def act(self, state: TradingState, traderObject) -> None:
        order = state.order_depths[self.symbol]
        mid_price = self.get_mid_price(order)

        if mid_price is None:
            return

        signal = self.update_signal(mid_price)
        position = state.position.get(self.symbol, 0)

        basket1=self.get_mid_price(state.order_depths['PICNIC_BASKET1'])
        positionb1 = state.position.get('PICNIC_BASKET1', 0)
        positionb2 = state.position.get('PICNIC_BASKET2', 0)
        basket2=self.get_mid_price(state.order_depths['PICNIC_BASKET2'])
        logger.print(state.position)
        if signal > self.alpha:
            if position < self.limit:
                self.buy(int(mid_price), self.limit - position)
            if positionb1 < 60 and random.rand()>0.02:
                self.buy_other(int(basket1), 60 - positionb1, 'PICNIC_BASKET1')
            if positionb2 < 100 and random.rand()>0.02:
                self.buy_other(int(basket2), 100 - positionb2, 'PICNIC_BASKET2')

        elif signal < -self.alpha:
            if position > -self.limit:
                self.sell(int(mid_price), self.limit + position)
            if positionb1 > -60:
                self.sell_other(int(basket1), 60 + positionb1, 'PICNIC_BASKET1')
            if positionb2 > -100:
                self.sell_other(int(basket2), 100 + positionb2, 'PICNIC_BASKET2')

    def buy_other(self, price: int, quantity: int,symbol) -> None:
        if symbol=="PICNIC_BASKET1":
            self.b1.append(Order(symbol, round(price), quantity))
        self.b2.append(Order(symbol, round(price), quantity))

    def sell_other(self, price: int, quantity: int,symbol) -> None:
        if symbol=="PICNIC_BASKET1":
            self.b1.append(Order(symbol, round(price), -quantity))
        self.b2.append(Order(symbol, round(price), -quantity))

    def save(self) -> JSON:
        return list(self.prices)

    def load(self, data: JSON) -> None:
        self.prices = deque(data if data else [], maxlen=20)


class Basket1Strat(Strategy):

    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)   
        self.window = deque()
        self.params=[  0.99   , 0.98 ,0.98 , 0.98]
        self.ewm=EWM(1/2300)
    
    def get_mid_price(self, order, traderObject=None):
        
        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            return (best_bid + best_ask) / 2.0
    
    def act(self,state,traderObject):

        order_depth = state.order_depths[self.symbol]
        position=state.position.get(self.symbol, 0)
        
        
        basket=self.get_mid_price(order_depth)

        cros,jams=self.get_mid_price(state.order_depths['CROISSANTS']),self.get_mid_price(state.order_depths['JAMS'])
        


        diff = basket - (4 * cros + 2 * jams)


        signal = self.ewm.update(diff) - diff

        # Order book for trading GIFT_BASKET
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        buy_price = max(order_depth.sell_orders.keys())
        sell_price =  min(order_depth.buy_orders.keys())

        buy_vol = order_depth.sell_orders[buy_price]
        sell_vol = order_depth.buy_orders[sell_price]
        

        # if diff>self.params[0]:

        #     self.sell(sell_price, self.limit-position)
        # elif diff<self.params[1]:

        #     self.buy(buy_price, self.limit-position)
        # else:

        threshold = 35

        if signal < -threshold:
            self.sell(sell_price, sell_vol)
        elif signal > threshold:
            self.buy(buy_price,- buy_vol)







class Basket2Strat(Strategy):

    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)   
        self.window = deque()
        self.params=[40,-10]
        self.ewm=EWM(1/2300)

    
    def get_mid_price(self, order, traderObject=None):
        
        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            return (best_bid + best_ask) / 2.0
    
    def act(self,state: TradingState,traderObject):

        order_depth = state.order_depths[self.symbol]
        position=state.position.get(self.symbol, 0)
        
        
        basket=self.get_mid_price(order_depth)

        cros,jams=self.get_mid_price(state.order_depths['CROISSANTS']),self.get_mid_price(state.order_depths['JAMS'])
        


        diff = basket - (4 * cros + 2 * jams)


        signal = self.ewm.update(diff) - diff

        # Order book for trading GIFT_BASKET
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        buy_price = max(order_depth.sell_orders.keys())
        sell_price =  min(order_depth.buy_orders.keys())

        buy_vol = order_depth.sell_orders[buy_price]
        sell_vol = order_depth.buy_orders[sell_price]
        

        # if diff>self.params[0]:

        #     self.sell(sell_price, self.limit-position)
        # elif diff<self.params[1]:

        #     self.buy(buy_price, self.limit-position)
        # else:

        threshold = 35

        if signal < -threshold:
            self.sell(sell_price, sell_vol)
        elif signal > threshold:
            self.buy(buy_price,- buy_vol)



class MacaronStrategy(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.window_size = 25
        self.window = []

        # Indicator history
        self.sugar_history = []
        self.sunlight_history = []
        self.import_tariff_history = []
        self.export_tariff_history = []
        self.transport_fee_history = []

    def get_mid_price(self, order_depth):
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2.0
        return None

    def act(self, state: TradingState, traderObject):
        order_depth = state.order_depths[self.symbol]
        cur_mid_price = self.get_mid_price(order_depth)
        if cur_mid_price is None:
            return

        # Update price window
        self.window.append(cur_mid_price)
        if len(self.window) > self.window_size:
            self.window.pop(0)

        obs = state.observations.conversionObservations[self.symbol]

        # Extract observations
        sugar = obs.sugarPrice
        sunlight = obs.sunlightIndex
        imp_tariff = obs.importTariff
        exp_tariff = obs.exportTariff
        trans_fee = obs.transportFees

        # Update environmental history
        self.sugar_history.append(sugar)
        self.sunlight_history.append(sunlight)
        self.import_tariff_history.append(imp_tariff)
        self.export_tariff_history.append(exp_tariff)
        self.transport_fee_history.append(trans_fee)

        for history in [
            self.sugar_history,
            self.sunlight_history,
            self.import_tariff_history,
            self.export_tariff_history,
            self.transport_fee_history,
        ]:
            if len(history) > self.window_size:
                history.pop(0)

        # Indicators
        sugar_momentum = (
            np.log(self.sugar_history[-1] / self.sugar_history[-2])
            if len(self.sugar_history) > 1 else 0
        )
        sunlight_trend = (
            self.sunlight_history[-1] - self.sunlight_history[-2]
            if len(self.sunlight_history) > 1 else 0
        )

        # Averages and deviations
        avg_imp = np.mean(self.import_tariff_history)
        avg_exp = np.mean(self.export_tariff_history)
        avg_trans = np.mean(self.transport_fee_history)

        std_imp = np.std(self.import_tariff_history) or 0.1
        std_exp = np.std(self.export_tariff_history) or 0.1
        std_trans = np.std(self.transport_fee_history) or 0.1

        # Deviation signals
        import_penalty = -(imp_tariff - avg_imp) / (std_imp + 1e-5) if imp_tariff > avg_imp + std_imp else 0
        import_bonus = (avg_imp - imp_tariff) / (std_imp + 1e-5) if imp_tariff < avg_imp - std_imp else 0

        export_bonus = (avg_exp - exp_tariff) / (std_exp + 1e-5) if exp_tariff < avg_exp - std_exp else 0
        export_penalty = -(exp_tariff - avg_exp) / (std_exp + 1e-5) if exp_tariff > avg_exp + std_exp else 0

        transport_penalty = -(trans_fee - avg_trans) / (std_trans + 1e-5) if trans_fee > avg_trans + std_trans else 0

        # Combined tariff modifier (bounded)
        tariff_modifier = np.clip(import_penalty + import_bonus + export_bonus + export_penalty + transport_penalty, -1, 1)

        # Sugar and sunlight impacts
        sugar_impact = np.tanh(sugar_momentum * 50)
        sunlight_impact = np.tanh(-sunlight_trend * 2)

        # Final modifier
        fundamental_modifier = 1 + 0.01 * (sugar_impact + sunlight_impact + tariff_modifier)

        # Fair value calculation
        technical_fair = np.mean(self.window) if self.window else cur_mid_price
        fair_value = technical_fair * fundamental_modifier

        # Round price levels
        bid_price = int(fair_value - 1)
        ask_price = int(fair_value + 1)

        if (sugar_momentum > 0.005 or sunlight_trend < 0.01) and imp_tariff < avg_imp + std_imp:
            self.buy(bid_price, 3)

        if (sugar_momentum < -0.005 or sunlight_trend > 0.01) and exp_tariff > avg_exp - std_exp:
            self.sell(ask_price, 3)


    def save(self) -> JSON:
        return {
            "last_price": getattr(self, "last_price", None),
            "window": list(self.window)
        }

    def load(self, data: JSON) -> None:
        self.last_price = data.get("last_price", None)
        self.window = list(data.get("window", []))

class EWM:
    def __init__(self, alpha=0.002):
        self.alpha = alpha
        self.value = None

    def update(self, price):
        if self.value is None:
            self.value = price
        else:
            self.value = self.alpha * price + (1 - self.alpha) * self.value
        return self.value


class EWMAbs:
    def __init__(self, price_alpha=0.0015, dev_alpha=0.002, long_alpha=0.0001):
        self.price_ema = EWM(alpha=price_alpha)
        self.long_ema = EWM(alpha=long_alpha)
        self.deviation_ema = EWM(alpha=dev_alpha)

    def update(self, price):
        ema_price = self.price_ema.update(price)
        long_ema = self.long_ema.update(price)

        abs_deviation = abs(price - ema_price)

        ema_abs_deviation = self.deviation_ema.update(abs_deviation)

        return ema_price, ema_abs_deviation, long_ema


class Trader:
    def __init__(self) -> None:
        self.strategies = {}
        self.strategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, LIMITS[symbol]) for symbol, clazz in {
            # "KELP": KelpStrategy,
            # "RAINFOREST_RESIN": ResinStrategy,
            # "SQUID_INK": SquidInkStrategy,
            # # "DJEMBES":JamStrategy,
            # "PICNIC_BASKET1":Basket1Strat,
            # "PICNIC_BASKET2":Basket2Strat,
            "MAGNIFICENT_MACARONS": MacaronStrategy
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        jam_strat=JamStrategy('JAMS',LIMITS['JAMS'])
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        for symbol, strategy in [("JAMS",jam_strat)]:
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if (symbol in state.order_depths and
                    len(state.order_depths[symbol].buy_orders) > 0 and
                    len(state.order_depths[symbol].sell_orders) > 0
            ):
                (strategy_orders, strategy_conversions),t1,t2 = strategy.run(state,
                                                                     traderObject=old_trader_data.get(symbol, {}), )
            new_trader_data[symbol] = strategy.save()

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