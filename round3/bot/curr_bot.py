import json
from abc import abstractmethod
from collections import deque
import math
from statistics import NormalDist
import numpy as np
from typing import List, Any, TypeAlias, Tuple
import jsonpickle
import numpy.random as random

from datamodel import TradingState, Symbol, Order, Listing, OrderDepth, Trade, Observation, ProsperityEncoder,ConversionObservation

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
     "VOLCANIC_ROCK_VOUCHER_9500":200,
     "VOLCANIC_ROCK_VOUCHER_10000":200,
     "VOLCANIC_ROCK_VOUCHER_9750":200,
     "VOLCANIC_ROCK_VOUCHER_10250":200,
     "VOLCANIC_ROCK_VOUCHER_10500":200,

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
        if vol > 0.02:
            return False

        self.tickCount += 1

        cooldown_ms = 15
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
        self.prices = deque(maxlen=20)  # price window for JAMS
        self.alpha = 0.1  # signal threshold to act

    def update_signal(self, price: float) -> float:
        self.prices.append(price)  # append cur price to window
        if len(self.prices) < 2:  # if window is too small to calc mean ignore
            return 0
        return price - np.mean(self.prices)  # simple momentum signal, positive means it will move up, negative down

    def get_mid_price(self, order: OrderDepth):
        if order.buy_orders and order.sell_orders:  # if orderbook has orders on both sides
            return (max(order.buy_orders) + min(order.sell_orders)) / 2  # return mid-price
        return None

    def act(self, state: TradingState, traderObject) -> None:
        order = state.order_depths[self.symbol]
        mid_price = self.get_mid_price(order)
        if mid_price is None:  # don't trade if order-book is empty on either side
            return

        signal = self.update_signal(mid_price)  # update signal
        position = state.position.get(self.symbol, 0)

        if signal > self.alpha and position < self.limit:  # if signal is strong enough and we aren't at limit
            # price is moving up: buy now, sell later
            self.buy(int(mid_price), self.limit - position)
        elif signal < -self.alpha and position > -self.limit:
            # price is dropping: sell now, buy later
            self.sell(int(mid_price), self.limit + position)

    def save(self) -> JSON:
        return list(self.prices)

    def load(self, data: JSON) -> None:
        self.prices = deque(data if data else [], maxlen=20)


class Basket1Strat(Strategy):

    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)   
        self.window = deque()
        self.params=[-20,-50]
        self.ewm=EWM(1/4000)
    

    def get_mid_price(self, order, traderObject=None):


        if not (order.buy_orders and order.sell_orders):
            return


        best_ask=-1
        best_bid=-1
        vol=-1

        for key,val in order.buy_orders.items():
            if not np.isnan(val) and val>vol:
                vol=val
                best_ask=key
        vol=-1
        for key,val in order.sell_orders.items():
            if not np.isnan(val) and val>vol:
                vol=val
                best_bid=key

        return (best_bid+best_ask)/2
        


    def act(self,state,traderObject):

        order_depth = state.order_depths[self.symbol]
        # position=state.position.get(self.symbol, 0)
        
        
        basket=self.get_mid_price(order_depth)

        cros,jams=self.get_mid_price(state.order_depths['CROISSANTS']),self.get_mid_price(state.order_depths['JAMS'])
        


        diff = basket - (4 * cros + 2 * jams)


        signal = self.ewm.update(diff) - diff

        # Order book for trading GIFT_BASKET
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        buy_price = max(order_depth.buy_orders.keys())
        sell_price = min(order_depth.sell_orders.keys())
        buy_vol = order_depth.buy_orders[buy_price]
        sell_vol = order_depth.sell_orders[sell_price]

        threshold = [70,-65]

        if signal > threshold[0]:
            self.buy(sell_price,-sell_vol)
        elif signal < threshold[1]:
            self.sell(buy_price, buy_vol)




class Basket2Strat(Strategy):

    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)   
        self.window = deque()
        self.params=[  0.99   , 0.98 ,0.98 , 0.98]
        self.ewm=EWM(1/4000)
    
    def get_mid_price(self, order, traderObject=None):
        
        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            return (best_bid + best_ask) / 2.0
    
    def act(self,state,traderObject):

        order_depth = state.order_depths[self.symbol]
        
        
        basket=self.get_mid_price(order_depth)

        cros,jams=self.get_mid_price(state.order_depths['CROISSANTS']),self.get_mid_price(state.order_depths['JAMS'])
        


        diff = basket - (4 * cros + 2 * jams)


        signal = self.ewm.update(diff) - diff

        # Order book for trading GIFT_BASKET
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        buy_price = max(order_depth.buy_orders.keys())
        sell_price = min(order_depth.sell_orders.keys())
        buy_vol = order_depth.buy_orders[buy_price]
        sell_vol = order_depth.sell_orders[sell_price]

        threshold = [35,-35]

        if signal > threshold[0]:
            self.buy(sell_price, -sell_vol)
        elif signal < threshold[1]:
            self.sell(buy_price, buy_vol)

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



def rough_iv(S, K, T, C):

   
    approx_iv = C / (0.2 * S * np.sqrt(T))
    approx_iv = np.clip(approx_iv, 0.01, 2.0)
    return approx_iv

class CouponStrategy(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self.voucher_strikes = [9500, 9750, 10000, 10250, 10500]
        self.smile_window = deque(maxlen=100)
        self.window=[]


    def run(self, state: TradingState, traderObject) -> tuple[list[Any], list[Any]]:
        self.orders = []
        self.conversions = []

        under_order=self.act(state, traderObject=traderObject)
        return self.orders, self.conversions,under_order
        
    def BS_CALL(self, S, K, T, r):
        coeffs = [0.23797695, 0.00152097, 0.15513003]

        # Compute log-moneyness
        m = math.log(K / S) / math.sqrt(T)
        
        sigma=np.polyval(coeffs,m)

        # # Get fitted implied vol from parabola
        # sigma = 0.22959255410663698

        # Black-Scholes call price
        def bs_call_price(S, K, T, r, sigma):
            N = NormalDist().cdf
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            return S * N(d1) - K * math.exp(-r * T) * N(d2)

        # Finite difference prices for Vega estimation
        price = bs_call_price(S, K, T, r, sigma)
        price_up = bs_call_price(S, K, T, r, sigma + 0.03)
        price_down = bs_call_price(S, K, T, r, sigma - 0.03)
        vega = (price_up - price_down) / (2 * 0.06)

        return price, price_up, price_down, sigma, vega



    def get_mid_price(self, order, traderObject=None):
        
        if order.buy_orders and order.sell_orders:
            best_bid = max(order.buy_orders.keys())
            best_ask = min(order.sell_orders.keys())
            return (best_bid + best_ask) / 2.0


    def act(self, state, traderObject) -> None:
        base_asset = "VOLCANIC_ROCK"
        if base_asset not in state.order_depths:
            return

        for sym in [base_asset, self.symbol]:
            if sym not in state.order_depths or not (
                state.order_depths[sym].buy_orders and state.order_depths[sym].sell_orders
            ):
                return

        rock = self.get_mid_price(state.order_depths[base_asset])
        coupon = self.get_mid_price(state.order_depths[self.symbol])
        if rock is None or coupon is None:
            return

        # timestamp=state.timestamp
        # mill=1_000_000
        K = int(self.symbol.split('_')[-1])
        T = 3/365

        r = 0

        # Smile-based BS pricing
        fair, up, down, sigma, vega = self.BS_CALL(rock, K, T, r)

        # Delta calc
        d1 = (np.log(rock / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = NormalDist().cdf(d1) 

        order_depth = state.order_depths[self.symbol]
        buy_price = max(order_depth.buy_orders)
        sell_price = min(order_depth.sell_orders)
        buy_vol = order_depth.buy_orders[buy_price]
        sell_vol = order_depth.sell_orders[sell_price]

        rock_depth = state.order_depths[base_asset]
        rock_buy_price = max(rock_depth.buy_orders)
        rock_sell_price = min(rock_depth.sell_orders)
        rock_buy_vol = rock_depth.buy_orders[rock_buy_price]
        rock_sell_vol = rock_depth.sell_orders[rock_sell_price]

        hedge_order = []
        residual = coupon - fair
        z_score = residual / vega if vega != 0 else 0
        # logger.print(f"Residual: {residual:.4f} | Vega: {vega:.4f} | delta: {delta:.2f} | coupon: {coupon} d+c: {fair+delta} fair:{fair}")

        threshold = 4

        rock_pos=400-abs(state.position.get(base_asset,0))

        # Sell coupon if overpriced, hedge by buying rock
        if residual > threshold and coupon>up:
            self.sell(buy_price, buy_vol)
            hedge = int(min(delta * abs(buy_vol), rock_pos))

            if hedge > 0:
                hedge_order.append(Order(base_asset, round(rock_sell_price),round( hedge*0.0175)))

        # Buy coupon if underpriced, hedge by selling rock
        elif residual <- threshold and coupon<down:
            self.buy(sell_price, rock_pos)
            hedge = int(min(delta * abs(sell_vol), rock_buy_vol))

            if hedge > 0:
                hedge_order.append(Order(base_asset, round(rock_buy_price), round(-hedge*0.0175)))

        return hedge_order




class Trader:
    def __init__(self) -> None:
        self.strategies = {}
        self.strategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, LIMITS[symbol]) for symbol, clazz in {
            "KELP": KelpStrategy,
            "RAINFOREST_RESIN": ResinStrategy,
            "SQUID_INK": SquidInkStrategy,
            "PICNIC_BASKET1": Basket1Strat,
            "PICNIC_BASKET2": Basket2Strat,
        }.items()}
        self.couponStrategies: dict[Symbol, Strategy] = {symbol: clazz(symbol, LIMITS[symbol]) for symbol, clazz in {
            "VOLCANIC_ROCK_VOUCHER_9500":CouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_9750":CouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_10500":CouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_10250":CouponStrategy,
            "VOLCANIC_ROCK_VOUCHER_10000":CouponStrategy,
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        #trade all except coupons
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

        #trade coupons
        rock_orders=[]
        for symbol, strategy in self.couponStrategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if (symbol in state.order_depths and
                    len(state.order_depths[symbol].buy_orders) > 0 and
                    len(state.order_depths[symbol].sell_orders) > 0
            ):
                strategy_orders, strategy_conversions,under_trade = strategy.run(state,
                                                                     traderObject=old_trader_data.get(symbol, {}), )

                orders[symbol] = strategy_orders
                rock_orders.extend(under_trade)
                
                conversions += sum(strategy_conversions)

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        orders['VOLCANIC_ROCK']=rock_orders

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data