from typing import Dict, List,Any
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import math
import numpy as np

KELP  = "KELP"
RAINFOREST_RESIN = "RAINFOREST_RESIN"
SUBMISSION = "SUBMISSION"
PRODUCTS = [KELP, RAINFOREST_RESIN]


DEFAULT_PRICES = {
    RAINFOREST_RESIN : 10_000,
    KELP : 2000,
}


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
    def __init__(self, hyperparam = 0.04):
        self.mid_price_history = {}
        self.round = 0

        self.position_limit = {
            RAINFOREST_RESIN : 50,
            KELP : 50,
        }

        # Values to compute pnl
        self.cash = 0
        # positions can be obtained from state.position
        
        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None

        self.ema_param = hyperparam

    def track_mid_price(self, product: str, mid_price: float, window: int) -> float:
        history = self.mid_price_history.get(product, [])
        history.append(mid_price)
        if len(history) > window:
            history.pop(0)
        self.mid_price_history[product] = history
        return sum(history) / len(history)

    def adjust_order_size(self, position: int, level_offset: int = 1, max_position: int = 50, base_size: int = 10) -> int:
        size = base_size * (1 - abs(position / max_position))
        decay = max(0.5, 1.0 - 0.3 * (level_offset - 1))
        return max(1, int(size * decay))

    def place_ladder_orders(self, product: str, fair_price: float, position: int, config: Dict) -> List[Order]:
        """
        Place ladder orders using given config: {start_offset, levels, max_position, base_size}
        """
        orders = []
        start_offset = config["start_offset"]
        levels = config["levels"]
        max_pos = config["max_position"]
        base_size = config["base_size"]

        for level in range(1, levels + 1):
            offset = start_offset + level - 1

            if position < max_pos:
                buy_price = int(fair_price - offset)
                size = self.adjust_order_size(position, level, max_pos, base_size)
                orders.append(Order(product, buy_price, size))

            if position > -max_pos:
                sell_price = int(fair_price + offset)
                size = self.adjust_order_size(position, level, max_pos, base_size)
                orders.append(Order(product, sell_price, -size))

        return orders
    def rain_order(self,order_depth:OrderDepth,fair=10000,position=0,position_limit=50):

        orders=[]

        buy_volume=0
        sell_volume=0


        sell_orders=order_depth.sell_orders
        buy_orders=order_depth.buy_orders
        try:
            best_ask_fair = min([p for p in sell_orders.keys() if p > fair+1], default=fair+2)
        except ValueError:
            best_ask_fair = fair+1
            
        try:
            best_bid_fair = max([p for p in buy_orders.keys() if p < fair+1], default=fair-2)
        except ValueError:
            best_bid_fair = fair-1

        if sell_orders:
            best_ask=min(sell_orders.keys())
            best_ask_ammount=-sell_orders[best_ask]
            if best_ask<fair:
                quant=min(best_ask_ammount,position_limit-position)
                if quant>0:
                    orders.append(Order("RAINFOREST_RESIN",best_ask,quant))
                    buy_volume+=quant
        if buy_orders:
            best_bid = max(buy_orders.keys())
            best_bid_amount = buy_orders[best_bid]
            if best_bid > fair:
                quant = min(best_bid_amount, position_limit + position)
                if quant > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -quant))
                    sell_volume += quant
        
        buy_quant=position_limit-(position+buy_volume)
        if buy_quant>0:
            orders.append(Order("RAINFOREST_RESIN",  best_bid_fair+ 1, buy_quant))


        sell_quant=position_limit+(position-sell_volume)
        if sell_quant>0:
            orders.append(Order("RAINFOREST_RESIN", best_ask_fair - 1, -sell_quant))


        return orders


    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0) 

    def KELP_strategy(self, state : TradingState):
        """
        Returns a list of orders with trades of KELP.

        Comment: Mudar depois. Separar estrategia por produto assume que
        cada produto eh tradado independentemente
        """

        position_KELP = self.get_position(KELP, state)

        bid_volume = self.position_limit[KELP] - position_KELP
        ask_volume = - self.position_limit[KELP] - position_KELP

        orders = []

        if position_KELP == 0:
            # Not long nor short
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP] - 1), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP] + 1), ask_volume))
        
        if position_KELP > 0:
            # Long position
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP] - 2), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP]), ask_volume))

        if position_KELP < 0:
            # Short position
            orders.append(Order(KELP, math.floor(self.ema_prices[KELP]), bid_volume))
            orders.append(Order(KELP, math.ceil(self.ema_prices[KELP] + 2), ask_volume))

        return orders
    
    def get_value_on_product(self, product, state : TradingState):
        """
        Returns the amount of MONEY currently held on the product.  
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)

    def update_pnl(self, state : TradingState):
        """
        Updates the pnl.
        """
        def update_cash():
            # Update cash
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        # Trade was already analyzed
                        continue

                    if trade.buyer == SUBMISSION:
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == SUBMISSION:
                        self.cash += trade.quantity * trade.price
        
        def get_value_on_positions():
            value = 0
            for product in state.position:
                value += self.get_value_on_product(product, state)
            return value
        
        # Update cash
        update_cash()
        return self.cash + get_value_on_positions()
    
    def update_ema_prices(self, state : TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]
    def get_mid_price(self, product, state : TradingState):

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price
        
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)


        try:
            result[KELP] = self.KELP_strategy(state)
        except Exception as e:
            print("Error in KELP strategy")
            print(e)


        if 'RAINFOREST_RESIN' in state.order_depths:
            rain_position = state.position.get('RAINFOREST_RESIN',0)
            rain_orders=self.rain_order(state.order_depths['RAINFOREST_RESIN'],position=rain_position)
            result['RAINFOREST_RESIN']=rain_orders
        
        # if 'KELP' in state.order_depths:
            
        #     kelp_position = state.position.get('RAINFOREST_RESIN',0)
        #     fair_price = self.track_mid_price('KELP', ((max(state.order_depths.buy_orders.keys())+min(state.order_depths.sell_orders.keys()))/2), window=10)
        #     rain_orders=self.kelp_order(state.order_depths['RAINFOREST_RESIN'],fair_priceposition=kelp_position)
        #     result['KELP']=rain_orders



        traderData = "MM_MeanReversion_Ladder"
        conversions = 1


        logger.flush(state, result, conversions, traderData) #this is necessary for visualiser
        return result, conversions, traderData