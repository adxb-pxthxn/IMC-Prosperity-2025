from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

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
    
    def calculate_macd(self, product: str, short_window=12, long_window=26, signal_window=9):
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
                self.track_mid_price(product, mid_price, window=20) # don't need the return value, just need it saved for MACD
                orders = self.squid_ink_momentum(product, order_depth, position)


            result[product] = orders

        traderData = "KELP_Sniper_RESIN_Ladder_SQUID_momentum"
        conversions = 1
        return result, conversions, traderData
