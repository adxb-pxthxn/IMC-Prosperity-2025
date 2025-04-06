from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        self.mid_price_history = {}

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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            position = state.position.get(product, 0)

            if product == "KELP":
                fair_price = self.track_mid_price(product, mid_price, window=10)
                config = {
                    "start_offset": 3,
                    "levels": 2,
                    "max_position": 50,
                    "base_size": 10,
                }
                orders = self.place_ladder_orders(product, fair_price, position, config)

            elif product == "RAINFOREST_RESIN":
                fair_price = self.track_mid_price(product, mid_price, window=20)
                config = {
                    "start_offset": 2,
                    "levels": 3,
                    "max_position": 50,
                    "base_size": 10,
                }
                orders = self.place_ladder_orders(product, fair_price, position, config)

            result[product] = orders

        traderData = "Hybrid_Refined_KELP_Resin"
        conversions = 1
        return result, conversions, traderData
