from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        self.mid_price_history = {}

    def compute_fair_price(self, product: str, mid_price: float, window: int = 20) -> float:
        """
        Compute rolling average fair price.
        """
        history = self.mid_price_history.get(product, [])
        history.append(mid_price)
        if len(history) > window:
            history.pop(0)
        self.mid_price_history[product] = history
        return sum(history) / len(history)

    def adjust_order_size(self, position: int, level_offset: int, max_position: int = 50, base_size: int = 10) -> int:
        """
        Scale order size based on inventory and distance from fair value.
        """
        size = base_size * (1 - abs(position / max_position))
        decay = max(0.5, 1.0 - 0.2 * (level_offset - 1))  # Smaller size further out
        return max(1, int(size * decay))

    def place_ladder_orders(self, product: str, fair_price: float, position: int, max_position: int = 50) -> List[Order]:
        """
        Place multiple buy/sell orders at increasing distance from fair price.
        """
        orders = []
        base_size = 10
        max_levels = 3  # number of buy/sell levels

        # BUY ladder (fair - 2, -3, -4, ...)
        if position < max_position:
            for level in range(1, max_levels + 1):
                price = int(fair_price - (level + 1))
                size = self.adjust_order_size(position, level, max_position, base_size)
                orders.append(Order(product, price, size))

        # SELL ladder (fair + 2, +3, +4, ...)
        if position > -max_position:
            for level in range(1, max_levels + 1):
                price = int(fair_price + (level + 1))
                size = self.adjust_order_size(position, level, max_position, base_size)
                orders.append(Order(product, price, -size))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in ["KELP", "RAINFOREST_RESIN"]:
            order_depth: OrderDepth = state.order_depths[product]
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2
            position = state.position.get(product, 0)

            fair_price = self.compute_fair_price(product, mid_price)
            orders = self.place_ladder_orders(product, fair_price, position)

            result[product] = orders

        traderData = "MM_MeanReversion_Ladder"
        conversions = 1
        return result, conversions, traderData
