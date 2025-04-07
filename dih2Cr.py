from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        self.mid_price_history = {}

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
            offset = 2 + (level - 1)
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

    # === Main Bot ===
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths:
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

            result[product] = orders

        traderData = "KELP_Sniper_RESIN_Ladder"
        conversions = 1
        return result, conversions, traderData
