from datamodel import TradingState, OrderDepth, Order
from typing import List, Dict


class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in state.order_depths.keys():
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            # When there is depth in both sides of the exchange
            if order_depth.buy_orders and order_depth.sell_orders:
                # Pick the highest bid in the exchange which will have most priority
                best_bid = max(order_depth.buy_orders.keys())
                # Pick the lowest ask in the exchange which will have most priority
                best_ask = min(order_depth.sell_orders.keys())
                # Estimate fair price based on the mean of the two
                fair_price = (best_bid + best_ask) / 2
                # Distance from fair price which serves as our delta
                spread_delta = 2

                buy_price = int(fair_price - spread_delta)
                buy_volume = min(10, order_depth.sell_orders.get(best_ask, 0))
                if buy_volume > 0:
                    orders.append(Order(product, buy_price, buy_volume))
                    print(f"[{product}] BUY {buy_volume} @ {buy_price} (Fair={fair_price})")

                sell_price = int(fair_price + spread_delta)
                sell_volume = min(10, order_depth.buy_orders.get(best_bid, 0))
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))
                    print(f"[{product}] SELL {sell_volume} @ {sell_price} (Fair={fair_price})")

            result[product] = orders

        traderData = "DMM"
        conversions = 1
        return result, conversions, traderData
