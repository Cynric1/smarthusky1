from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import jsonpickle
import statistics
import json
import math
from typing import Any
import numpy as np

class Product:
    # Original products
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    
    # New products
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

# Basket compositions
BASKET_COMPOSITIONS = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2
    }
}

PARAMS = {
    # Original products - keeping the same parameters
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,  # Based on the data provided
        "take_width": 2,      # Will buy if price is at least 3 below fair value
        "clear_width": 0,     # No additional clearing width
        "disregard_edge": 1,  # Disregard orders within 1 of fair value
        "join_edge": 2,       # Join orders within this edge
        "default_edge": 4,    # Default spread
        "soft_position_limit": 43, # Adjust positioning when approaching limit
    },
    Product.KELP: {
        "take_width": 2,       # Will buy if price is at least 2 below fair value
        "clear_width": 0,      # No additional clearing width
        "prevent_adverse": True, # Prevent trading against large orders
        "adverse_volume": 20,   # Consider orders with volume >= 20 as potentially adverse
        "reversion_beta": -0.3, # Mean reversion factor
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 2,     # Tighter spread for more active trading
        "soft_position_limit": 40,
    },
    Product.SQUID_INK: {
        "take_width": 3,       # Will buy if price is at least 2 below fair value
        "clear_width": 0,      # No additional clearing width
        "prevent_adverse": True, # Prevent trading against large orders
        "adverse_volume": 10,   # Consider orders with volume >= 20 as potentially adverse
        "ou_theta": 0.34,       # OU mean reversion speed
        "ou_mu": 1960,         # OU mean reversion level
        "ou_sigma": 50,        # OU volatility
        "momentum_lookback": 3, # Momentum lookback window
        "momentum_weight": 0.77, # Weight of momentum component (OU weight = 1 - momentum_weight)
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 2,     # Tighter spread for more active trading
        "soft_position_limit": 40,
    },
    
    # New products
    Product.CROISSANTS: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 20,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 2,
        "soft_position_limit": 200,
    },
    Product.JAMS: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 20,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 2,
        "soft_position_limit": 300,
    },
    Product.DJEMBES: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 3,
        "soft_position_limit": 50,
    },
    Product.PICNIC_BASKET1: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 3,
        "soft_position_limit": 50,
    },
    Product.PICNIC_BASKET2: {
        "take_width": 2,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "disregard_edge": 1,
        "join_edge": 1,
        "default_edge": 3,
        "soft_position_limit": 80,
    },
    
    # Parameters for basket arbitrage
    "basket_arb": {
        "lookback_period": 20,  # Number of periods to calculate spread
        "spread_factor": 0.3,   # Factor to multiply std dev by for spread
        "min_history": 3,       # Minimum history required before trading
    },
}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[str, list[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_listings(self, listings: dict[str, Any]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[str, OrderDepth]) -> dict[str, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[str, list[Any]]) -> list[list[Any]]:
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

    def compress_observations(self, observations: Any) -> list[Any]:
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

    def compress_orders(self, orders: dict[str, list[Order]]) -> list[list[Any]]:
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

class ProsperityEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, dict):
            return {key: self.default(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self.default(item) for item in obj]
        if hasattr(obj, "__dict__"):
            return self.default(vars(obj))
        return obj

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        
        # Position limits for each product
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50, 
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100
        }
        
        # Initialize logger
        self.logger = Logger()

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the mid price from an order depth"""
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        return None

    def is_tradable(self, product, position, quantity):
        """Check if a trade is within position limits"""
        if quantity > 0:  # Buy
            return position + quantity <= self.LIMIT[product]
        else:  # Sell
            return position + quantity >= -self.LIMIT[product]

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
        """Take advantage of mispriced orders in the market"""
        position_limit = self.LIMIT[product]

        # Buy underpriced asks
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            # Check if this isn't an adversely large order (if we care)
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    # This is a good buy opportunity
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt we can buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # Sell overpriced bids
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            # Check if this isn't an adversely large order (if we care)
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    # This is a good sell opportunity
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # max amt we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

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
        """Place market making orders at the specified bid and ask prices"""
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

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
    ) -> (int, int):
        """Place orders to clear positions at acceptable prices"""
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        # If we're long, try to sell at acceptable prices
        if position_after_take > 0:
            # Find bids above our fair ask price
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # If we're short, try to buy at acceptable prices
        if position_after_take < 0:
            # Find asks below our fair bid price
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def calculate_fair_value(self, product: str, order_depth: OrderDepth, traderObject) -> float:
        """Calculate fair value for Kelp or Squid Ink based on order book and historical data"""
        if product not in [Product.KELP, Product.SQUID_INK]:
            return None
            
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            # Filter out large orders that might be adversely selecting
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[product]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[product]["adverse_volume"]
            ]
            
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            
            # Calculate mid price
            product_key = f"{product.lower()}_last_price"
            if mm_ask is None or mm_bid is None:
                if traderObject.get(product_key, None) is None:
                    mid_price = (best_ask + best_bid) / 2
                else:
                    mid_price = traderObject[product_key]
            else:
                mid_price = (mm_ask + mm_bid) / 2

            # For Kelp, use the original mean reversion model
            if product == Product.KELP:
                if traderObject.get(product_key, None) is not None:
                    last_price = traderObject[product_key]
                    last_returns = (mid_price - last_price) / last_price
                    pred_returns = (
                        last_returns * self.params[product]["reversion_beta"]
                    )
                    fair = mid_price + (mid_price * pred_returns)
                else:
                    fair = mid_price
            
            # For Squid Ink, use combined OU process and momentum model
            elif product == Product.SQUID_INK:
                # Store the current mid price
                if traderObject.get(product_key) is None:
                    traderObject[product_key] = mid_price
                
                # Add to price history
                prices_key = f"{product.lower()}_prices"
                prices = traderObject.get(prices_key, [])
                prices.append(mid_price)
                
                # Keep a limited history
                max_history = max(10, self.params[product]["momentum_lookback"] + 2)
                if len(prices) > max_history:
                    prices = prices[-max_history:]
                traderObject[prices_key] = prices
                
                # Calculate OU process prediction
                ou_theta = self.params[product]["ou_theta"]  # Mean reversion speed
                ou_mu = self.params[product]["ou_mu"]        # Long-term mean
                ou_sigma = self.params[product]["ou_sigma"]  # Volatility
                
                # OU process: dX = θ(μ - X)dt + σdW
                # Discretized: X_t+1 = X_t + θ(μ - X_t) + σ√dt * ε
                # For our implementation, dt = 1 and ε is ignored for prediction
                last_price = traderObject[product_key]
                ou_prediction = last_price + ou_theta * (ou_mu - last_price)
                
                # Calculate momentum component if we have enough history
                momentum_prediction = mid_price  # Default to current price
                if len(prices) >= self.params[product]["momentum_lookback"] + 1:
                    # Calculate recent price trend
                    recent_returns = []
                    for i in range(1, self.params[product]["momentum_lookback"] + 1):
                        ret = (prices[-i] - prices[-(i+1)]) / prices[-(i+1)]
                        recent_returns.append(ret)
                    
                    # Average recent returns
                    avg_return = sum(recent_returns) / len(recent_returns)
                    
                    # Project forward based on momentum
                    momentum_prediction = mid_price * (1 + avg_return)
                
                # Combine OU and momentum predictions with respective weights
                momentum_weight = self.params[product]["momentum_weight"]
                ou_weight = 1 - momentum_weight
                
                fair = (ou_weight * ou_prediction) + (momentum_weight * momentum_prediction)
                
                # Log some debugging info
                self.logger.print(
                    f"SQUID_INK - Mid: {mid_price:.2f}, OU: {ou_prediction:.2f}, "
                    f"Momentum: {momentum_prediction:.2f}, Fair: {fair:.2f}"
                )
            
            else:
                # Fallback for any other product (shouldn't reach here)
                fair = mid_price
                
            traderObject[product_key] = mid_price
            return fair
        return None

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
        """Take advantageous orders from the market"""
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
        """Place orders to clear our position"""
        orders: List[Order] = []
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

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        """Create market making orders based on the current state"""
        orders: List[Order] = []
        
        # Find asks above fair value (by more than disregard_edge)
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        
        # Find bids below fair value (by more than disregard_edge)
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        # Set ask price based on existing orders
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join the existing ask
            else:
                ask = best_ask_above_fair - 1  # penny the existing ask

        # Set bid price based on existing orders
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair  # join the existing bid
            else:
                bid = best_bid_below_fair + 1  # penny the existing bid

        # Adjust prices based on our position
        if manage_position:
            if position > soft_position_limit:
                ask -= 1  # Lower ask to encourage selling when we're long
            elif position < -1 * soft_position_limit:
                bid += 1  # Raise bid to encourage buying when we're short

        # Place market making orders
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

    def get_basket_mid_prices(self, state, traderObject):
        """Calculate and store mid prices for baskets and their components"""
        # Store mid prices for all products in the baskets
        for basket, components in BASKET_COMPOSITIONS.items():
            # Calculate basket mid price
            if basket in state.order_depths:
                basket_mid = self.calculate_mid_price(state.order_depths[basket])
                if basket_mid:
                    basket_key = f"{basket.lower()}_mid"
                    traderObject[basket_key] = basket_mid
            
            # Calculate component mid prices
            for component in components:
                if component in state.order_depths:
                    component_mid = self.calculate_mid_price(state.order_depths[component])
                    if component_mid:
                        component_key = f"{component.lower()}_mid"
                        traderObject[component_key] = component_mid
        
        # Calculate spread between basket and components
        for basket, components in BASKET_COMPOSITIONS.items():
            basket_key = f"{basket.lower()}_mid"
            
            if basket_key in traderObject:
                basket_price = traderObject[basket_key]
                components_price = 0
                
                # Calculate weighted sum of components
                all_components_available = True
                for component, quantity in components.items():
                    component_key = f"{component.lower()}_mid"
                    if component_key in traderObject:
                        components_price += traderObject[component_key] * quantity
                    else:
                        all_components_available = False
                        break
                
                # If we have all prices, calculate and store the spread
                if all_components_available:
                    spread = basket_price - components_price
                    spread_key = f"{basket.lower()}_spread"
                    
                    # Store the current spread
                    spreads_history_key = f"{basket.lower()}_spreads"
                    spreads_history = traderObject.get(spreads_history_key, [])
                    spreads_history.append(spread)
                    
                    # Keep only the most recent history
                    lookback = self.params["basket_arb"]["lookback_period"]
                    if len(spreads_history) > lookback:
                        spreads_history = spreads_history[-lookback:]
                    
                    traderObject[spreads_history_key] = spreads_history
                    traderObject[spread_key] = spread
                    
                    self.logger.print(f"{basket} Spread: {spread:.2f}, History length: {len(spreads_history)}")

    def basket_arbitrage(self, state, traderObject, result):
        """Execute basket arbitrage strategy"""
        all_basket_orders = {}
        
        for basket, components in BASKET_COMPOSITIONS.items():
            # Skip if we don't have all required order depths
            if basket not in state.order_depths:
                continue
                
            # Skip if any component is missing
            component_missing = False
            for component in components:
                if component not in state.order_depths:
                    component_missing = True
                    break
            if component_missing:
                continue
            
            # Get current positions
            basket_position = state.position.get(basket, 0)
            component_positions = {
                component: state.position.get(component, 0)
                for component in components
            }
            
            # Get current spread
            spread_key = f"{basket.lower()}_spread"
            current_spread = traderObject.get(spread_key)
            
            if current_spread is None:
                continue
                
            self.logger.print(f"{basket} - Current spread: {current_spread:.2f}")
            
            # Initialize orders for this product
            basket_orders = []
            component_orders = {comp: [] for comp in components}
            
            # Check if the current spread meets our fixed thresholds
            if current_spread > 75:
                # Spread is high enough: sell basket, buy components
                # Check if we can execute this trade (position limits)
                
                # First check basket position
                can_sell_basket = basket_position < self.LIMIT[basket]
                
                # Then check all component positions
                can_buy_all_components = True
                for component, quantity in components.items():
                    if component_positions[component] + quantity > self.LIMIT[component]:
                        can_buy_all_components = False
                        break
                
                if can_sell_basket and can_buy_all_components:
                    # Find the best price to execute the basket trade
                    if len(state.order_depths[basket].buy_orders) > 0:
                        best_basket_bid = max(state.order_depths[basket].buy_orders.keys())
                        basket_bid_volume = state.order_depths[basket].buy_orders[best_basket_bid]
                        
                        # Sell 1 basket
                        if basket_bid_volume > 0:
                            basket_orders.append(Order(basket, best_basket_bid, -1))
                            
                            # Buy corresponding components
                            for component, quantity in components.items():
                                if len(state.order_depths[component].sell_orders) > 0:
                                    best_component_ask = min(state.order_depths[component].sell_orders.keys())
                                    component_ask_volume = -state.order_depths[component].sell_orders[best_component_ask]
                                    
                                    if component_ask_volume >= quantity:
                                        component_orders[component].append(Order(component, best_component_ask, quantity))
            
            elif current_spread < -30:
                # Spread is low enough: buy basket, sell components
                # Check if we can execute this trade (position limits)
                
                # First check basket position
                can_buy_basket = basket_position + 1 <= self.LIMIT[basket]
                
                # Then check all component positions
                can_sell_all_components = True
                for component, quantity in components.items():
                    if component_positions[component] - quantity < -self.LIMIT[component]:
                        can_sell_all_components = False
                        break
                
                if can_buy_basket and can_sell_all_components:
                    # Find the best price to execute the basket trade
                    if len(state.order_depths[basket].sell_orders) > 0:
                        best_basket_ask = min(state.order_depths[basket].sell_orders.keys())
                        basket_ask_volume = -state.order_depths[basket].sell_orders[best_basket_ask]
                        
                        # Buy 1 basket
                        if basket_ask_volume > 0:
                            basket_orders.append(Order(basket, best_basket_ask, 1))
                            
                            # Sell corresponding components
                            for component, quantity in components.items():
                                if len(state.order_depths[component].buy_orders) > 0:
                                    best_component_bid = max(state.order_depths[component].buy_orders.keys())
                                    component_bid_volume = state.order_depths[component].buy_orders[best_component_bid]
                                    
                                    if component_bid_volume >= quantity:
                                        component_orders[component].append(Order(component, best_component_bid, -quantity))
            
            # Add orders to the result
            if basket_orders:
                # Check if all component orders are valid
                all_components_valid = True
                for component, orders in component_orders.items():
                    if len(orders) == 0:
                        all_components_valid = False
                        break
                
                # Only execute the entire basket trade if all component trades are valid
                if all_components_valid:
                    # Add basket order
                    if basket not in result:
                        result[basket] = []
                    result[basket].extend(basket_orders)
                    
                    # Add component orders
                    for component, orders in component_orders.items():
                        if component not in result:
                            result[component] = []
                        result[component].extend(orders)
                    
                    self.logger.print(f"Executed {basket} arbitrage trade!")
    
    def run(self, state: TradingState):
        """Main entry point for the trading algorithm"""
        traderObject = {}
        if state.traderData and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except:
                pass  # If we can't decode, use empty dict

        result = {}
        
        # Update basket mid prices and spreads
        self.get_basket_mid_prices(state, traderObject)
        
        # Execute basket arbitrage strategy
        self.basket_arbitrage(state, traderObject, result)

        # Trade RAINFOREST_RESIN (stable product)
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            
            # Take advantageous orders
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                resin_position,
            )
            
            # Clear position if needed
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
            )
            
            # Place market making orders
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            
            # Combine all orders for this product
            if Product.RAINFOREST_RESIN not in result:
                result[Product.RAINFOREST_RESIN] = []
            result[Product.RAINFOREST_RESIN].extend(resin_take_orders + resin_clear_orders + resin_make_orders)

        # Trade KELP (fluctuating product)
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            
            # Calculate fair value for Kelp based on current market
            kelp_fair_value = self.calculate_fair_value(
                Product.KELP,
                state.order_depths[Product.KELP], 
                traderObject
            )
            
            if kelp_fair_value is None:
                # If we can't determine fair value, use mid price from order book
                order_depth = state.order_depths[Product.KELP]
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    kelp_fair_value = (best_ask + best_bid) / 2
                else:
                    kelp_fair_value = 2026  # Fallback value based on data
            
            # Take advantageous orders
            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["take_width"],
                kelp_position,
                self.params[Product.KELP]["prevent_adverse"],
                self.params[Product.KELP]["adverse_volume"],
            )
            
            # Clear position if needed
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["clear_width"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            
            # Place market making orders
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
                True,
                self.params[Product.KELP]["soft_position_limit"],
            )
            
            # Combine all orders for this product
            if Product.KELP not in result:
                result[Product.KELP] = []
            result[Product.KELP].extend(kelp_take_orders + kelp_clear_orders + kelp_make_orders)
            
        # Trade SQUID_INK (fluctuating product with patterns)
        if Product.SQUID_INK in state.order_depths:
            squid_ink_position = state.position.get(Product.SQUID_INK, 0)
            
            # Calculate fair value for Squid Ink based on current market
            squid_ink_fair_value = self.calculate_fair_value(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK], 
                traderObject
            )
            
            if squid_ink_fair_value is None:
                # If we can't determine fair value, use mid price from order book
                order_depth = state.order_depths[Product.SQUID_INK]
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    squid_ink_fair_value = (best_ask + best_bid) / 2
                else:
                    squid_ink_fair_value = 2500  # Fallback value for Squid Ink
            
            # Take advantageous orders
            squid_ink_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                self.params[Product.SQUID_INK]["take_width"],
                squid_ink_position,
                self.params[Product.SQUID_INK]["prevent_adverse"],
                self.params[Product.SQUID_INK]["adverse_volume"],
            )
            
            # Clear position if needed
            squid_ink_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                self.params[Product.SQUID_INK]["clear_width"],
                squid_ink_position,
                buy_order_volume,
                sell_order_volume,
            )
            
            # Place market making orders
            squid_ink_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                squid_ink_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
                True,
                self.params[Product.SQUID_INK]["soft_position_limit"],
            )
            
            # Combine all orders for this product
            if Product.SQUID_INK not in result:
                result[Product.SQUID_INK] = []
            result[Product.SQUID_INK].extend(squid_ink_take_orders + squid_ink_clear_orders + squid_ink_make_orders)
        
        # Trade new individual products (CROISSANTS, JAMS, DJEMBES)
        for product in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                
                # For new products, use simple mid price as fair value
                order_depth = state.order_depths[product]
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    fair_value = (best_ask + best_bid) / 2
                else:
                    # Default values if we can't determine mid price
                    if product == Product.CROISSANTS:
                        fair_value = 100
                    elif product == Product.JAMS:
                        fair_value = 75
                    else:  # DJEMBES
                        fair_value = 200
                
                # Take advantageous orders
                take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    product,
                    state.order_depths[product],
                    fair_value,
                    self.params[product]["take_width"],
                    position,
                    self.params[product]["prevent_adverse"],
                    self.params[product]["adverse_volume"],
                )
                
                # Clear position if needed
                clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    product,
                    state.order_depths[product],
                    fair_value,
                    self.params[product]["clear_width"],
                    position,
                    buy_order_volume,
                    sell_order_volume,
                )
                
                # Place market making orders
                make_orders, _, _ = self.make_orders(
                    product,
                    state.order_depths[product],
                    fair_value,
                    position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[product]["disregard_edge"],
                    self.params[product]["join_edge"],
                    self.params[product]["default_edge"],
                    True,
                    self.params[product]["soft_position_limit"],
                )
                
                # Combine all orders for this product
                if product not in result:
                    result[product] = []
                result[product].extend(take_orders + clear_orders + make_orders)
        
        # Trade picnic baskets (when not already handled by arbitrage)
        for product in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
            if product in state.order_depths and product not in result:
                position = state.position.get(product, 0)
                
                # For picnic baskets, use simple mid price as fair value
                order_depth = state.order_depths[product]
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    fair_value = (best_ask + best_bid) / 2
                else:
                    # Default values if we can't determine mid price
                    if product == Product.PICNIC_BASKET1:
                        fair_value = 1000  # Rough estimate
                    else:  # PICNIC_BASKET2
                        fair_value = 800   # Rough estimate
                
                # Take advantageous orders
                take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    product,
                    state.order_depths[product],
                    fair_value,
                    self.params[product]["take_width"],
                    position,
                    self.params[product]["prevent_adverse"],
                    self.params[product]["adverse_volume"],
                )
                
                # Clear position if needed
                clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    product,
                    state.order_depths[product],
                    fair_value,
                    self.params[product]["clear_width"],
                    position,
                    buy_order_volume,
                    sell_order_volume,
                )
                
                # Place market making orders
                make_orders, _, _ = self.make_orders(
                    product,
                    state.order_depths[product],
                    fair_value,
                    position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[product]["disregard_edge"],
                    self.params[product]["join_edge"],
                    self.params[product]["default_edge"],
                    True,
                    self.params[product]["soft_position_limit"],
                )
                
                # Combine all orders for this product
                if product not in result:
                    result[product] = []
                result[product].extend(take_orders + clear_orders + make_orders)

        # No conversions needed in this scenario
        conversions = 0
        
        # Save our state for next run
        traderData = jsonpickle.encode(traderObject)

        # Log the trading activity
        self.logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData