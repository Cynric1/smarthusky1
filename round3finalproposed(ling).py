from statistics import NormalDist
import json
import numpy as np
from datamodel import Order, OrderDepth, TradingState
from typing import Any, List, Dict, Tuple
import jsonpickle
import math
from collections import deque

class Product:
    # Original products
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    
    # Basket products
    CROISSANT = "CROISSANTS"
    JAM = "JAMS"
    DJEMBE = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    
    # New volcanic products
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

# Basket compositions
BASKET_COMPOSITIONS = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANT: 6,
        Product.JAM: 3,
        Product.DJEMBE: 1
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANT: 4,
        Product.JAM: 2
    }
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

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders = []
        self.conversions = 0

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0
        
        self.act(state)
        
        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def act(self, state: TradingState) -> None:
        raise NotImplementedError()
        
    def save(self):
        return None
        
    def load(self, data):
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        
        self.window = deque()
        self.window_size = 10

    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0] if buy_orders else true_value - 2
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0] if sell_orders else true_value + 2
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self):
        return list(self.window)

    def load(self, data):
        self.window = deque(data)

class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10000

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return 2026  # Fallback value

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)

class SquidInkStrategy(MarketMakingStrategy):
    """Enhanced Squid Ink Strategy with Ornstein-Uhlenbeck process and momentum"""
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        
        # Parameters for the OU process and momentum
        self.params = {
            "take_width": 3,           # Will buy if price is at least 3 below fair value
            "clear_width": 0,          # No additional clearing width
            "prevent_adverse": True,   # Prevent trading against large orders
            "adverse_volume": 10,      # Consider orders with volume >= 10 as potentially adverse
            "ou_theta": 0.34,          # OU mean reversion speed
            "ou_mu": 1780,             # OU mean reversion level
            "ou_sigma": 50,            # OU volatility
            "momentum_lookback": 3,    # Momentum lookback window
            "momentum_weight": 0.77,   # Weight of momentum component (OU weight = 1 - momentum_weight)
            "disregard_edge": 1,
            "join_edge": 1,
            "default_edge": 2,         # Tighter spread for more active trading
            "soft_position_limit": 40,
        }
        
        # Price history for the calculations
        self.price_history = []
        self.max_history_size = 20  # Keep track of last 20 prices
        
    def get_true_value(self, state: TradingState) -> int:
        """Calculate fair value for Squid Ink using OU process and momentum"""
        order_depth = state.order_depths[self.symbol]
        
        # If no orders, use default value
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 2500  # Fallback value
        
        # Get current mid price
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # Filter out potentially adverse orders
        filtered_ask_prices = [
            price for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price]) >= self.params["adverse_volume"]
        ]
        
        filtered_bid_prices = [
            price for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price]) >= self.params["adverse_volume"]
        ]
        
        mm_ask = min(filtered_ask_prices) if filtered_ask_prices else None
        mm_bid = max(filtered_bid_prices) if filtered_bid_prices else None
        
        # Use filtered mid price if available
        if mm_ask is not None and mm_bid is not None:
            mid_price = (mm_ask + mm_bid) / 2
        
        # Add to price history
        self.price_history.append(mid_price)
        if len(self.price_history) > self.max_history_size:
            self.price_history = self.price_history[-self.max_history_size:]
        
        # If we don't have enough history, return the mid price
        if len(self.price_history) < 2:
            return round(mid_price)
        
        # Calculate OU process prediction
        last_price = self.price_history[-2]  # Previous price
        ou_theta = self.params["ou_theta"]   # Mean reversion speed
        ou_mu = self.params["ou_mu"]         # Long-term mean
        
        # OU process: dX = θ(μ - X)dt + σdW
        # Discretized: X_t+1 = X_t + θ(μ - X_t)
        ou_prediction = last_price + ou_theta * (ou_mu - last_price)
        
        # Calculate momentum component if we have enough history
        momentum_prediction = mid_price  # Default to current price
        if len(self.price_history) >= self.params["momentum_lookback"] + 1:
            # Calculate recent price trend
            recent_returns = []
            for i in range(1, self.params["momentum_lookback"] + 1):
                ret = (self.price_history[-i] - self.price_history[-(i+1)]) / self.price_history[-(i+1)]
                recent_returns.append(ret)
            
            # Average recent returns
            avg_return = sum(recent_returns) / len(recent_returns)
            
            # Project forward based on momentum
            momentum_prediction = mid_price * (1 + avg_return)
        
        # Combine OU and momentum predictions with respective weights
        momentum_weight = self.params["momentum_weight"]
        ou_weight = 1 - momentum_weight
        
        fair_value = (ou_weight * ou_prediction) + (momentum_weight * momentum_prediction)
        
        return round(fair_value)
        
    def save(self):
        """Save the state for next iteration"""
        data = {
            'window': list(self.window),
            'price_history': self.price_history
        }
        return data
        
    def load(self, data):
        """Load the state from previous iteration"""
        if isinstance(data, list):
            # Old format - just the window
            self.window = deque(data)
        elif isinstance(data, dict):
            # New format with price history
            if 'window' in data:
                self.window = deque(data['window'])
            if 'price_history' in data:
                self.price_history = data['price_history']
                
class PicnicBasketStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, trader_data_key: str = None) -> None:
        super().__init__(symbol, limit)
        self.trader_data_key = trader_data_key or symbol
        
        # Set thresholds based on which basket we're trading
        if symbol == Product.PICNIC_BASKET1:
            self.long_threshold = -20
            self.short_threshold = 170
        elif symbol == Product.PICNIC_BASKET2:
            self.long_threshold = -20
            self.short_threshold = 140
        
        # Components of each basket
        self.components = BASKET_COMPOSITIONS[symbol]
    
    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        """Calculate mid price from order book"""
        if symbol not in state.order_depths:
            return None
            
        order_depth = state.order_depths[symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        return (best_bid + best_ask) / 2
    
    def go_long(self, state: TradingState) -> None:
        """Create orders to buy basket and sell components"""
        # First, check if we can execute this trade
        if self.symbol not in state.order_depths:
            return
            
        # Get basket order depth
        order_depth = state.order_depths[self.symbol]
        if not order_depth.sell_orders:
            return
            
        # Get basket best ask price (to buy)
        basket_price = min(order_depth.sell_orders.keys())
        
        # Check position limit for basket
        position = state.position.get(self.symbol, 0)
        max_basket_buy = self.limit - position
        
        if max_basket_buy <= 0:
            return
            
        # Check position limits for components
        max_trade_size = max_basket_buy
        for component, quantity in self.components.items():
            if component not in state.order_depths:
                return
                
            # Get component bid price (to sell)
            component_order_depth = state.order_depths[component]
            if not component_order_depth.buy_orders:
                return
                
            component_position = state.position.get(component, 0)
            component_limit = self._get_component_limit(component)
            max_component_sell = component_limit + component_position
            
            # Calculate max baskets we can trade based on this component
            max_baskets_for_component = max_component_sell // quantity
            max_trade_size = min(max_trade_size, max_baskets_for_component)
        
        # Apply halving 
        trade_size = max(1, max_trade_size//2)
        
        if trade_size <= 0:
            return
            
        # Create basket buy order
        self.buy(basket_price, trade_size)
        
        # Create component sell orders
        for component, quantity in self.components.items():
            component_bid_price = max(state.order_depths[component].buy_orders.keys())
            component_quantity = quantity * trade_size
            
            # Order needs to be added to result
            self.component_orders[component].append(
                Order(component, component_bid_price, -component_quantity)
            )
    
    def go_short(self, state: TradingState) -> None:
        """Create orders to sell basket and buy components"""
        # First, check if we can execute this trade
        if self.symbol not in state.order_depths:
            return
            
        # Get basket order depth
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders:
            return
            
        # Get basket best bid price (to sell)
        basket_price = max(order_depth.buy_orders.keys())
        
        # Check position limit for basket
        position = state.position.get(self.symbol, 0)
        max_basket_sell = self.limit + position
        
        if max_basket_sell <= 0:
            return
            
        # Check position limits for components
        max_trade_size = max_basket_sell
        for component, quantity in self.components.items():
            if component not in state.order_depths:
                return
                
            # Get component ask price (to buy)
            component_order_depth = state.order_depths[component]
            if not component_order_depth.sell_orders:
                return
                
            component_position = state.position.get(component, 0)
            component_limit = self._get_component_limit(component)
            max_component_buy = component_limit - component_position
            
            # Calculate max baskets we can trade based on this component
            max_baskets_for_component = max_component_buy // quantity
            max_trade_size = min(max_trade_size, max_baskets_for_component)
        
        # Apply halving
        trade_size = max(1, max_trade_size)
        
        if trade_size <= 0:
            return
            
        # Create basket sell order
        self.sell(basket_price, trade_size)
        
        # Create component buy orders
        for component, quantity in self.components.items():
            component_ask_price = min(state.order_depths[component].sell_orders.keys())
            component_quantity = quantity * trade_size
            
            # Order needs to be added to result
            self.component_orders[component].append(
                Order(component, component_ask_price, component_quantity)
            )
    
    def _get_component_limit(self, component: str) -> int:
        """Get position limit for a component"""
        component_limits = {
            Product.CROISSANT: 250,
            Product.JAM: 350,
            Product.DJEMBE: 60
        }
        return component_limits.get(component, 100)  # Default to 100 if not specified
    
    def act(self, state: TradingState) -> None:
        """Implement basket arbitrage strategy"""
        # Initialize component orders dictionary
        self.component_orders = {component: [] for component in self.components}
        
        # Check if all needed symbols are in order_depths
        required_symbols = list(self.components.keys()) + [self.symbol]
        if any(symbol not in state.order_depths for symbol in required_symbols):
            return
        
        # Get mid prices
        basket_price = self.get_mid_price(state, self.symbol)
        if basket_price is None:
            return
            
        component_prices = {}
        for component in self.components:
            price = self.get_mid_price(state, component)
            if price is None:
                return
            component_prices[component] = price
        
        # Calculate basket theoretical value
        theoretical_value = sum(component_prices[component] * quantity 
                               for component, quantity in self.components.items())
        
        # Calculate price difference
        diff = basket_price - theoretical_value
        
        # Log the spread
        print(f"{self.symbol} Price: {basket_price}, Theoretical: {theoretical_value}, Diff: {diff}")
        
        # Trade based on price differential
        if diff < self.long_threshold:
            # Basket is underpriced compared to components - BUY basket, SELL components
            self.go_long(state)
        elif diff > self.short_threshold:
            # Basket is overpriced compared to components - SELL basket, BUY components
            self.go_short(state)
        
        # Return component orders
        return self.component_orders

class RegressionHedgingStrategy:
    """
    A strategy that uses linear regression to determine optimal hedge ratios
    between baskets and their components, helping to reduce overall position exposure.
    """
    def __init__(self, state_history_length=30):
        # Maximum history length for price data
        self.state_history_length = state_history_length
        
        # Price history for regression
        self.price_history = {
            "CROISSANTS": [],
            "JAMS": [],
            "DJEMBES": [],
            "PICNIC_BASKET1": [],
            "PICNIC_BASKET2": []
        }
        
        # Regression coefficients for hedging
        self.hedge_coefficients = {
            "PICNIC_BASKET1": {
                "CROISSANTS": None,
                "JAMS": None,
                "DJEMBES": None
            },
            "PICNIC_BASKET2": {
                "CROISSANTS": None,
                "JAMS": None
            }
        }
        
        # Default basket compositions (used as fallback)
        self.default_compositions = {
            "PICNIC_BASKET1": {
                "CROISSANTS": 6,
                "JAMS": 3,
                "DJEMBES": 1
            },
            "PICNIC_BASKET2": {
                "CROISSANTS": 4,
                "JAMS": 2
            }
        }
        
        # Position limits
        self.position_limits = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }
        
        # Thresholds for considering a position "high" (as percentage of limit)
        self.high_position_threshold = 0.7  # 70% of position limit
        
        # Hedging parameters
        self.max_hedge_trade_size = 10
        self.min_hedge_trade_size = 1
    
    def update_price_history(self, state: TradingState) -> None:
        """
        Update price history with current mid prices from the market.
        """
        for product in self.price_history.keys():
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                    
                    self.price_history[product].append(mid_price)
                    
                    # Keep history to defined length
                    if len(self.price_history[product]) > self.state_history_length:
                        self.price_history[product] = self.price_history[product][-self.state_history_length:]
    
    def update_hedge_coefficients(self) -> None:
        """
        Update hedge coefficients using linear regression on price history.
        For each basket, calculate regression coefficients for its components.
        """
        # Minimum required history length for regression
        min_history = 10
        
        for basket in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            # Skip if not enough history for basket
            if len(self.price_history[basket]) < min_history:
                continue
                
            basket_prices = np.array(self.price_history[basket])
            components = self.default_compositions[basket].keys()
            
            # Prepare component data for regression
            component_data = {}
            all_components_have_data = True
            
            for component in components:
                if len(self.price_history[component]) < min_history:
                    all_components_have_data = False
                    break
                    
                # Use same length as basket history
                history_length = min(len(self.price_history[basket]), len(self.price_history[component]))
                component_data[component] = np.array(self.price_history[component][-history_length:])
            
            if not all_components_have_data:
                continue
                
            # Adjust basket prices to same length
            basket_prices = basket_prices[-history_length:]
            
            # Perform regression for each component individually
            for component in components:
                component_prices = component_data[component]
                
                # Simple linear regression: price_basket = α + β * price_component
                X = component_prices.reshape(-1, 1)
                ones = np.ones(len(X))
                X_with_const = np.column_stack((ones, X))
                beta = np.linalg.lstsq(X_with_const, basket_prices, rcond=None)[0]
                
                # beta[1] is the regression coefficient (slope)
                self.hedge_coefficients[basket][component] = beta[1]
    
    def identify_high_positions(self, state: TradingState) -> List[str]:
        """
        Identify products with high positions that need hedging.
        """
        high_position_products = []
        
        for product, limit in self.position_limits.items():
            position = state.position.get(product, 0)
            position_pct = abs(position) / limit
            
            if position_pct >= self.high_position_threshold:
                high_position_products.append(product)
        
        return high_position_products
    
    def calculate_hedge_trades(
        self, state: TradingState, high_position_products: List[str]
    ) -> Dict[str, List[Order]]:
        """
        Calculate hedge trades to reduce exposure on high position products.
        """
        hedge_orders = {}
        
        # For each high position product, calculate hedge trades
        for product in high_position_products:
            position = state.position.get(product, 0)
            
            # Skip if position is zero
            if position == 0:
                continue
                
            # Determine hedge direction (opposite of current position)
            is_long = position > 0
            
            # Determine which products to use for hedging
            if product in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
                # If a basket has high position, hedge with components
                self._hedge_basket_with_components(state, product, is_long, hedge_orders)
            else:
                # If a component has high position, hedge with baskets
                self._hedge_component_with_baskets(state, product, is_long, hedge_orders)
        
        return hedge_orders

class VolcanicVolatilityStrategy(Strategy):
    """
    Unified strategy for trading VOLCANIC_ROCK and its vouchers (options)
    Uses pair trading based on implied vs. realized volatility differences
    But with a contrarian approach for the underlying asset
    """
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        
        # Store price history for realized volatility calculation
        self.price_history = []
        self.max_history_size = 15  # Balance between accuracy and performance
        
        # Risk-free rate (assumed 0 in this environment)
        self.risk_free_rate = 0
        
        # Voucher (option) symbols and their strike prices
        self.vouchers = {
            Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10500
        }
        
        # Position limits for each product
        self.position_limits = {
            Product.VOLCANIC_ROCK: min(limit, 200),  # Increased limit for underlying
            Product.VOLCANIC_ROCK_VOUCHER_9500: 40,  # Increased option limits
            Product.VOLCANIC_ROCK_VOUCHER_9750: 40,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 40,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 40,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 40
        }
        
        # Strategy parameters
        self.pair_trade_size = 10  # Size for each pair trade
        self.vol_diff_threshold = 0.03  # 3% difference to trigger trade
        
        # Tracking orders for all products
        self.all_orders = {}
    
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int]:
        """
        Override the run method to handle multiple products
        Returns orders for all products and conversions (always 0 for volcanic products)
        """
        # Initialize orders dictionary for all products
        self.all_orders = {product: [] for product in [self.symbol] + list(self.vouchers.keys())}
        self.orders = []  # Will be used for the main symbol
        
        # Execute trading strategy
        self.act(state)
        
        # Collect all orders for the main product
        if self.orders:
            self.all_orders[self.symbol] = self.orders
        
        # Flatten the orders dictionary into the expected format
        result_orders = {}
        for product, order_list in self.all_orders.items():
            if order_list:
                result_orders[product] = order_list
        
        return result_orders, 0  # No conversions for volcanic products
    
    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        """Calculate mid price from order book"""
        if symbol not in state.order_depths:
            return None
            
        order_depth = state.order_depths[symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        return (best_bid + best_ask) / 2
    
    def act(self, state: TradingState) -> None:
        """Execute a volatility-based pair trading strategy"""
        # Check if underlying is in the market
        if self.symbol not in state.order_depths:
            return
            
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return
        
        # Get current price of the underlying
        rock_price = self.get_mid_price(state, self.symbol)
        if rock_price is None:
            return
            
        # Add to price history for volatility calculation
        self.price_history.append(rock_price)
        if len(self.price_history) > self.max_history_size:
            self.price_history = self.price_history[-self.max_history_size:]
        
        # Fixed at 4 days to expiry as specified
        days_remaining = 4
        time_to_expiry = days_remaining / 252  # Convert to years (252 trading days)
        
        # Calculate realized volatility
        realized_vol = self._get_realized_volatility()
        
        # First, calculate delta exposure (but don't hedge yet)
        net_delta = self._calculate_net_delta(state, rock_price, time_to_expiry)
        
        # Then, look for volatility arbitrage opportunities
        self._trade_volatility_arbitrage(state, rock_price, time_to_expiry, realized_vol, net_delta)
    
    def _calculate_net_delta(self, state, rock_price, time_to_expiry):
        """Calculate net delta exposure from options without hedging"""
        # Calculate total delta exposure from all options
        net_delta = 0
        
        # Use a fixed volatility for delta calculation
        fixed_vol = 0.1
        
        for voucher_symbol, strike_price in self.vouchers.items():
            # Get current position
            position = state.position.get(voucher_symbol, 0)
            
            # Skip if no position
            if position == 0:
                continue
                
            # Calculate option delta
            option_delta = self._calculate_option_delta(
                rock_price, strike_price, time_to_expiry, self.risk_free_rate, fixed_vol
            )
            
            # Add to net delta exposure
            net_delta += position * option_delta
            
        return net_delta
    
    def _trade_volatility_arbitrage(self, state, rock_price, time_to_expiry, realized_vol, current_net_delta):
        """
        Execute volatility arbitrage with contrarian approach for underlying
        """
        # For each option, estimate implied vol and compare to realized vol
        options_traded = 0
        underlying_trades = []
        
        # Set a maximum of options to trade per round to avoid timeouts
        max_options_per_round = 2
        
        for voucher_symbol, strike_price in self.vouchers.items():
            # Skip if we've already traded enough options this round
            if options_traded >= max_options_per_round:
                break
                
            # Check if this option is in the order book
            if voucher_symbol not in state.order_depths:
                continue
                
            order_depth = state.order_depths[voucher_symbol]
            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue
                
            # Get current market price
            option_price = self.get_mid_price(state, voucher_symbol)
            if option_price is None:
                continue
                
            # Get current position
            current_position = state.position.get(voucher_symbol, 0)
            
            # Calculate position limit remaining
            position_limit = self.position_limits.get(voucher_symbol, 40)
            buy_capacity = position_limit - current_position
            sell_capacity = position_limit + current_position
            
            # Skip if we're at position limits
            if buy_capacity <= 0 and sell_capacity <= 0:
                continue
            
            # Calculate implied volatility using a simple approximation
            import math
            atm_factor = 0.4  # Approximation factor
            implied_vol_approx = option_price / (rock_price * atm_factor * math.sqrt(time_to_expiry))
            
            # Apply a factor based on moneyness
            moneyness = rock_price / strike_price
            if moneyness > 1.1:  # Deep ITM
                implied_vol_approx *= 1.3
            elif moneyness < 0.9:  # Deep OTM
                implied_vol_approx *= 0.7
            
            # Calculate volatility difference
            vol_diff = implied_vol_approx - realized_vol
            
            # Trade size based on capacity and pair trade size
            trade_size = min(self.pair_trade_size, max(buy_capacity, sell_capacity))
            if trade_size < 2:  # Skip if can't trade meaningful size
                continue
                
            # Execute trades based on volatility differences
            if abs(vol_diff) > self.vol_diff_threshold:
                if vol_diff > 0:  # Implied vol > realized vol (options overpriced)
                    # Only sell if we have capacity
                    if sell_capacity >= trade_size:
                        # Sell option
                        price = max(order_depth.buy_orders.keys())
                        self._add_order(voucher_symbol, price, -trade_size)
                        
                        # Calculate approximate delta
                        option_delta = self._calculate_simple_delta(rock_price, strike_price, time_to_expiry)
                        
                        # CONTRARIAN: Instead of buying underlying for hedge, we'll sell it
                        # This reverses the normal hedging relationship
                        hedge_size = round(trade_size * option_delta)
                        if hedge_size > 0:
                            underlying_trades.append((-hedge_size, "Contrarian hedge for " + voucher_symbol))
                                
                        options_traded += 1
                else:  # Implied vol < realized vol (options underpriced)
                    # Only buy if we have capacity
                    if buy_capacity >= trade_size:
                        # Buy option
                        price = min(order_depth.sell_orders.keys())
                        self._add_order(voucher_symbol, price, trade_size)
                        
                        # Calculate approximate delta
                        option_delta = self._calculate_simple_delta(rock_price, strike_price, time_to_expiry)
                        
                        # CONTRARIAN: Instead of selling underlying for hedge, we'll buy it
                        # This reverses the normal hedging relationship
                        hedge_size = round(trade_size * option_delta)
                        if hedge_size > 0:
                            underlying_trades.append((hedge_size, "Contrarian hedge for " + voucher_symbol))
                                
                        options_traded += 1
        
        # Now apply underlying trades, but with contrarian logic
        rock_position = state.position.get(self.symbol, 0)
        rock_limit = self.position_limits.get(self.symbol, 200)
        
        # First, calculate the "normal" delta hedge target
        normal_target = -round(current_net_delta)
        
        # Then, invert it for our contrarian approach
        # If normal target would be to buy, we'll sell instead and vice versa
        contrarian_target = -normal_target
        
        # Limit the target to our position limits
        if contrarian_target > rock_limit:
            contrarian_target = rock_limit
        elif contrarian_target < -rock_limit:
            contrarian_target = -rock_limit
            
        # Calculate the adjustment needed
        adjustment = contrarian_target - rock_position
        
        # Add the specific hedges from our option trades
        for hedge_size, reason in underlying_trades:
            adjustment += hedge_size
            
        # Ensure we don't exceed position limits after adding specific hedges
        if rock_position + adjustment > rock_limit:
            adjustment = rock_limit - rock_position
        elif rock_position + adjustment < -rock_limit:
            adjustment = -rock_limit - rock_position
            
        # Execute the adjustment if significant
        min_adjustment = 2
        if abs(adjustment) >= min_adjustment:
            if adjustment > 0:  # Need to buy
                order_depth = state.order_depths[self.symbol]
                if order_depth.sell_orders:
                    price = min(order_depth.sell_orders.keys())
                    self.buy(price, adjustment)
            else:  # Need to sell
                order_depth = state.order_depths[self.symbol]
                if order_depth.buy_orders:
                    price = max(order_depth.buy_orders.keys())
                    self.sell(price, -adjustment)
    
    def _calculate_simple_delta(self, S, K, T):
        """
        Calculate a simplified delta for a call option
        Using a quick approximation to avoid complex computations
        """
        # Very rough heuristic based approximation
        moneyness = S / K
        
        if moneyness > 1.05:  # ITM
            return 0.8
        elif moneyness < 0.95:  # OTM
            return 0.2
        else:  # ATM
            return 0.5
    
    def _add_order(self, symbol, price, quantity):
        """Add an order for a specific product"""
        if symbol not in self.all_orders:
            self.all_orders[symbol] = []
            
        self.all_orders[symbol].append(Order(symbol, price, quantity))
    
    def _calculate_option_delta(self, S, K, T, r, sigma):
        """Calculate Black-Scholes delta for a call option"""
        from statistics import NormalDist
        import math
        
        # If almost expired, use intrinsic value delta
        if T < 0.01:
            return 1.0 if S > K else 0.0
            
        # Calculate d1 from Black-Scholes formula
        d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
        
        # Delta of call option is N(d1)
        return NormalDist().cdf(d1)
    
    def _get_realized_volatility(self):
        """Calculate realized volatility from price history"""
        import math
        import numpy as np
        
        # Ensure we have enough data
        if len(self.price_history) < 2:
            return 0.2  # Default if not enough data
            
        # Calculate logarithmic returns
        returns = []
        for i in range(1, len(self.price_history)):
            returns.append(math.log(self.price_history[i] / self.price_history[i-1]))
            
        # Calculate standard deviation of returns
        if not returns:
            return 0.2
            
        # Annualize volatility (assuming 252 trading days per year)
        realized_vol = np.std(returns) * math.sqrt(252)
        
        # Ensure reasonable bounds
        return max(0.1, min(0.4, realized_vol))
    
    def save(self):
        """Save strategy state for next round"""
        return {'price_history': self.price_history}
    
    def load(self, data):
        """Load strategy state from previous round"""
        if data is None:
            return
            
        if isinstance(data, dict) and 'price_history' in data:
            self.price_history = data['price_history']

                                                    
class Trader:
    def __init__(self) -> None:
        # Position limits for each product
        self.limits = {
            Product.RAINFOREST_RESIN: 50, 
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANT: 250,
            Product.JAM: 350,
            Product.DJEMBE: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200
        }

        # Create strategies for each product
        self.strategies = {
            Product.RAINFOREST_RESIN: RainforestResinStrategy(Product.RAINFOREST_RESIN, self.limits[Product.RAINFOREST_RESIN]),
            Product.KELP: KelpStrategy(Product.KELP, self.limits[Product.KELP]),
            Product.SQUID_INK: SquidInkStrategy(Product.SQUID_INK, self.limits[Product.SQUID_INK]),
            Product.PICNIC_BASKET1: PicnicBasketStrategy(Product.PICNIC_BASKET1, self.limits[Product.PICNIC_BASKET1]),
            Product.PICNIC_BASKET2: PicnicBasketStrategy(Product.PICNIC_BASKET2, self.limits[Product.PICNIC_BASKET2])
        }
        
        # Create unified volcanic strategy to handle VOLCANIC_ROCK and all vouchers
        self.volcanic_strategy = VolcanicVolatilityStrategy(
            Product.VOLCANIC_ROCK,
            self.limits[Product.VOLCANIC_ROCK]
        )
        
        # Initialize logger
        self.logger = Logger()
        
        # Initialize regression hedging strategy
        self.hedging_strategy = RegressionHedgingStrategy()
    
    def _hedge_basket_with_components(
        self, state: TradingState, basket: str, is_long: bool, hedge_orders: Dict[str, List[Order]]
    ) -> None:
        """
        Hedge a high basket position using its components.
        If long basket, sell components; if short basket, buy components.
        """
        position = state.position.get(basket, 0)
        
        # Calculate trade size as a portion of the position to reduce (but not eliminate)
        hedge_size = min(abs(position) // 3, self.max_hedge_trade_size)
        hedge_size = max(hedge_size, self.min_hedge_trade_size)
        
        if hedge_size == 0:
            return
            
        # Get components for this basket
        components = self.hedge_coefficients[basket].keys()
        
        # Check if we can actually hedge with components
        for component in components:
            # Skip if component not in market
            if component not in state.order_depths:
                return
                
            # Check if order book has necessary liquidity
            order_depth = state.order_depths[component]
            if (is_long and not order_depth.sell_orders) or (not is_long and not order_depth.buy_orders):
                return
        
        # Create hedge orders
        for component in components:
            # Get coefficient (or use default composition if regression not ready)
            coefficient = self.hedge_coefficients[basket][component]
            if coefficient is None:
                coefficient = self.default_compositions[basket][component]
            
            # Calculate component quantity based on hedge size and coefficient
            component_qty = round(hedge_size * coefficient)
            
            # Skip if quantity is zero
            if component_qty == 0:
                continue
                
            # Determine price
            order_depth = state.order_depths[component]
            if is_long:  # If long basket, we sell components
                price = max(order_depth.buy_orders.keys())  # Sell at best bid
                # Make the order opposite to basket position
                qty = -component_qty
            else:  # If short basket, we buy components
                price = min(order_depth.sell_orders.keys())  # Buy at best ask
                # Make the order opposite to basket position
                qty = component_qty
            
            # Add order
            if component not in hedge_orders:
                hedge_orders[component] = []
            
            hedge_orders[component].append(Order(component, price, qty))
        
        # Also reduce the basket position
        if basket in state.order_depths:
            order_depth = state.order_depths[basket]
            
            if is_long and order_depth.buy_orders:  # If long basket, sell some
                price = max(order_depth.buy_orders.keys())  # Sell at best bid
                
                if basket not in hedge_orders:
                    hedge_orders[basket] = []
                    
                hedge_orders[basket].append(Order(basket, price, -hedge_size))
                
            elif not is_long and order_depth.sell_orders:  # If short basket, buy some
                price = min(order_depth.sell_orders.keys())  # Buy at best ask
                
                if basket not in hedge_orders:
                    hedge_orders[basket] = []
                    
                hedge_orders[basket].append(Order(basket, price, hedge_size))
    
    def _hedge_component_with_baskets(
        self, state: TradingState, component: str, is_long: bool, hedge_orders: Dict[str, List[Order]]
    ) -> None:
        """
        Hedge a high component position using baskets that contain it.
        If long component, buy baskets and sell other components; 
        if short component, sell baskets and buy other components.
        """
        position = state.position.get(component, 0)
        
        # Calculate which baskets contain this component
        containing_baskets = []
        for basket, components in self.default_compositions.items():
            if component in components:
                containing_baskets.append(basket)
        
        if not containing_baskets:
            return
            
        # Calculate hedge size
        hedge_size = min(abs(position) // 5, self.max_hedge_trade_size)
        hedge_size = max(hedge_size, self.min_hedge_trade_size)
        
        if hedge_size == 0:
            return
        
        # Try to hedge with each basket
        for basket in containing_baskets:
            # Skip if basket not in market
            if basket not in state.order_depths:
                continue
                
            # Check if order book has necessary liquidity
            order_depth = state.order_depths[basket]
            if (not is_long and not order_depth.sell_orders) or (is_long and not order_depth.buy_orders):
                continue
            
            # Get coefficient (or use default composition if regression not ready)
            coefficient = self.hedge_coefficients.get(basket, {}).get(component)
            if coefficient is None:
                coefficient = self.default_compositions[basket][component]
            
            # Calculate basket quantity based on hedge size and coefficient
            # We divide by coefficient because we're going in the opposite direction
            basket_qty = round(hedge_size / coefficient) if coefficient != 0 else 0
            
            # Skip if quantity is zero
            if basket_qty == 0:
                continue
                
            # Determine price for basket
            if is_long:  # If long component, we buy baskets
                price = min(order_depth.sell_orders.keys())  # Buy at best ask
                # Add order for basket
                if basket not in hedge_orders:
                    hedge_orders[basket] = []
                
                hedge_orders[basket].append(Order(basket, price, basket_qty))
                
            else:  # If short component, we sell baskets
                price = max(order_depth.buy_orders.keys())  # Sell at best bid
                # Add order for basket
                if basket not in hedge_orders:
                    hedge_orders[basket] = []
                
                hedge_orders[basket].append(Order(basket, price, -basket_qty))
            
            # Now we need to hedge the other components in the basket
            for other_component, other_coef in self.default_compositions[basket].items():
                if other_component == component:
                    continue  # Skip the component we're hedging
                
                if other_component not in state.order_depths:
                    continue
                
                order_depth = state.order_depths[other_component]
                
                # When buying baskets (long component), we sell other components
                # When selling baskets (short component), we buy other components
                other_qty = basket_qty * other_coef
                
                if other_qty == 0:
                    continue
                
                if is_long:  # If long our component, sell other components
                    if not order_depth.buy_orders:
                        continue
                    price = max(order_depth.buy_orders.keys())  # Sell at best bid
                    qty = -other_qty
                else:  # If short our component, buy other components
                    if not order_depth.sell_orders:
                        continue
                    price = min(order_depth.sell_orders.keys())  # Buy at best ask
                    qty = other_qty
                
                # Add order for other component
                if other_component not in hedge_orders:
                    hedge_orders[other_component] = []
                
                hedge_orders[other_component].append(Order(other_component, price, qty))
    
    def execute_hedge(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Main method to execute regression-based hedging strategy.
        """
        # Update price history with latest market data
        self.update_price_history(state)
        
        # Update regression coefficients
        self.update_hedge_coefficients()
        
        # Identify products with high positions
        high_position_products = self.identify_high_positions(state)
        
        # If no high positions, return empty orders
        if not high_position_products:
            return {}
        
        # Calculate hedge trades for high position products
        hedge_orders = self.calculate_hedge_trades(state, high_position_products)
        
        return hedge_orders
        
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        """Main trading logic"""
        orders = {}
        conversions = 0
        handled_products = set()

        # Load trader data from previous states
        old_trader_data = {}
        if state.traderData and state.traderData != "":
            try:
                old_trader_data = jsonpickle.decode(state.traderData)
            except:
                old_trader_data = {}
        
        new_trader_data = {}

        # First handle the basket arbitrage strategies
        for basket in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
            # Skip if basket isn't in order depths
            if basket not in state.order_depths:
                continue
                
            # Load previous state if available
            if basket in old_trader_data:
                self.strategies[basket].load(old_trader_data[basket])
            
            # Run the basket strategy
            strategy = self.strategies[basket]
            basket_orders, _ = strategy.run(state)
            component_orders = getattr(strategy, 'component_orders', {})
            
            # If we have basket orders, add them and mark products as handled
            if basket_orders:
                # For PICNIC_BASKET1, reverse the basket orders only (not component orders)
                if basket == Product.PICNIC_BASKET1:
                    reversed_basket_orders = []
                    for order in basket_orders:
                        # Reverse the order direction (buy to sell, sell to buy)
                        reversed_basket_orders.append(Order(order.symbol, order.price, -order.quantity))
                    orders[basket] = reversed_basket_orders
                else:
                    # For PICNIC_BASKET2, use orders as is
                    orders[basket] = basket_orders
                
                handled_products.add(basket)
                
                # Add component orders - these stay the same for both baskets
                for component, comp_orders in component_orders.items():
                    if comp_orders:
                        if component not in orders:
                            orders[component] = []
                        orders[component].extend(comp_orders)
                        handled_products.add(component)
            
            # Save strategy state
            new_trader_data[basket] = strategy.save()
        
        # Handle volcanic products with the unified strategy
        volcanic_products = [
            Product.VOLCANIC_ROCK,
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500
        ]
        
        # Check if any volcanic product is in order depths
        if any(product in state.order_depths for product in volcanic_products):
            # Load previous state if available
            if 'volcanic_strategy' in old_trader_data:
                self.volcanic_strategy.load(old_trader_data['volcanic_strategy'])
            
            # Run the volcanic strategy
            volcanic_orders, _ = self.volcanic_strategy.run(state)
            
            # Add volcanic orders to overall orders
            for product, product_orders in volcanic_orders.items():
                if product not in orders:
                    orders[product] = []
                orders[product].extend(product_orders)
                handled_products.add(product)
            
            # Save strategy state
            new_trader_data['volcanic_strategy'] = self.volcanic_strategy.save()
            
            # Log that we executed the volcanic strategy
            self.logger.print(f"Executed volcanic strategy with orders for {len(volcanic_orders)} products")
        
        # Process remaining products with standard strategies
        for product, strategy in self.strategies.items():
            # Skip products already handled
            if product in handled_products:
                self.logger.print(f"Skipping {product} - already handled")
                continue
                
            # Skip volcanic products (handled by unified strategy)
            if product in volcanic_products:
                continue
                
            # Skip products not in order depths
            if product not in state.order_depths:
                continue
                
            # Load previous state if available
            if product in old_trader_data:
                strategy.load(old_trader_data[product])
            
            # Run the strategy
            product_orders, product_conversions = strategy.run(state)
            
            # Add orders and conversions
            if product_orders:
                orders[product] = product_orders
            conversions += product_conversions
            
            # Save strategy state
            new_trader_data[product] = strategy.save()
        
        # Apply regression hedging after regular strategies
        hedge_orders = self.hedging_strategy.execute_hedge(state) if hasattr(self.hedging_strategy, 'execute_hedge') else {}
        
        # Merge hedge orders with regular orders
        if hedge_orders:
            self.logger.print("Executing regression-based position hedging:")
            for product, product_orders in hedge_orders.items():
                self.logger.print(f"  Hedging {product} with {len(product_orders)} orders")
                if product not in orders:
                    orders[product] = []
                orders[product].extend(product_orders)
        
        # Save trader data for next round
        trader_data = jsonpickle.encode(new_trader_data)
        
        # Log the trading activity
        self.logger.flush(state, orders, conversions, trader_data)
        
        return orders, conversions, trader_data