from statistics import NormalDist
import json
import numpy as np
from datamodel import Order, OrderDepth, TradingState, Symbol, Listing, Trade, Observation
from typing import Any, List, Dict, Tuple
import jsonpickle
import math
from collections import deque


class Product:
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


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
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


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


class MacaronsStrategy(Strategy):
    """
    Simplified strategy for MAGNIFICENT_MACARONS:
    1. Buys everything when sunlight index < 45
    2. Sells everything when sunlight index starts increasing
    3. Executes arbitrage in other situations
    """
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        # Strategy parameters
        self.conversion_limit = 10
        self.conversions = 0
        self.critical_sunlight_index = 45
        
        # Track sunlight history for trend detection
        self.sunlight_history = []
    
    def act(self, state: TradingState) -> None:
        """Simplified strategy execution based solely on sunlight index"""
        # Reset conversions
        self.conversions = 0
        
        # Quick validity checks
        if (self.symbol not in state.order_depths or 
            self.symbol not in state.observations.conversionObservations):
            return
            
        # Get order book data
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return
            
        # Get conversion data
        conv_obs = state.observations.conversionObservations[self.symbol]
        
        # Get current sunlight index and update history
        current_sunlight = conv_obs.sunlightIndex
        self.sunlight_history.append(current_sunlight)
        
        # Keep only the most recent sunlight readings
        if len(self.sunlight_history) > 10:
            self.sunlight_history = self.sunlight_history[-10:]
        
        # Get current position
        position = state.position.get(self.symbol, 0)
        
        # ALWAYS try arbitrage first
        if self._try_arbitrage(state, order_depth, position, conv_obs):
            return
        
        # Check if sunlight index is below critical threshold
        if current_sunlight < self.critical_sunlight_index:
            # Only buy if we're not at position limit
            if position < self.limit:
                self._buy_all(state, order_depth, position, conv_obs)
                return
        
        # Check if sunlight has started increasing (positive slope)
        if len(self.sunlight_history) >= 2:
            # Calculate slope from the last two readings
            slope = self.sunlight_history[-1] - self.sunlight_history[-2]
            
            # If slope is positive and we have a position, SELL everything
            if slope > 0 and position > 0:
                self._sell_all(state, order_depth, position, conv_obs)
                return

    def _try_arbitrage(self, state, order_depth, position, conv_obs):
        """Execute pure arbitrage and return True if executed"""
        # Get market prices
        market_best_bid = max(order_depth.buy_orders.keys())
        market_best_ask = min(order_depth.sell_orders.keys())
        
        # Get conversion prices
        conversion_buy_price = conv_obs.askPrice + conv_obs.transportFees + conv_obs.importTariff
        conversion_sell_price = conv_obs.bidPrice - conv_obs.transportFees - conv_obs.exportTariff
        
        # Arbitrage 1: Buy from conversion, sell to market
        if conversion_buy_price < market_best_bid:
            # Calculate potential profit
            profit_per_unit = market_best_bid - conversion_buy_price
            # Account for storage costs
            storage_cost = 0.1  # Storage cost per unit per timestamp
            adjusted_profit = profit_per_unit - storage_cost
            
            if adjusted_profit > 0:
                # Calculate trade size (default to conversion limit for pure arbitrage)
                trade_size = min(self.conversion_limit, 
                               sum(volume for price, volume in order_depth.buy_orders.items() 
                                  if price >= market_best_bid))
                
                if trade_size > 0:
                    self.conversions = trade_size  # Buy from conversion
                    self.sell(market_best_bid, trade_size)  # Sell to market
                    return True
        
        # Arbitrage 2: Buy from market, sell to conversion
        elif market_best_ask < conversion_sell_price:
            # Calculate potential profit
            profit_per_unit = conversion_sell_price - market_best_ask
            
            if profit_per_unit > 0:
                # Calculate trade size (default to conversion limit for pure arbitrage)
                trade_size = min(self.conversion_limit, 
                               sum(-volume for price, volume in order_depth.sell_orders.items() 
                                  if price <= market_best_ask))
                
                if trade_size > 0:
                    self.buy(market_best_ask, trade_size)  # Buy from market
                    self.conversions = -trade_size  # Sell to conversion
                    return True
        
        return False

    def _buy_all(self, state, order_depth, position, conv_obs):
        """Buy as much as possible using market and conversion"""
        # Calculate how much we can buy
        to_buy = self.limit - position
        if to_buy <= 0:
            return
        
        # Get market prices
        market_best_ask = min(order_depth.sell_orders.keys())
        
        # Get conversion price
        conversion_buy_price = conv_obs.askPrice + conv_obs.transportFees + conv_obs.importTariff
        
        # Track if any trades executed
        trades_executed = False
        
        # First, buy from cheaper source
        if market_best_ask <= conversion_buy_price:
            # Buy from market first
            market_liquidity = sum(-volume for price, volume in order_depth.sell_orders.items())
            market_buy_qty = min(to_buy, market_liquidity)
            
            if market_buy_qty > 0:
                self.buy(market_best_ask, market_buy_qty)
                to_buy -= market_buy_qty
                trades_executed = True
        else:
            # Market is more expensive, try conversion
            conversion_buy_qty = min(to_buy, self.conversion_limit)
            if conversion_buy_qty > 0:
                self.conversions = conversion_buy_qty
                to_buy -= conversion_buy_qty
                trades_executed = True
        
        # If no trades executed, try buying from more expensive source
        if not trades_executed:
            if market_best_ask < float('inf'):  # If market has liquidity
                market_buy_qty = min(to_buy, sum(-volume for price, volume in order_depth.sell_orders.items()))
                if market_buy_qty > 0:
                    self.buy(market_best_ask, market_buy_qty)
            else:
                # Try conversion as last resort
                conversion_buy_qty = min(to_buy, self.conversion_limit)
                if conversion_buy_qty > 0:
                    self.conversions = conversion_buy_qty

    def _sell_all(self, state, order_depth, position, conv_obs):
        """Sell entire position using market and conversion"""
        if position <= 0:
            return
        
        # Get market prices
        market_best_bid = max(order_depth.buy_orders.keys())
        
        # Get conversion price
        conversion_sell_price = conv_obs.bidPrice - conv_obs.transportFees - conv_obs.exportTariff
        
        # Track if any trades executed
        trades_executed = False
        
        # First, sell to the higher priced source
        if market_best_bid >= conversion_sell_price:
            # Sell to market first
            market_liquidity = sum(volume for price, volume in order_depth.buy_orders.items())
            market_sell_qty = min(position, market_liquidity)
            
            if market_sell_qty > 0:
                self.sell(market_best_bid, market_sell_qty)
                position -= market_sell_qty
                trades_executed = True
        else:
            # Conversion offers better price
            conversion_sell_qty = min(position, self.conversion_limit)
            if conversion_sell_qty > 0:
                self.conversions = -conversion_sell_qty
                position -= conversion_sell_qty
                trades_executed = True
        
        # If no trades executed, try selling to less favorable source
        if not trades_executed:
            if market_best_bid > 0:  # If market has liquidity
                market_sell_qty = min(position, sum(volume for price, volume in order_depth.buy_orders.items()))
                if market_sell_qty > 0:
                    self.sell(market_best_bid, market_sell_qty)
            else:
                # Try conversion as last resort
                conversion_sell_qty = min(position, self.conversion_limit)
                if conversion_sell_qty > 0:
                    self.conversions = -conversion_sell_qty
    
    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0
        
        self.act(state)
        
        return self.orders, self.conversions
        
    def save(self):
        """Save strategy state for next iteration"""
        return {
            'sunlight_history': self.sunlight_history
        }
        
    def load(self, data):
        """Load strategy state from previous iteration"""
        if not data:
            return
            
        if 'sunlight_history' in data:
            self.sunlight_history = data['sunlight_history']
                        
class Trader:
    def __init__(self) -> None:
        # Position limits - only for macarons
        self.limits = {
            Product.MAGNIFICENT_MACARONS: 75
        }

        # Create strategies - only for macarons
        self.strategies = {
            Product.MAGNIFICENT_MACARONS: MacaronsStrategy(Product.MAGNIFICENT_MACARONS, self.limits[Product.MAGNIFICENT_MACARONS])
        }
        
        # Initialize logger
        self.logger = Logger()
    
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        """Main trading logic - optimized for macarons only"""
        orders = {}
        conversions = 0

        # Load trader data from previous states
        old_trader_data = {}
        if state.traderData and state.traderData != "":
            try:
                old_trader_data = jsonpickle.decode(state.traderData)
            except:
                old_trader_data = {}
        
        new_trader_data = {}

        # Process macarons strategy
        product = Product.MAGNIFICENT_MACARONS
        if product in state.order_depths:
            strategy = self.strategies[product]
            
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
        
        # Save trader data for next round
        trader_data = jsonpickle.encode(new_trader_data)
        
        # Log the trading activity
        self.logger.flush(state, orders, conversions, trader_data)
        
        return orders, conversions, trader_data