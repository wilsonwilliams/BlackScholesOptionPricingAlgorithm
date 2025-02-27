from config import API_KEY, API_SECRET, BASE_URL
from BlackScholesSolver import BlackScholesSolver

from datetime import date, datetime, timedelta
import numpy as np

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import OrderRequest, GetOptionContractsRequest, MarketOrderRequest, GetOrdersRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, AssetStatus, QueryOrderStatus
from alpaca.data import StockHistoricalDataClient, StockTradesRequest, StockBarsRequest, TimeFrame, TimeFrameUnit, ContractType, OptionHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, OptionLatestTradeRequest, StockLatestTradeRequest



class TradingAlgorithm:
    """
    *
    """
    def __init__(self, _symbol: str, _price_diff_tolerance: float=0.2) -> None:
        self.tradingClient = TradingClient(API_KEY, API_SECRET, paper=True)
        self.dataClient = StockHistoricalDataClient(API_KEY, API_SECRET)
        self.optionClient = OptionHistoricalDataClient(API_KEY, API_SECRET)

        self.symbol = _symbol
        self.price_diff_tolerance = _price_diff_tolerance

        self.black_scholes_solver = BlackScholesSolver()

    """
    *
    """
    def get_open_options(self) -> None:
        open_positions = self.tradingClient.get_all_positions()
        self.open_options = set()
        for position in open_positions:
            self.open_options.add(position.symbol)
    
    """
    *
    """
    def get_pending_orders(self) -> None:
        reqParams = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
        )
        open_orders = self.tradingClient.get_orders(reqParams)
        pending_limit_orders = [order for order in open_orders if order.order_type == 'limit']
        for order in pending_limit_orders:
            self.open_options.add(order.symbol)
    
    """
    *
    """
    def place_market_order(self, _symbol, _qty, _side, _time_in_force=TimeInForce.DAY) -> None:
        market_order_request = MarketOrderRequest(
            symbol=_symbol,
            qty=_qty,
            side=_side,
            time_in_force=_time_in_force,
        )

        self.tradingClient.submit_order(market_order_request)

    """
    *
    """
    def place_limit_order(self, _symbol, _qty, _side, _limit_price, _time_in_force=TimeInForce.DAY) -> None:
        limit_order_request = LimitOrderRequest(
            symbol=_symbol,
            qty=_qty,
            side=_side,
            time_in_force=_time_in_force,
            limit_price=_limit_price,
        )

        self.tradingClient.submit_order(limit_order_request)

    """
    *
    """
    def compute_T(self, future_date: date):
        today = date.today()
        T = (future_date - today).days
        return T / 252

    """
    *
    """
    def compute_sigma(self, stock_bars):
        close_prices = []
        for data in stock_bars:
            close_prices.append(data.close)

        returns = []
        for i in range(len(close_prices[1:])):
            returns.append((close_prices[i] - close_prices[i - 1]) / close_prices[i - 1])
        
        return np.std(returns) * np.sqrt(252)
    
    """
    *
    """
    # def compute_r(self, S, K, sigma, T, C):
    #     return self.black_scholes_solver.solve_for_r(S, K, sigma, T, C)

    """
    *
    """
    def compute_call_price(self, S, K, sigma, T, r):
        return self.black_scholes_solver.solve_for_C(S, K, sigma, T, r)

    """
    *
    """
    def get_stock_returns(self):
        today = datetime.now()
        one_year_ago = today - timedelta(days=365)

        reqParams = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=one_year_ago,
        )

        stock_bars = self.dataClient.get_stock_bars(reqParams)
        return stock_bars.data[self.symbol]
    
    """
    *
    """
    def get_latest_stock_price(self):
        reqParams = StockLatestTradeRequest(
            symbol_or_symbols=self.symbol,
        )

        latest_trade = self.dataClient.get_stock_latest_trade(reqParams)
        return latest_trade[self.symbol].price

    """
    *
    """
    def get_available_options(self):
        reqParams = GetOptionContractsRequest(
            underlying_symbols=[self.symbol],
        )

        available_options = self.tradingClient.get_option_contracts(reqParams)
        return available_options.option_contracts
    
    """
    *
    """
    def filter_options(self, available_options, stock_price, lower_boundary=0.95, upper_boundary=1.1):
        options = []

        for option in available_options:
            if option.symbol in self.open_options:
                continue

            if not option.type == ContractType.CALL or not option.status == AssetStatus.ACTIVE:
                continue

            if option.strike_price < lower_boundary * stock_price or option.strike_price > upper_boundary * stock_price:
                continue

            options.append(option)
        
        return options

    """
    *
    """
    def get_option_market_prices(self, options):
        option_to_market_price = {}

        for option in options:
            reqParams = OptionLatestTradeRequest(
                symbol_or_symbols=option.symbol,
            )

            option_data = self.optionClient.get_option_latest_trade(reqParams)
            option_to_market_price[option.symbol] = option_data[option.symbol].price
        
        return option_to_market_price

    """
    *
    """
    def get_fair_value_price(self, option, stock_price, volatility, r) -> float:
        black_scholes_price = self.compute_call_price(stock_price, option.strike_price, volatility, self.compute_T(option.expiration_date), r)
        fair_value_price = float(black_scholes_price)
        return fair_value_price

    """
    *
    """
    def run(self):
        self.get_open_options()
        self.get_pending_orders()

        volatility = self.compute_sigma(self.get_stock_returns())
        print(f"Volatility: {volatility}")

        latest_stock_price = self.get_latest_stock_price()
        print(f"Latest Stock Price {self.symbol}: {latest_stock_price}")

        available_options = self.get_available_options()
        filtered_options = self.filter_options(available_options, latest_stock_price)
        option_to_market_price = self.get_option_market_prices(filtered_options)

        r = 0.0432  # 10 Year Treasury Bond Rate

        for option in filtered_options:
            fair_value_price = self.get_fair_value_price(option, latest_stock_price, volatility, r)

            if fair_value_price > option_to_market_price[option.symbol] + self.price_diff_tolerance:
                print(f"Buying Call Option (K = {option.strike_price}, Exp = {option.expiration_date}): Fair Value Above Market Value ({fair_value_price} > {option_to_market_price[option.symbol]})")
                # self.place_market_order(option.symbol, 1, OrderSide.BUY, TimeInForce.DAY)
                self.place_limit_order(option.symbol, 1, OrderSide.BUY, round(fair_value_price, 2))

            elif fair_value_price < option_to_market_price[option.symbol] - self.price_diff_tolerance:
                print(f"Selling Call Option (K = {option.strike_price}, Exp = {option.expiration_date}): Fair Value Below Market Value ({fair_value_price} < {option_to_market_price[option.symbol]})")
                for i in range(100):
                    try:
                        self.place_limit_order(option.symbol, 1, OrderSide.SELL, round(fair_value_price, 2))
                        break
                    except:
                        if i == 0:
                            self.place_market_order(self.symbol, 100, OrderSide.BUY)

            else:
                print(f"Call Option (K = {option.strike_price}, Exp = {option.expiration_date}): Fair Value Close To Market Value ({fair_value_price} ~ {option_to_market_price[option.symbol]})")
