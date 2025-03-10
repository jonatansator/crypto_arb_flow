import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ccxt
from datetime import datetime
from scipy.optimize import minimize

# Constants
START_DATE = "2024-10-12T00:00:00Z"
END_DATE = "2024-10-12T00:16:40Z"  # ~1000 minutes
COLOR_BTC = '#FF9900'
COLOR_ETH = '#00B7EB'
COLOR_PROFIT = '#00FF00'  # Green for profit

def fetch_order_book(ticker, exchange, limit=100):
    """Fetch order book snapshot from Binance."""
    order_book = exchange.fetch_order_book(ticker, limit=limit)
    bids = pd.DataFrame(order_book['bids'], columns=['price', 'volume'])
    asks = pd.DataFrame(order_book['asks'], columns=['price', 'volume'])
    timestamp = order_book.get('timestamp', None)
    if timestamp is not None:
        timestamp = pd.to_datetime(timestamp, unit='ms')
    else:
        timestamp = pd.Timestamp.now()
    return bids, asks, timestamp

def simulate_order_flow(bids, asks, n_events=1000):
    """Simulate order arrivals with a Poisson process and stochastic intensity."""
    mid_price = (bids['price'].iloc[0] + asks['price'].iloc[0]) / 2
    spread = asks['price'].iloc[0] - bids['price'].iloc[0]
    
    base_rate = 0.1  # Orders per second
    intensity = base_rate + 0.05 * spread / mid_price
    
    inter_times = np.random.exponential(1 / intensity, n_events)
    times = np.cumsum(inter_times)
    
    events = pd.DataFrame({
        'time': times,
        'side': np.random.choice(['bid', 'ask'], size=n_events),
    })
    events['price'] = mid_price + (events['side'] == 'ask') * spread - (events['side'] == 'bid') * spread
    events['volume'] = np.random.exponential(0.1, n_events)
    return events

def detect_triangular_arbitrage(exchange):
    """Detect arbitrage opportunities across BTC/USDT, ETH/USDT, ETH/BTC."""
    tickers = ['BTC/USDT', 'ETH/USDT', 'ETH/BTC']
    prices = {}
    for ticker in tickers:
        order_book = exchange.fetch_order_book(ticker)
        prices[ticker] = {
            'bid': order_book['bids'][0][0],
            'ask': order_book['asks'][0][0]
        }
    
    # Path 1: USDT -> BTC -> ETH -> USDT
    usdt_to_btc = 1 / prices['BTC/USDT']['ask']  # Buy BTC with USDT
    btc_to_eth = 1 / prices['ETH/BTC']['ask']  # Sell BTC for ETH (inverse of ETH/BTC)
    eth_to_usdt = prices['ETH/USDT']['bid']  # Sell ETH for USDT
    path1 = usdt_to_btc * btc_to_eth * eth_to_usdt
    
    # Path 2: USDT -> ETH -> BTC -> USDT
    usdt_to_eth = 1 / prices['ETH/USDT']['ask']  # Buy ETH with USDT
    eth_to_btc = prices['ETH/BTC']['bid']  # Buy BTC with ETH
    btc_to_usdt = prices['BTC/USDT']['bid']  # Sell BTC for USDT
    path2 = usdt_to_eth * eth_to_btc * btc_to_usdt
    
    return path1, path2

def optimize_execution(events, path_profit, latency=0.01):
    """Optimize arbitrage execution minimizing slippage."""
    def objective(weights):
        slippage = np.sum(weights * events['volume'] * np.abs(events['price'] - events['price'].mean()))
        return -path_profit + slippage + latency * np.sum(weights)
    
    n_events = len(events)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_events)]
    initial_weights = np.ones(n_events) / n_events
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def plot_liquidity_surface(events, ticker):
    """Visualize 3D liquidity surface (price, volume, time)."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=events['time'],
        y=events['price'],
        z=events['volume'],
        mode='markers',
        marker=dict(size=3, color=events['price'], colorscale=[[0, COLOR_BTC], [1, COLOR_ETH]], opacity=0.8),
        name='Order Flow'
    ))
    
    fig.update_layout(
        title=dict(text=f'{ticker} Liquidity Surface (Poisson Process)', font_color='white', x=0.5),
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Price (USD)',
            zaxis_title='Volume',
            xaxis=dict(backgroundcolor='rgb(40,40,40)', gridcolor='rgba(255,255,255,0.2)', color='white'),
            yaxis=dict(backgroundcolor='rgb(40,40,40)', gridcolor='rgba(255,255,255,0.2)', color='white'),
            zaxis=dict(backgroundcolor='rgb(40,40,40)', gridcolor='rgba(255,255,255,0.2)', color='white'),
        ),
        plot_bgcolor='rgb(40,40,40)',
        paper_bgcolor='rgb(40,40,40)',
        font_color='white',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.show()

def plot_arbitrage_heatmap(profits, times, ticker):
    """Visualize arbitrage profit heatmap over time."""
    profit_bins = np.linspace(min(profits), max(profits), 50)
    heatmap_data, x_edges, y_edges = np.histogram2d(
        times,
        profits,
        bins=[len(times), len(profit_bins)],
        range=[[times.min(), times.max()], [profit_bins[0], profit_bins[-1]]]
    )
    heatmap_data = np.log1p(heatmap_data.T)  # Transpose to (y, x)
    
    x_bins = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_bins = 0.5 * (y_edges[:-1] + y_edges[1:])
    
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=x_bins,
        y=y_bins,
        colorscale=[[0, 'rgb(40,40,40)'], [0.5, COLOR_PROFIT], [1, 'white']],
        opacity=0.9,
        showscale=True,
        colorbar=dict(title=dict(text='Log Density', side='right'), tickfont=dict(color='white'))
    ))
    
    fig.update_layout(
        title=dict(text=f'{ticker} Arbitrage Profit Heatmap', font_color='white', x=0.5),
        xaxis_title=dict(text='Time (s)', font_color='white'),
        yaxis_title=dict(text='Profit (%)', font_color='white'),
        plot_bgcolor='rgb(40,40,40)',
        paper_bgcolor='rgb(40,40,40)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickangle=45, color='white'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.show()

if __name__ == "__main__":
    exchange = ccxt.binance()
    ticker = 'BTC/USDT'
    
    try:
        # Fetch order book and simulate events
        bids, asks, timestamp = fetch_order_book(ticker, exchange)
        print(f"Order book fetched at {timestamp}")
        
        events = simulate_order_flow(bids, asks)
        print(f"Simulated {len(events)} order events")
        
        # Detect arbitrage multiple times for heatmap
        n_samples = 10  # Simulate multiple checks
        profits = []
        times = events['time'].iloc[::len(events)//n_samples][:n_samples]  # Sample times
        for _ in range(n_samples):
            path1, path2 = detect_triangular_arbitrage(exchange)
            profit = max(path1, path2) - 1 if max(path1, path2) > 1 else min(path1, path2) - 1
            profits.append(profit * 100)  # Convert to percentage
            print(f"Path 1: {path1:.6f}, Path 2: {path2:.6f}, Profit: {profit*100:.4f}%")
        
        # Optimize execution if profitable
        max_profit = max(profits) / 100  # Back to decimal
        if max_profit > 0:
            weights = optimize_execution(events, max_profit)
            print(f"Optimal execution weights: {weights[:5]}... (first 5)")
        
        # Visualize
        plot_liquidity_surface(events, ticker)
        plot_arbitrage_heatmap(profits, times, ticker)
        
    except Exception as e:
        print(f"Error: {str(e)}")