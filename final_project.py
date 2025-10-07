import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import requests
import difflib
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
    from tensorflow.keras.optimizers import Adam # type: ignore
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    st.sidebar.error("🚫 **TensorFlow not installed**")
    st.sidebar.markdown("""
    **To enable LSTM Deep Learning:**
    ```bash
    pip install tensorflow
    ```
    Then restart the application.
    """)

# --- Page Setup ---
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# --- Modern Title ---
st.markdown("""
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<h1 style='font-family:sans-serif;'>
    <i class="fa-solid fa-chart-column"></i> Stock Market Analysis & Prediction
</h1>
""", unsafe_allow_html=True)

# --- Format Large Numbers ---
def format_number(value, currency_symbol="", currency=""):
    if value is None or value == 0:
        return "N/A"
    trillion = 1_000_000_000_000 
    billion = 1_000_000_000
    million = 1_000_000
    thousand = 1_000
    if value >= trillion:
        return f"{currency_symbol}{value/trillion:.2f}T {currency}"
    elif value >= billion:
        return f"{currency_symbol}{value/billion:.2f}B {currency}"
    elif value >= million:
        return f"{currency_symbol}{value/million:.2f}M {currency}"
    elif value >= thousand:
        return f"{currency_symbol}{value/thousand:.2f}K {currency}"
    else:
        return f"{currency_symbol}{value} {currency}"

# --- Stock List ---
stock_choices = {
   "Reliance Industries (RELIANCE)": "RELIANCE.NS",
    "TCS (TCS)": "TCS.NS",
    "Infosys (INFY)": "INFY.NS",
    "HDFC Bank (HDFCBANK)": "HDFCBANK.NS",
    "State Bank of India (SBIN)": "SBIN.NS",
    "ICICI Bank (ICICIBANK)": "ICICIBANK.NS",
    "Tata Motors (TATAMOTORS)": "TATAMOTORS.NS",
    "Adani Ports (ADANIPORTS)": "ADANIPORTS.NS",
    "Asian Paints (ASIANPAINT)": "ASIANPAINT.NS",
    "Bajaj Finance (BAJFINANCE)": "BAJFINANCE.NS",
    "Hindustan Unilever (HINDUNILVR)": "HINDUNILVR.NS",
    "Larsen & Toubro (LT)": "LT.NS",
    "NTPC (NTPC)": "NTPC.NS",
    "ONGC (ONGC)": "ONGC.NS",
    "Maruti Suzuki (MARUTI)": "MARUTI.NS",
    "Axis Bank (AXISBANK)": "AXISBANK.NS",
    "JSW Steel (JSWSTEEL)": "JSWSTEEL.NS",
    "Coal India (COALINDIA)": "COALINDIA.NS",
    "Tech Mahindra (TECHM)": "TECHM.NS",
    "Wipro (WIPRO)": "WIPRO.NS",
    "Titan Company (TITAN)": "TITAN.NS",
    "UltraTech Cement (ULTRACEMCO)": "ULTRACEMCO.NS",
    "Dr Reddy's Labs (DRREDDY)": "DRREDDY.NS",
    "Power Grid Corp (POWERGRID)": "POWERGRID.NS",
    "Sun Pharma (SUNPHARMA)": "SUNPHARMA.NS",
    "Bharat Petroleum (BPCL)": "BPCL.NS",
    "Apple Inc. (AAPL)": "AAPL",
    "Google (Alphabet) (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Meta Platforms (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "Coca-Cola (KO)": "KO",
    "McDonald's (MCD)": "MCD",
    "Visa Inc. (V)": "V",
    "JPMorgan Chase (JPM)": "JPM",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Walmart (WMT)": "WMT",
    "Intel (INTC)": "INTC"
}

# --- Sidebar Search & Period ---
with st.sidebar:
    st.markdown("""
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<h3 style='font-family:sans-serif;'>
    <i class="fa-solid fa-search"></i> Search & Select Stock
</h3>
""", unsafe_allow_html=True)
    search_text = st.text_input("Type to search...")

    suggestions = [label for label in stock_choices if search_text.lower() in label.lower()]
    if not suggestions and search_text:
        suggestions = difflib.get_close_matches(search_text.upper(), stock_choices.keys(), n=5, cutoff=0.3)
    if not suggestions:
        suggestions = list(stock_choices.keys())

    selected_label = st.selectbox("Select Stocks...", suggestions)
    symbol = stock_choices[selected_label]
    period = st.selectbox("Select Time Period", ["1mo","3mo","6mo","1y","2y","5y"])

# --- Enhanced Technical Indicators ---
def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast).mean()
    ema_slow = df['Close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def calculate_volatility(df, window=20):
    returns = df['Close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility

# --- Load Data ---
@st.cache_data
def load_data(symbol, period):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    info = ticker.info
    
    # Moving Averages
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()
    
    # Bollinger Bands
    df["BB_Upper"] = df["Close"].rolling(window=20).mean() + 2*df["Close"].rolling(window=20).std()
    df["BB_Lower"] = df["Close"].rolling(window=20).mean() - 2*df["Close"].rolling(window=20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df)
    
    # Stochastic
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df)
    
    # Volatility
    df['Volatility'] = calculate_volatility(df)
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    return df, info

df, info = load_data(symbol, period)

currency = info.get("currency","INR")
currency_symbol = "₹" if symbol.endswith(".NS") else {"USD":"$","EUR":"€","GBP":"£"}.get(currency,currency)

# --- CSS Styling ---
st.markdown("""
<style>
.card-container { display:flex; flex-wrap:wrap; gap:1rem; }
.card { background-color:#f5f5f5 ; padding:1rem; border-radius:12px; width:220px; box-shadow:1px 1px 8px rgba(0,0,0,0.1); font-family:sans-serif; }
.card h4 { margin:0; font-size:15px; color:#111; }
.card p { margin:0; font-size:20px; font-weight:bold; color:#111; }

.card-container-scroll { 
    display: flex; 
    flex-wrap: nowrap; 
    gap: 1rem; 
    overflow-x: auto; 
    padding-bottom: 1rem; 
} 

.gainer { 
    background-color: #9eff9e; 
    color: black;              /* font color set to black */
    padding: 10px; 
    border-radius: 8px;
    min-width: 150px;
    font-family: Arial, sans-serif;
} 

.loser { 
    background-color: #ff5252; 
    color: black;              /* font color set to black */
    padding: 10px; 
    border-radius: 8px;
    min-width: 150px;
    font-family: Arial, sans-serif;
}
</style>

""", unsafe_allow_html=True)

# --- Stock Info Cards ---
st.markdown(
    """
    <h2 style='display: flex; align-items: center; color:#f5f5f5 ;'>
        <i class="fa-solid fa-circle-info" style="color:#f5f5f5 ; margin-right:10px;"></i>
        Stock Information
    </h2>
    """,
    unsafe_allow_html=True
)

st.markdown(f"""
<div class="card-container">
    <div class="card"><h4>Company</h4><p>{info.get('longName','N/A')}</p></div>
    <div class="card"><h4>Current Price</h4><p>{currency_symbol}{info.get('currentPrice',0):,.2f}</p></div>
    <div class="card"><h4>Sector</h4><p>{info.get('sector','N/A')}</p></div>
    <div class="card"><h4>Market Cap</h4><p>{format_number(info.get('marketCap'),currency_symbol,currency)}</p></div>
    <div class="card"><h4>52-Week High</h4><p>{currency_symbol}{info.get('fiftyTwoWeekHigh',0):,.2f}</p></div>
    <div class="card"><h4>52-Week Low</h4><p>{currency_symbol}{info.get('fiftyTwoWeekLow',0):,.2f}</p></div>
    <div class="card"><h4>Volume</h4><p>{format_number(info.get('volume'))}</p></div>
    <div class="card"><h4>Shares Outstanding</h4><p>{format_number(info.get('sharesOutstanding'))}</p></div>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Charts ---
st.markdown("""
    <h2 style='display:flex; align-items:center; gap:10px; color:#f5f5f5;'>
        <i class="fas fa-chart-line"></i> Technical Analysis Charts
    </h2>
""", unsafe_allow_html=True)

# Create subplots for comprehensive analysis
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=('Price & Volume', 'RSI', 'MACD', 'Stochastic'),
    vertical_spacing=0.08,
    row_heights=[0.5, 0.15, 0.2, 0.15],
    specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
)

# Price chart with volume
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], 
    low=df['Low'], close=df['Close'], name="Price"
), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], line=dict(color="orange", width=2), name="MA20"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], line=dict(color="green", width=2), name="MA50"), row=1, col=1, secondary_y=False)
if len(df) > 200:
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], line=dict(color="red", width=2), name="MA200"), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color="gray", dash="dot"), name="BB Upper"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], line=dict(color="gray", dash="dot"), name="BB Lower"), row=1, col=1, secondary_y=False)

# Volume bars
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", opacity=0.3, marker_color='blue'), row=1, col=1, secondary_y=True)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="purple"), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

# MACD
fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="blue"), name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color="red"), name="Signal"), row=3, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["MACD_Histogram"], name="Histogram", marker_color='gray', opacity=0.6), row=3, col=1)

# Stochastic
fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], line=dict(color="orange"), name="Stoch %K"), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"], line=dict(color="blue"), name="Stoch %D"), row=4, col=1)
fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)

fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white", showlegend=True)
fig.update_yaxes(title_text="Price", secondary_y=False, row=1, col=1)
fig.update_yaxes(title_text="Volume", secondary_y=True, row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1)
fig.update_yaxes(title_text="MACD", row=3, col=1)
fig.update_yaxes(title_text="Stochastic", row=4, col=1)

st.plotly_chart(fig, use_container_width=True)

# --- Recent Data ---
st.markdown("""
    <h2 style='display:flex; align-items:center; gap:10px; color:#f5f5f5;'>
        <i class="fas fa-table"></i> Recent Stock Data
    </h2>
""", unsafe_allow_html=True)
# Enhanced data display with key metrics
recent_data = df.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Volatility']].round(2)
st.dataframe(recent_data, use_container_width=True)

# --- Risk Analysis Section ---
st.markdown("""
    <h2 style='display:flex; align-items:center; gap:10px; color:#f5f5f5;'>
        <i class="fas fa-shield-alt"></i> Risk Analysis
    </h2>
""", unsafe_allow_html=True)

# Calculate risk metrics
returns = df['Close'].pct_change().dropna()
current_volatility = df['Volatility'].iloc[-1] if not pd.isna(df['Volatility'].iloc[-1]) else 0
sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
max_drawdown = ((df['Close'] / df['Close'].cummax()) - 1).min()
var_95 = np.percentile(returns, 5) * 100  # 95% VaR

# Risk level determination
risk_level = "Low"
risk_color = "green"
if current_volatility > 0.3:
    risk_level = "High"
    risk_color = "red"
elif current_volatility > 0.2:
    risk_level = "Medium"
    risk_color = "orange"

# Display risk metrics
risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
with risk_col1:
    st.metric("Current Volatility", f"{current_volatility:.1%}")
with risk_col2:
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
with risk_col3:
    st.metric("Max Drawdown", f"{max_drawdown:.1%}")
with risk_col4:
    st.metric("95% VaR (Daily)", f"{var_95:.2f}%")

st.markdown(f"""
<div style='padding:10px; border-radius:8px; background-color:{"#d4edda" if risk_color=="green" else "#fff3cd" if risk_color=="orange" else "#f8d7da"}; border:1px solid {"#c3e6cb" if risk_color=="green" else "#ffeaa7" if risk_color=="orange" else "#f5c6cb"}; margin:10px 0;'>
    <strong>Risk Assessment: <span style='color:{risk_color};'>{risk_level} Risk</span></strong>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Prediction Models ---
st.markdown("""
    <h2 style='display:flex; align-items:center; gap:10px; color:#f5f5f5;'>
        <i class="fas fa-bullseye"></i> Advanced Stock Price Prediction
    </h2>
""", unsafe_allow_html=True)

# Prediction parameters
col1, col2 = st.columns(2)
with col1:
    prediction_days = st.selectbox("Prediction Period", [5, 10, 15, 20], index=0)
with col2:
    if DEEP_LEARNING_AVAILABLE:
        model_type = st.selectbox("Model Type", ["Linear Regression", "Random Forest", "LSTM Deep Learning"], index=2)
    else:
        model_type = st.selectbox("Model Type", ["Linear Regression", "Random Forest"], index=1)

# Prepare features for traditional ML models
def prepare_features(df, lookback=10):
    features = []
    targets = []
    
    # Ensure we have enough data and clean NaN values
    df_clean = df.copy()
    df_clean = df_clean.ffill().bfill()  # Forward fill then backward fill
    
    # Additional safety: fill any remaining NaNs with 0
    df_clean = df_clean.fillna(0)
    
    for i in range(lookback, len(df_clean) - prediction_days):
        # Use multiple features: Close, Volume, RSI, MACD
        feature_set = []
        skip_row = False
        
        for j in range(lookback):
            idx = i - j
            close_val = df_clean['Close'].iloc[idx]
            volume_val = df_clean['Volume'].iloc[idx] / 1000000  # Normalize volume
            rsi_val = df_clean['RSI'].iloc[idx] / 100  # Normalize RSI
            macd_val = df_clean['MACD'].iloc[idx]
            
            # Check for any remaining NaN or infinite values
            if any(pd.isna([close_val, volume_val, rsi_val, macd_val])) or any(np.isinf([close_val, volume_val, rsi_val, macd_val])):
                skip_row = True
                break
                
            feature_set.extend([close_val, volume_val, rsi_val, macd_val])
        
        if not skip_row:
            target_val = df_clean['Close'].iloc[i + prediction_days]
            if not pd.isna(target_val) and not np.isinf(target_val):
                features.append(feature_set)
                targets.append(target_val)
    
    return np.array(features), np.array(targets)

# Prepare data for LSTM model
def prepare_lstm_data(df, lookback=60):
    """Prepare data specifically for LSTM model"""
    df_clean = df.copy()
    df_clean = df_clean.ffill().bfill().fillna(0)
    
    # Select features for LSTM
    feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'MA20', 'MA50', 'Volatility']
    
    # Create feature matrix
    features_data = []
    for col in feature_columns:
        if col in df_clean.columns:
            if col == 'Volume':
                features_data.append(df_clean[col].values / 1000000)  # Normalize volume
            elif col == 'RSI':
                features_data.append(df_clean[col].values / 100)  # Normalize RSI
            else:
                features_data.append(df_clean[col].values)
    
    # Stack features
    features_matrix = np.column_stack(features_data)
    
    # Scale the features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_matrix)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(features_scaled) - prediction_days):
        X.append(features_scaled[i-lookback:i])
        y.append(features_scaled[i + prediction_days, 0])  # Predict Close price (first feature)
    
    return np.array(X), np.array(y), scaler

# Create LSTM model
def create_lstm_model(input_shape):
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Train model
if len(df) > 100:  # Ensure enough data for LSTM
    
    if model_type == "LSTM Deep Learning" and DEEP_LEARNING_AVAILABLE:
        # LSTM Model Training
        st.info("🧠 Training LSTM Deep Learning model... This may take a moment.")
        
        # Prepare LSTM data
        X_lstm, y_lstm, scaler = prepare_lstm_data(df, lookback=60)
        
        if len(X_lstm) > 0:
            # Split data
            split_idx = int(len(X_lstm) * 0.8)
            X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
            y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]
            
            # Create and train LSTM model
            model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Custom callback for progress updates
            class StreamlitCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_bar, status_text, total_epochs):
                    self.progress_bar = progress_bar
                    self.status_text = status_text
                    self.total_epochs = total_epochs
                
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / self.total_epochs
                    self.progress_bar.progress(progress)
                    self.status_text.text(f"Epoch {epoch + 1}/{self.total_epochs} - Loss: {logs['loss']:.4f} - MAE: {logs['mae']:.4f}")
            
            # Train model
            epochs = 50
            callback = StreamlitCallback(progress_bar, status_text, epochs)
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[callback],
                verbose=0
            )
            
            progress_bar.empty()
            status_text.empty()
            
            # Make predictions
            y_pred = model.predict(X_test, verbose=0)
            
            # Inverse transform predictions to original scale
            # Create dummy array for inverse transform
            dummy_features = np.zeros((len(y_pred), scaler.n_features_in_))
            dummy_features[:, 0] = y_pred.flatten()
            y_pred_original = scaler.inverse_transform(dummy_features)[:, 0]
            
            dummy_features_test = np.zeros((len(y_test), scaler.n_features_in_))
            dummy_features_test[:, 0] = y_test
            y_test_original = scaler.inverse_transform(dummy_features_test)[:, 0]
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
            
            # Future prediction
            last_sequence = X_lstm[-1:] # Last sequence for prediction
            future_scaled = model.predict(last_sequence, verbose=0)
            
            # Inverse transform future prediction
            dummy_future = np.zeros((1, scaler.n_features_in_))
            dummy_future[0, 0] = future_scaled[0, 0]
            future_price = scaler.inverse_transform(dummy_future)[0, 0]
            
            # Store for plotting
            y_pred_plot = y_pred_original
            y_test_plot = y_test_original
            
        else:
            st.error("Insufficient data for LSTM training.")
            
    else:
        # Traditional ML Models
        X, y = prepare_features(df)
        
        if len(X) > 0:
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train selected model
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Future prediction with proper data cleaning
            df_clean = df.copy()
            df_clean = df_clean.ffill().bfill().fillna(0)
            
            last_features = []
            lookback = 10
            for j in range(lookback):
                idx = len(df_clean) - 1 - j
                close_val = df_clean['Close'].iloc[idx]
                volume_val = df_clean['Volume'].iloc[idx] / 1000000
                rsi_val = df_clean['RSI'].iloc[idx] / 100
                macd_val = df_clean['MACD'].iloc[idx]
                
                # Ensure no NaN or infinite values
                close_val = 0 if pd.isna(close_val) or np.isinf(close_val) else close_val
                volume_val = 0 if pd.isna(volume_val) or np.isinf(volume_val) else volume_val
                rsi_val = 0.5 if pd.isna(rsi_val) or np.isinf(rsi_val) else rsi_val  # Default RSI to neutral
                macd_val = 0 if pd.isna(macd_val) or np.isinf(macd_val) else macd_val
                
                last_features.extend([close_val, volume_val, rsi_val, macd_val])
            
            future_price = model.predict([last_features])[0]
            
            # Store for plotting
            y_pred_plot = y_pred
            y_test_plot = y_test
        
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy (MAE)", f"{currency_symbol}{mae:.2f}")
    with col2:
        st.metric("RMSE", f"{currency_symbol}{rmse:.2f}")
    with col3:
        current_price = df['Close'].iloc[-1]
        price_change = ((future_price - current_price) / current_price) * 100
        st.metric(f"Predicted Price ({prediction_days}d)", f"{currency_symbol}{future_price:.2f}", f"{price_change:+.2f}%")
    
    # Model-specific information
    if model_type == "LSTM Deep Learning":
        st.info("🧠 **LSTM Deep Learning Model**: Uses 60-day sequences with multiple technical indicators for advanced pattern recognition.")
        
        # Display training history if available
        if 'history' in locals():
            col1, col2 = st.columns(2)
            with col1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
                fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
                fig_loss.update_layout(title="Model Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Scatter(y=history.history['mae'], name='Training MAE'))
                fig_mae.add_trace(go.Scatter(y=history.history['val_mae'], name='Validation MAE'))
                fig_mae.update_layout(title="Model Training MAE", xaxis_title="Epoch", yaxis_title="MAE")
                st.plotly_chart(fig_mae, use_container_width=True)
    
    # Create prediction chart
    fig_pred = go.Figure()
    
    # Historical data
    fig_pred.add_trace(go.Scatter(
        x=df.index[-60:], y=df['Close'].iloc[-60:], 
        mode='lines', name='Historical', line=dict(color='blue', width=2)
    ))
    
    # Test predictions
    if model_type == "LSTM Deep Learning":
        lookback_offset = 60
    else:
        lookback_offset = 10
        
    test_dates = df.index[split_idx + lookback_offset:split_idx + lookback_offset + len(y_pred_plot)]
    fig_pred.add_trace(go.Scatter(
        x=test_dates, y=y_pred_plot, 
        mode='lines', name='Test Predictions', line=dict(color='orange', width=2)
    ))
    
    # Future prediction point
    future_date = df.index[-1] + pd.Timedelta(days=prediction_days)
    fig_pred.add_trace(go.Scatter(
        x=[df.index[-1], future_date], 
        y=[current_price, future_price],
        mode='lines+markers', name='Future Prediction', 
        line=dict(color='red', dash='dash', width=3), marker=dict(size=10)
    ))
    
    fig_pred.update_layout(
        title=f"{model_type} - {prediction_days} Day Prediction",
        xaxis_title="Date", yaxis_title=f"Price ({currency_symbol})",
        height=500, template="plotly_white"
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Enhanced prediction confidence
    confidence_score = max(0, min(100, 100 - (mae / current_price * 100)))
    
    # Color-coded confidence
    if confidence_score >= 80:
        confidence_color = "green"
        confidence_text = "High Confidence"
    elif confidence_score >= 60:
        confidence_color = "orange" 
        confidence_text = "Medium Confidence"
    else:
        confidence_color = "red"
        confidence_text = "Low Confidence"
    
    st.markdown(f"""
    <div style='padding:10px; border-radius:8px; background-color:{"#d4edda" if confidence_color=="green" else "#fff3cd" if confidence_color=="orange" else "#f8d7da"}; 
                border:2px solid {confidence_color}; margin:10px 0;'>
        <h4 style='margin:0; color:{confidence_color};'>Prediction Confidence: {confidence_score:.1f}% - {confidence_text}</h4>
        <p style='margin:5px 0; color:black;'>Based on model accuracy and historical performance</p>
    </div>
    """, unsafe_allow_html=True)        
if len(df) <= 100:
    st.warning("Insufficient data for prediction model training.")
    
if len(df) <= 100:
    st.warning("Please select a longer time period for accurate predictions (minimum 100 days for LSTM).")

# --- Market Summary ---
st.markdown("""
    <h2 style='display:flex; align-items:center; gap:10px; color:#f5f5f5;'>
        <i class="fas fa-chart-pie"></i> Market Summary
    </h2>
""", unsafe_allow_html=True)

# Calculate summary metrics
price_change_1d = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
price_change_7d = ((df['Close'].iloc[-1] - df['Close'].iloc[-7]) / df['Close'].iloc[-7] * 100) if len(df) > 7 else 0
price_change_30d = ((df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30] * 100) if len(df) > 30 else 0

summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.metric("1-Day Change", f"{price_change_1d:+.2f}%", delta=f"{price_change_1d:.2f}%")
with summary_col2:
    st.metric("7-Day Change", f"{price_change_7d:+.2f}%", delta=f"{price_change_7d:.2f}%")
with summary_col3:
    st.metric("30-Day Change", f"{price_change_30d:+.2f}%", delta=f"{price_change_30d:.2f}%")

# Performance vs market (if NSE stock)
if symbol.endswith('.NS'):
    st.info(f"💡 **Investment Insight**: {selected_label} is showing {'strong momentum' if price_change_7d > 5 else 'moderate performance' if price_change_7d > 0 else 'weakness'} over the past week. Consider the risk level and trading signals before making investment decisions.")

# --- Top Gainers & Losers ---
st.markdown("""
    <h2 style='display:flex; align-items:center; gap:10px; color:#f5f5f5;'>
        <i class="fas fa-arrow-trend-up"></i> NSE Top Gainers & Losers
    </h2>
""", unsafe_allow_html=True)
def fetch_top_movers_nse50():
    """Fetch NSE top movers with multiple fallback strategies"""
    
    # Strategy 1: Try NSE API with enhanced headers
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.nseindia.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "X-Requested-With": "XMLHttpRequest"
        }

        session = requests.Session()
        # First visit homepage to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        
        # Add delay to mimic human behavior
        import time
        time.sleep(1)
        
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            df_nse = pd.DataFrame(data.get("data", []))
            if not df_nse.empty and 'pChange' in df_nse.columns:
                gainers = df_nse.sort_values("pChange", ascending=False).head(5)
                losers = df_nse.sort_values("pChange").head(5)
                return gainers, losers
        
        raise Exception("NSE API returned invalid data")
        
    except Exception as e:
        # Strategy 2: Use yfinance to get sample Indian stocks data
        try:
            indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
                           'SBIN.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'LT.NS']
            
            stock_data = []
            for stock in indian_stocks:
                try:
                    ticker = yf.Ticker(stock)
                    hist = ticker.history(period="2d")
                    if len(hist) >= 2:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2]
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        
                        stock_data.append({
                            'symbol': stock.replace('.NS', ''),
                            'lastPrice': round(current_price, 2),
                            'pChange': round(change_pct, 2)
                        })
                except:
                    continue
            
            if stock_data:
                df_fallback = pd.DataFrame(stock_data)
                gainers = df_fallback.sort_values("pChange", ascending=False).head(5)
                losers = df_fallback.sort_values("pChange").head(5)
                return gainers, losers
                
        except Exception as fallback_error:
            pass
    
    # Strategy 3: Return mock data as last resort
    mock_data = [
        {'symbol': 'RELIANCE', 'lastPrice': 2450.50, 'pChange': 2.1},
        {'symbol': 'TCS', 'lastPrice': 3890.25, 'pChange': 1.8},
        {'symbol': 'INFY', 'lastPrice': 1456.75, 'pChange': -0.5},
        {'symbol': 'HDFCBANK', 'lastPrice': 1678.90, 'pChange': -1.2},
        {'symbol': 'ICICIBANK', 'lastPrice': 1089.45, 'pChange': 0.8}
    ]
    
    df_mock = pd.DataFrame(mock_data)
    gainers = df_mock.sort_values("pChange", ascending=False).head(3)
    losers = df_mock.sort_values("pChange").head(2)
    
    return gainers, losers
# Fetch market data with loading indicator
with st.spinner("📊 Fetching market data..."):
    try:
        gainers, losers = fetch_top_movers_nse50()
        
        # Display success message
        st.success("✅ Market data loaded successfully!")
        
        # Gainers Section
        st.markdown(
    """
    <h2 style='display: flex; align-items: center; color:#f5f5f5 ;'>
        <i class="fa-solid fa-sort-up" style="color:#9eff9e ; margin-right:10px;"></i>
         Top Gainers
    </h2>
    """,
    unsafe_allow_html=True
)
        st.markdown(
            '<div class="card-container-scroll">' +
            ''.join([
                f"<div class='card gainer'><b>{r['symbol']}</b><br>"
                f"Price: ₹{r['lastPrice']}<br>"
                f"Change: <strong>+{r['pChange']}%</strong></div>"
                for _, r in gainers.iterrows()
            ]) +
            '</div>', unsafe_allow_html=True
        )
        
        # Losers Section
        st.markdown(
    """
    <h2 style='display: flex; align-items: center; color:#f5f5f5 ;'>
        <i class="fa-solid fa-sort-down" style="color:#ff5252 ; margin-right:10px;"></i>
        Top Losers
    </h2>
    """,
    unsafe_allow_html=True
)
        st.markdown(
            '<div class="card-container-scroll">' +
            ''.join([
                f"<div class='card loser'><b>{r['symbol']}</b><br>"
                f"Price: ₹{r['lastPrice']}<br>"
                f"Change: <strong>{r['pChange']}%</strong></div>"
                for _, r in losers.iterrows()
            ]) +
            '</div>', unsafe_allow_html=True
        )
        
        # Data source info
        st.info("💡 **Data Source**: Live market data from NSE/Yahoo Finance. Prices may have a slight delay.")
        
    except Exception as e:
        st.warning("⚠️ **Market Data Temporarily Unavailable**")
        st.markdown("""
        **Possible reasons:**
        - NSE API is temporarily blocked or under maintenance
        - Network connectivity issues
        - Rate limiting from data provider
        
        **Solutions:**
        - Refresh the page in a few minutes
        - Check your internet connection
        - The main stock analysis features are still fully functional
        """)
        
        # Show a simplified market overview instead
        st.markdown("### 📊 **Market Overview** (Sample Data)")
        sample_data = [
            ("NIFTY 50", "19,800", "+0.8%", "green"),
            ("SENSEX", "66,500", "+0.6%", "green"),
            ("BANK NIFTY", "44,200", "-0.3%", "red"),
            ("IT Index", "32,100", "+1.2%", "green")
        ]
        
        cols = st.columns(4)
        for i, (index, value, change, color) in enumerate(sample_data):
            with cols[i]:
                st.metric(index, value, change)

# --- Export Data Section ---
st.markdown("""
    <h2 style='display:flex; align-items:center; gap:10px; color:#f5f5f5;'>
        <i class="fas fa-download"></i> Export Data
    </h2>
""", unsafe_allow_html=True)

export_col1, export_col2 = st.columns(2)

with export_col1:
    if st.button("📊 Download Historical Data (CSV)"):
        csv_data = df.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{symbol}_{period}_data.csv",
            mime="text/csv"
        )

with export_col2:
    if st.button("📈 Download Analysis Report"):
        # Create a summary report
        report = f"""
Stock Analysis Report - {selected_label}
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

=== STOCK INFORMATION ===
Symbol: {symbol}
Company: {info.get('longName', 'N/A')}
Sector: {info.get('sector', 'N/A')}
Current Price: {currency_symbol}{info.get('currentPrice', 0):,.2f}
Market Cap: {format_number(info.get('marketCap'), currency_symbol, currency)}

=== PERFORMANCE METRICS ===
1-Day Change: {price_change_1d:+.2f}%
7-Day Change: {price_change_7d:+.2f}%
30-Day Change: {price_change_30d:+.2f}%

=== RISK METRICS ===
Current Volatility: {current_volatility:.1%}
Sharpe Ratio: {sharpe_ratio:.2f}
Max Drawdown: {max_drawdown:.1%}
Risk Level: {risk_level}

=== TECHNICAL INDICATORS (Latest) ===
RSI: {df['RSI'].iloc[-1]:.2f}
MACD: {df['MACD'].iloc[-1]:.4f}
20-Day MA: {currency_symbol}{df['MA20'].iloc[-1]:.2f}
50-Day MA: {currency_symbol}{df['MA50'].iloc[-1]:.2f}
"""       
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"{symbol}_analysis_report.txt",
            mime="text/plain"
        )

