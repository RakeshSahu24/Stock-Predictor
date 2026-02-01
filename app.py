import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
import traceback
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = r'd:\Python\Stock\Stock_Predictions_Model.keras'

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: var(--color-background);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Card styling */
    .custom-card {
        background-color: var(--color-surface);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid var(--color-card-border);
    }
    
    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background-color: rgba(33, 128, 141, 0.1);
        border-left: 4px solid var(--color-primary);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: rgba(33, 128, 141, 0.1);
        border-left: 4px solid var(--color-success);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: rgba(168, 75, 47, 0.1);
        border-left: 4px solid var(--color-warning);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Chart container */
    .chart-container {
        background-color: var(--color-surface);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- MODEL LOADING ----------------------
model = None
try:
    if not os.path.exists(MODEL_PATH):
        st.error(f"🚫 Model file not found at: {MODEL_PATH}")
        st.stop()

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception:
        from keras.models import load_model as _km
        model = _km(MODEL_PATH)
except Exception as exc:
    tb = traceback.format_exc()
    st.error("❌ Failed to load the Keras model.")
    st.code(tb)
    st.stop()

# ---------------------- HEADER ----------------------
st.markdown("""
    <div class="main-header">
        <h1>📈 Stock Price Predictor</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">AI-Powered Stock Market Analysis & Forecasting</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Stock Selection
    st.markdown("#### 📊 Stock Selection")
    stock_symbols = [
        'RELIANCE.NS (Reliance Industries)', 'TCS.NS (Tata Consultancy Services)',
        'HDFCBANK.NS (HDFC Bank)', 'INFY.NS (Infosys)', 'ICICIBANK.NS (ICICI Bank)',
        'HINDUNILVR.NS (Hindustan Unilever)', 'BAJFINANCE.NS (Bajaj Finance)',
        'WIPRO.NS (Wipro Ltd.)', 'TATAMOTORS.NS (Tata Motors)', 'SBIN.NS (State Bank of India)',
        'AAPL (Apple Inc.)', 'MSFT (Microsoft Corp.)', 'AMZN (Amazon.com Inc.)',
        'GOOGL (Alphabet Inc.)', 'TSLA (Tesla Inc.)', 'NVDA (NVIDIA Corp.)',
        'META (Meta Platforms Inc.)', 'JPM (JPMorgan Chase & Co.)', 'V (Visa Inc.)', 'WMT (Walmart Inc.)'
    ]

    input_method = st.radio("Choose input method:", ["📋 Select from list", "✍️ Manual entry"])

    if input_method == "📋 Select from list":
        stock = st.selectbox('Select stock', stock_symbols)
        stock = stock.split(' ')[0]
    else:
        user_input = st.text_input('Enter Stock Symbol', placeholder='e.g., TATASTEEL.NS or AAPL')
        stock = user_input.strip().upper()

    if not stock:
        st.error("⚠️ Please provide a stock symbol.")
        st.stop()

    st.markdown("---")
    
    # Mode Selection
    st.markdown("#### 🎯 Analysis Mode")
    data_mode = st.radio("Choose Mode:", ["📊 Analysis", "🔮 Prediction"])
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        This application uses deep learning (LSTM) to:
        - Analyze historical stock trends
        - Predict next-day closing prices
        - Visualize moving averages
        
        **Note:** Predictions are estimates based on historical patterns.
        """)

# ---------------------- DATA FETCHING ----------------------
with st.spinner('🔄 Fetching stock data...'):
    try:
        if data_mode == "📊 Analysis":
            data = yf.download(stock, start="2015-04-01", end="2025-04-01", progress=False)
        else:
            data = yf.download(stock, period="2y", interval="1d", progress=False)

        if data is None or data.empty:
            st.error(f"🚫 No data found for '{stock}'.")
            st.stop()

    except Exception as e:
        st.error("❌ Error fetching stock data. Check internet connection or ticker symbol.")
        st.write(e)
        st.stop()

close_count = data['Close'].dropna().shape[0] if 'Close' in data.columns else 0
if close_count < 101:
    st.error(f"⚠️ Not enough data ({close_count} days). Need at least 101 trading days.")
    st.stop()

# ---------------------- STOCK INFO CARD ----------------------
latest_date = data.index[-1].strftime("%Y-%m-%d")
last_close = float(data['Close'].iloc[-1])  # Convert to float
prev_close = float(data['Close'].iloc[-2])  # Convert to float
price_change = last_close - prev_close
price_change_pct = (price_change / prev_close) * 100

currency_symbol = "₹" if stock.endswith(".NS") else "$"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="custom-card">
            <h4 style="margin:0; color: var(--color-text-secondary);">Stock</h4>
            <h2 style="margin:0.5rem 0; color: var(--color-text);">{stock}</h2>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="custom-card">
            <h4 style="margin:0; color: var(--color-text-secondary);">Latest Price</h4>
            <h2 style="margin:0.5rem 0; color: var(--color-text);">{currency_symbol}{last_close:,.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

with col3:
    change_color = "#28a745" if price_change >= 0 else "#dc3545"
    st.markdown(f"""
        <div class="custom-card">
            <h4 style="margin:0; color: var(--color-text-secondary);">Daily Change</h4>
            <h2 style="margin:0.5rem 0; color: {change_color};">{currency_symbol}{price_change:+,.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="custom-card">
            <h4 style="margin:0; color: var(--color-text-secondary);">Change %</h4>
            <h2 style="margin:0.5rem 0; color: {change_color};">{price_change_pct:+.2f}%</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <div class="info-box">
        📅 <b>Latest available data:</b> {latest_date} | 📊 <b>Total data points:</b> {close_count} days
    </div>
""", unsafe_allow_html=True)

# ---------------------- DATA VIEW ----------------------
with st.expander("📋 View Raw Stock Data"):
    st.dataframe(data, use_container_width=True)

# ---------------------- PREPROCESSING ----------------------
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# ---------------------- MOVING AVERAGES ----------------------
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

# ---------------------- ANALYSIS MODE ----------------------
if data_mode == "📊 Analysis":
    st.markdown("## 📈 Technical Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Price vs MA50", "📉 MA50 vs MA100", "📈 MA100 vs MA200", "🎯 Model Performance"])
    
    with tab1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(ma_50_days, 'r', label='MA50', linewidth=2)
        plt.plot(data.Close, 'g', label='Stock Price', linewidth=1.5, alpha=0.7)
        plt.title('Stock Price vs 50-Day Moving Average', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(ma_50_days, 'r', label='MA50', linewidth=2)
        plt.plot(ma_100_days, 'b', label='MA100', linewidth=2)
        plt.plot(data.Close, 'g', label='Stock Price', linewidth=1.5, alpha=0.7)
        plt.title('Stock Price vs MA50 vs MA100', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig3 = plt.figure(figsize=(12, 6))
        plt.plot(ma_100_days, 'r', label='MA100', linewidth=2)
        plt.plot(ma_200_days, 'b', label='MA200', linewidth=2)
        plt.plot(data.Close, 'g', label='Stock Price', linewidth=1.5, alpha=0.7)
        plt.title('Stock Price vs MA100 vs MA200', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        # Model predictions
        x, y = [], []
        for i in range(100, data_test_scale.shape[0]):
            x.append(data_test_scale[i - 100:i])
            y.append(data_test_scale[i, 0])

        x, y = np.array(x), np.array(y)
        predict = model.predict(x, verbose=0)
        scale = 1 / scaler.scale_
        predict = predict * scale
        y = y * scale

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig4 = plt.figure(figsize=(12, 6))
        plt.plot(predict, 'r', label='Predicted Price', linewidth=2)
        plt.plot(y, 'g', label='Actual Price', linewidth=2)
        plt.title('Model Performance: Predicted vs Actual Price', fontsize=14, fontweight='bold')
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        st.pyplot(fig4)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate accuracy metrics
        mse = np.mean((predict.flatten() - y) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predict.flatten() - y))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Squared Error", f"{mse:,.2f}")
        with col2:
            st.metric("Root Mean Squared Error", f"{rmse:,.2f}")
        with col3:
            st.metric("Mean Absolute Error", f"{mae:,.2f}")

# ---------------------- PREDICTION MODE ----------------------
else:
    st.markdown("## 🔮 Next Day Price Prediction")
    
    # Get last 100 scaled prices
    last_100_scaled = data_test_scale[-100:]
    curr_input = last_100_scaled.reshape(1, -1, 1)

    # Predict next day
    with st.spinner('🧠 Generating prediction...'):
        next_day_scaled = model.predict(curr_input, verbose=0)[0][0]

    # Rescale prediction
    next_day_price = scaler.inverse_transform(np.array(next_day_scaled).reshape(-1, 1))[0][0]

    # Get last close and date safely
    last_close_price = float(data['Close'].iloc[-1])
    latest_date = data.index[-1].strftime("%Y-%m-%d")

    # Calculate % change
    perc_change = ((next_day_price - last_close_price) / last_close_price) * 100

    # Large prediction card
    direction_emoji = "🟢" if perc_change > 0 else "🔴"
    direction_text = "UP" if perc_change > 0 else "DOWN"
    card_gradient = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)" if perc_change > 0 else "linear-gradient(135deg, #eb3349 0%, #f45c43 100%)"
    
    st.markdown(f"""
        <div style="background: {card_gradient}; border-radius: 16px; padding: 2rem; color: white; text-align: center; margin: 2rem 0; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
            <h3 style="margin: 0; font-size: 1.2rem; opacity: 0.9;">Predicted Next Day Closing Price</h3>
            <h1 style="margin: 1rem 0; font-size: 3rem; font-weight: bold;">{currency_symbol}{next_day_price:,.2f}</h1>
            <div style="font-size: 1.5rem; margin: 1rem 0;">
                {direction_emoji} <span style="font-weight: bold;">{direction_text}</span> by {abs(perc_change):.2f}%
            </div>
            <p style="margin: 0; opacity: 0.8;">vs {currency_symbol}{last_close_price:,.2f} on {latest_date}</p>
        </div>
    """, unsafe_allow_html=True)

    # Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        future_date = data.index[-1] + pd.Timedelta(days=1)
        fig5 = plt.figure(figsize=(12, 6))
        plt.plot(data.index[-100:], data['Close'][-100:], label='Last 100 Days', color='#667eea', linewidth=2)
        plt.scatter(future_date, next_day_price, color='red', s=200, label='Predicted', zorder=5, edgecolors='darkred', linewidth=2)
        plt.axhline(y=last_close_price, color='gray', linestyle='--', alpha=0.5, label='Last Close')
        plt.title('Price Trend & Next Day Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig5)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Prediction Summary")
        st.markdown(f"""
            <div class="custom-card">
                <p><b>📅 Last Close Date:</b><br>{latest_date}</p>
                <p><b>💰 Last Closing Price:</b><br>{currency_symbol}{last_close_price:,.2f}</p>
                <p><b>🧠 Predicted Price:</b><br>{currency_symbol}{next_day_price:,.2f}</p>
                <p><b>📈 Expected Change:</b><br>{direction_emoji} {abs(perc_change):.2f}%</p>
                <p><b>💵 Price Difference:</b><br>{currency_symbol}{abs(next_day_price - last_close_price):,.2f}</p>
            </div>
        """, unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown("""
    <div class="warning-box">
        ⚠️ <b>Disclaimer:</b> Predictions are based on historical data and machine learning models. 
        They should not be considered as financial advice. Always do your own research before making investment decisions.
    </div>
""", unsafe_allow_html=True)
