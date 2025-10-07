# 📈 Stock Market Analysis & Prediction Dashboard

A comprehensive web-based stock market analysis tool built with Streamlit that provides real-time stock data, technical analysis, and AI-powered price predictions.

## ✨ Features

### 📊 **Real-time Stock Data**
- Live stock prices for Indian (NSE) and US markets
- Company information and key metrics
- Market capitalization, volume, and trading data

### 📈 **Advanced Technical Analysis**
- Interactive candlestick charts with volume
- Moving averages (MA20, MA50, MA200)
- Bollinger Bands for volatility analysis
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic oscillator
- Volatility indicators

### 🤖 **AI-Powered Predictions**
- **Linear Regression** - Simple trend analysis
- **Random Forest** - Advanced ensemble learning
- **LSTM Deep Learning** - Neural network for pattern recognition
- Configurable prediction periods (5-20 days)
- Model accuracy metrics and confidence scores

### 📱 **Risk Analysis**
- Volatility assessment
- Sharpe ratio calculation
- Maximum drawdown analysis
- Value at Risk (VaR) calculations
- Risk level categorization

### 🎯 **Market Summary**
- Performance tracking (1-day, 7-day, 30-day)
- Investment insights and recommendations
- Trend analysis and momentum indicators

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install TensorFlow for LSTM predictions**
   ```bash
   pip install tensorflow
   ```

4. **Run the application**
   ```bash
   streamlit run final_project.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in terminal

## 📱 Mobile Optimization

The dashboard is optimized for mobile devices with:
- Responsive card layouts that stack on smaller screens
- Horizontal scrolling for data tables
- Touch-friendly interactive charts
- Optimized font sizes and spacing
- Mobile-first design approach

## 🎮 How to Use

### 1. **Select a Stock**
- Use the search bar in the sidebar to find stocks
- Choose from popular Indian (NSE) and US stocks
- Select your preferred time period (1 month to 5 years)

### 2. **Analyze Technical Indicators**
- View comprehensive charts with multiple indicators
- Examine RSI for overbought/oversold conditions
- Check MACD for trend changes
- Monitor Bollinger Bands for volatility

### 3. **Get AI Predictions**
- Choose your prediction model (Linear, Random Forest, or LSTM)
- Select prediction period (5-20 days)
- Review accuracy metrics and confidence scores
- Analyze future price projections

### 4. **Assess Risk**
- Check volatility levels and risk categorization
- Review Sharpe ratio for risk-adjusted returns
- Monitor maximum drawdown for worst-case scenarios
- Use VaR for daily risk assessment

## 📊 Supported Stocks

### Indian Stocks (NSE)
- Reliance Industries, TCS, Infosys
- HDFC Bank, ICICI Bank, SBI
- Tata Motors, Asian Paints, Bajaj Finance
- And 20+ more major Indian companies

### US Stocks
- Apple, Google, Amazon, Microsoft
- Tesla, NVIDIA, Meta, Netflix
- Coca-Cola, McDonald's, Visa, JPMorgan
- And more major US companies

## 🔧 Technical Details

### Dependencies
- **Streamlit** - Web framework
- **yfinance** - Stock data API
- **Pandas** - Data manipulation
- **Plotly** - Interactive charts
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **TensorFlow** - Deep learning (optional)

### Data Sources
- Yahoo Finance API via yfinance
- Real-time and historical stock data
- Company fundamentals and metrics

### Machine Learning Models
1. **Linear Regression** - Fast, simple trend analysis
2. **Random Forest** - Ensemble method with feature importance
3. **LSTM Neural Network** - Deep learning for sequence prediction

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only.**

- Not financial advice or investment recommendations
- Past performance doesn't guarantee future results
- Always consult with financial professionals
- Consider your risk tolerance before investing
- Market predictions are inherently uncertain

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check that all dependencies are installed correctly
2. Ensure you have a stable internet connection for data fetching
3. Try refreshing the browser if charts don't load
4. For LSTM features, make sure TensorFlow is installed

---

**Happy Trading! 📈**
