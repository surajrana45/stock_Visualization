import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from pandas_datareader import data as pdr
from streamlit_autorefresh import st_autorefresh
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

yf.pdr_override()

from datetime import datetime, timedelta

end = datetime.now()
start = datetime(2014, end.month, end.day)

# Load model
model = load_model(r'''C:\\Users\\HP\Documents\Stock\Stock Visualization Model.keras''')

# Streamlit app setup
st.set_page_config(
    page_title="Stock Visualization and Forcasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('📈 Stock Visualization and Forcasting')
st.markdown("""
<style>
h1 {
    text-align: center;
    font-family: 'Arial', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header('Configuration')
stock = st.sidebar.text_input('Enter the Stock Symbol', 'GOOG')
price_threshold = st.sidebar.number_input('Set Alert Price Threshold', min_value=0, value=1000)
email = st.sidebar.text_input('Enter your email for alerts', '')
APP_PASSWORD = st.sidebar.text_input('Enter your APP Password for alerts', '')


count = st_autorefresh(interval=300000, limit=100, key="rerun")

# Fetching stock data
df = pdr.get_data_yahoo(stock, start=start, end=datetime.now())

st.header(f'Stock Data for {stock.upper()}')
st.write(df.tail())
# st.write(df.shape)

#password  xuak dzbt ojks thzc// 
def send_email_alert(to_email, stock_symbol, current_price):
    try:
        from_email = email 
        from_password = APP_PASSWORD     

        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)

        # Create the email content
        subject = f"Stock Alert for {stock_symbol}"
        body = f"The stock price for {stock_symbol} has crossed your threshold. The current price is ${current_price:.2f}."
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        server.send_message(msg)
        server.quit()
        st.success(f"Alert email sent to {to_email}")
    except Exception as e:
        st.error(f"Failed to send email alert: {e}")

# Ensure dataset is not empty
if df.empty:
    st.error("The dataset is empty. Please check the stock symbol and try again.")
else:
    # Data preparation
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .70))

    # Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create the testing data set
    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate metrics
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    mae = np.mean(np.abs(predictions - y_test))
    mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100
    accuracy = 100 - mape

    # Train and validation data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Determine future trend
    trend = "up" if predictions[-1] > predictions[-2] else "down"
    trend_color = "green" if trend == "up" else "red"
    trend_emoji = "📈" if trend == "up" else "📉"


    # Check if the latest stock price crosses the threshold
    latest_price = df['Close'].iloc[-1]
    if latest_price > price_threshold and email:
        send_email_alert(email, stock, latest_price)

    # Moving averages
    st.subheader('Moving Averages')

    ma_day = [10, 20, 50]
    for ma in ma_day:
        df[f"MA for {ma} days"] = df['Adj Close'].rolling(window=ma).mean()

    fig, ax = plt.subplots(figsize=(16, 8))
    plt.plot(df['Adj Close'], label='Adj Close', linewidth=2)
    for ma in ma_day:
        plt.plot(df[f"MA for {ma} days"], label=f'MA for {ma} days', linewidth=1.5)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price USD ($)', fontsize=12)
    plt.title(f'Moving Averages for {stock.upper()}', fontsize=16)
    plt.legend()
    st.pyplot(fig)


    # Plotting
    st.subheader('Closing Price')
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.plot(df['Close'], label='Actual Prices')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price USD ($)', fontsize=12)
    plt.title(f'Closing Price History of {stock.upper()}', fontsize=16)
    plt.legend()
    st.pyplot(fig)

    st.subheader('Model Predictions')
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    plt.plot(train['Close'], label='Train')
    plt.plot(valid[['Close', 'Predictions']], label=['Validation', 'Predictions'])
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price USD ($)', fontsize=12)
    plt.title(f'Model Predictions for {stock.upper()}', fontsize=16)
    plt.legend()
    st.pyplot(fig2)

    st.subheader('Prediction Table')
    st.write(valid)


    st.subheader('Model Accuracy Metrics')
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")
    st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{mape:.2f}%")
    st.metric(label="Total Accuracy", value=f"{accuracy:.2f}%")

    st.header('📊 Future Trend Analysis')
    st.markdown(f"""
    ### Future Trend: 
    - The stock price is predicted to go **{trend.upper()}** {trend_emoji}
    """)

    