import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Graph Predictor Using Keras")
stock = st.text_input("Enter the Stock ID","GOOG")
from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock,start,end)
# google_data = pd.read_csv("google_stock_data.csv")

model = load_model("stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize,values,full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,"Orange")
    plt.plot(full_data.Close,"b")
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader("Original Close Price & Mean Avg 200 Days Timestapms")
google_data["MA_200_days"] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,5),google_data["MA_200_days"],google_data,0))

st.subheader("Original Close Price & Mean Avg 100 Days Timestapms")
google_data["MA_100_days"] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,5),google_data["MA_100_days"],google_data,0))

st.subheader("Original Close Price & Mean Avg 200 Days & Mean Avg 100 Days Timestapms")
st.pyplot(plot_graph((15,5),google_data["MA_100_days"],google_data,1,google_data["MA_200_days"]))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[["Close"]])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data,y_data)
predictions = model.predict(x_data)

inverse_pred = scaler.inverse_transform(predictions)
inverse_y_test = scaler.inverse_transform(y_data)


ploting_data = pd.DataFrame(
    {
        "Original_Test_Data" : inverse_y_test.reshape(-1),
        "Predicted_Test_Data" : inverse_pred.reshape(-1)
    },
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original Data v/s Predicted Data")
st.write(ploting_data)

st.subheader("Original Close Price vs Predcited Close Price")
fig = plt.figure(figsize=(15,5))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data],axis=0))
plt.legend(["Data unused","Original Test Data","Predicted Test Data"])
st.pyplot(fig)
