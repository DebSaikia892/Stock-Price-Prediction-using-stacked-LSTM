import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from keras import models
from keras.models import load_model

st.title('Stock Market Prediction')

user_input=st.text_input('Enter Stock Ticker','AAPL')
df=pd.read_csv('AAPL.csv')

#Describing the data
st.subheader('Data from 2010-2019')
st.write(df.describe())

#Visualizing the data
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100=df.close.rolling(100).mean()
ma200=df.close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.close,'b')
st.pyplot(fig)

#Splitting the data into training and testing
data_training=pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)


#Splitting data into x_train and y_train
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100:i])
    y_train.append(data_training_array[i, 0])

#load my model
model=load_model('keras_model.h5')

x_train, y_train = np.array(x_train), np.array(y_train)

#Testing part
past_100_days=data_training.tail(100)
final_df=past_100_days._append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor


st.subheader('Predicted vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)