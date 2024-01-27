#!/usr/bin/env python
# coding: utf-8

# In[94]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data


# In[95]:


df=pd.read_csv('AAPL.csv')


# In[96]:


df.head()


# In[97]:


df.tail()


# In[98]:


df=df.drop(['adjClose','date','symbol','adjHigh','adjLow','adjOpen','adjVolume','divCash','splitFactor'],axis=1)


# In[99]:


df.head()


# In[100]:


plt.plot(df.close)


# In[101]:


df


# In[102]:


ma100=df.close.rolling(100).mean()
ma100


# In[103]:


plt.figure(figsize=(12,6))
plt.plot(df.close)
plt.plot(ma100,'r')


# In[104]:


ma200=df.close.rolling(200).mean()
ma200


# In[105]:


plt.figure(figsize=(12,6))
plt.plot(df.close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')


# In[106]:


df.shape


# In[107]:


#Splitting the data into training and testing
data_training=pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)


# In[108]:


data_training.head()


# In[109]:


data_testing.head()


# In[110]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))


# In[111]:


data_training_array=scaler.fit_transform(data_training)
data_training_array


# In[112]:


data_training_array.shape


# In[113]:


x_train=[]
y_train=[]
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)        


# In[114]:


x_train.shape


# In[115]:


#ML Model


# In[116]:


from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# In[117]:


model=Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,
               input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))



model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))



model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))



model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(units=1))


# In[118]:


model.summary()


# In[119]:


model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)


# In[120]:


model.save('keras_model.keras')


# In[121]:


data_testing.head()


# In[122]:


data_training.tail()


# In[123]:


past_100_days=data_training.tail(100)


# In[124]:


final_df=past_100_days.append(data_testing,ignore_index=True)


# In[125]:


final_df.head()


# In[126]:


input_data=scaler.fit_transform(final_df)
input_data


# In[127]:


input_data.shape


# In[128]:


x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


# In[129]:


x_test,y_test=np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)


# In[130]:


#Making Predictions
y_predicted=model.predict(x_test)


# In[131]:


y_predicted.shape


# In[134]:


y_test


# In[132]:


y_predicted


# In[135]:


scaler.scale_


# In[136]:


scale_factor=1/0.02099517
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor


# In[138]:


plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:




