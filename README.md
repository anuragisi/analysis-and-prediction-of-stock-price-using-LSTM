#Analysis and Prediction of stock price using LSTM

Stock price prediction refers to understanding various aspects of the stock market that can influence the price of a stock, and based on these potential factors, build a model to predict the stock's price. This can help individuals and institutions speculate on the stock price trend and help them decide whether to buy or short the stock price to maximize their profit. While using Machine Learning and Time Series helps us to discover the future value of a particular stock and other financial assets traded on an exchange. The entire idea of analysis and prediction is to gain significant profits.

<br>
<b>Focus areas for Analysis:</b>
<br>
 <ul>
  <li>The change in closing price of the stock over time.</li>
  <li>Visualization of Candlestick Monthly data. </li>
  <li>The % daily return of the stock.</li>
   <li>The moving average of various stocks.</li>
</ul> 

<b>Prediction:</b>
<ul>
  <li>We will be predicting future stock behaviour by predicting the closing price of the stock using LSTM.</li>
</ul>
<br>

#Import Libraries
<pre>
!pip install datetime numpy pandas yfinance seaborn matplotlib
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
</pre>

#Dataset
<p>I have taken the stock price data of United Breweries Holdings Limited from Yahoo Finance from 1st Jan 2022 to 1st Jan 2023.</p>
<p>Time Period of Data: Define the timeframe for which you want to fetch data.</p>
<pre>
start_date = datetime.datetime(2020, 1, 15)
end_date = datetime.datetime(2023, 12, 31)
</pre>

<p>
 Loading Data from Yahoo Finance
</p>
<pre>
 df = yf.download('UBL.NS', start_date, end_date)
</pre>
<p>View Dataframe</p>
<pre>df</pre>
<samp>
 <img width="580" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/62b77e1e-68b2-4298-84d6-7c33d434f217">
</samp>

<p>Check index</p>
<pre>print(df.index)</pre>
<samp>
 <img width="549" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/19d056de-9cf7-41e6-863a-4da809e12394">
</samp>

<p>Reset Index</p>
<pre> df1 = df.reset_index()
df1['Date'] = pd.to_datetime(df1['Date'])
df1</pre>
<samp>
 <img width="589" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/b75e3f13-c937-4dc2-af75-01496dd04d2f">
</samp>
<p>
 Converting from Daily to Monthly Frequency data
</p>
<pre>
monthly_data = df.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
monthly_data.head()
</pre>
<samp>
 <img width="837" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/255c9001-8d9a-4e5a-8258-25bd01fe887e">
</samp>
<p>
 Plot - Line and Frequency -Daily closing Price
</p>
<pre>
 # Plotting
plt.figure(figsize=(10, 6))
#plt.plot(df['Date'], df['Open'], label='Open')
#plt.plot(df['Date'], df['High'], label='High')
#plt.plot(df['Date'], df['Low'], label='Low')
plt.plot(df.index, df['Close'], label='Close')

plt.title('UBL Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()
</pre>
<samp>
<img width="733" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/f5114aa2-3174-4902-af1e-136b2334a824">
</samp>
<p>
 Plot - Candlestick and Frequency - Monthly OHLC Volume Data
</p>
<pre>
#Plotting monthly candlestick chart with a separate volume plot with MA(20)
#mpf.plot(monthly_data, type='candle', style='charles', volume=True, mav=(20), show_nontrading=True, addplot=mpf.make_addplot(monthly_data['Volume'], panel=1, ylabel='Volume'),tight_layout=True, figratio=(16, 9), scale_width_adjustment=dict(volume=0.7, candle=1))

#Plotting monthly candlestick chart with a separate volume plot
mpf.plot(monthly_data, type='candle', style='charles', volume=True, show_nontrading=True, tight_layout=True, figratio=(16, 9), scale_width_adjustment=dict(volume=0.7, candle=1))
</pre>
<samp>
<img width="1229" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/d252c914-c6a4-4c21-a06b-e96670a51a00">
</samp>

<p>
 Total Rows & Columns
</p>
<pre>
df.shape
</pre>
<samp>
 <img width="157" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/25e20156-cc4f-4f85-9a69-d1652ecd5a08">
</samp>
<p>
 Data Information
</p>
<pre>
 df.info()
</pre>
<samp>
 <img width="406" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/82694be3-c735-450a-ae14-61b17913aff6">
</samp>
<p>
 <b>Data Quality Check</b>
</p>
<p>
 Duplicate Values
</p>
<pre>
 len(df[df.duplicated()])
</pre>
<samp>
 <img width="227" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/1070cf28-131f-4817-8691-f2929567cd5d">
</samp>
<p>
 Missing Values/Null Values
</p>
<pre>
 print(df.isnull().sum())
</pre>
<samp>
<img width="231" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/c77fe60e-f740-4f2c-a2d2-5e414f7efca4">
</samp>
<p>
 Variable Information
</p>
<pre>
#Columns
df.columns
</pre>
<samp>
 <img width="580" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/c8c7d121-55e5-4f8e-a25e-5c66106c67ce">
</samp>
<pre>
 #Describe
df.describe()
</pre>
<samp>
 <img width="1106" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/6e227a75-27a0-443e-8b44-a99f23efdc3d">
</samp>
<pre>
 #Check unique values for each variable
for i in df.columns.tolist():
  print("No. of unique values in ",i,"is",df[i].nunique(),".")
</pre>
<samp>
<img width="477" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/cf66e481-a688-40d5-abca-2f14de136398">
</samp>

#Analysis
<p>
 Plotting Moving Average (50,200) is a simple technal analysis that smooths out price data.
</p>
<pre>
 ma_day = [50, 200]

plt.figure(figsize=(10, 6))

# Plot Close price
plt.plot(df.index, df['Close'], label='Close')

# Plot Moving Averages
for ma in ma_day:
    column_name = f"MA for {ma} days"
    df[column_name]=df['Close'].rolling(ma).mean()
    plt.plot(df.index, df[column_name], label=column_name)

plt.title('UBL Daily Close Prices and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
</pre>
<samp>
<img width="620" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/ea9c9255-afbb-463c-a37d-1e057362a90b">
</samp>
<p>
 Average Daily Returns
</p>
<pre>
 # Calculate daily return percentage
df['Daily Return'] = df['Close'].pct_change()


plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Daily Return'], linestyle='--', marker='o', label='Daily Return')

plt.title('Daily Return Percentage')
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.legend()
plt.grid(True)
plt.show()
</pre>
<samp>
<img width="631" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/542fec78-9d72-4fbe-bae9-2e976ae41e70">
</samp>
<pre>
plt.figure(figsize=(12, 9))
df['Daily Return'].hist(bins=50, alpha=0.5, label='UBL')

plt.xlabel('Daily Return')
plt.ylabel('Counts')
plt.title('Daily Return of UBL using histogram')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
</pre>
<samp>
<img width="837" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/7527c022-b3c8-4d96-b0e8-e28e6e336955">
</samp>

# 2. Prediction using LTSM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture well-suited for sequence prediction problems.
<br>
<p>
 2.0 Before Prediction
</p>
<pre>
 plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()
</pre>
<samp>
 <img width="934" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/c1508dbf-aaf3-4adf-82b9-55a0d146c92f">
</samp>
<p>2.1 Data Prepartion</p>
<pre>
 # Create a new dataframe with only the 'Close column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array because ML/DL libraries requires numpy arrays as inputs
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .80 ))

training_data_len
</pre>
<samp>
<img width="594" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/bc9e62e0-3748-4df6-8e7c-da161f0c7e62">
</samp>
<p>
 2.2 Data Scaling
</p>
<pre>
 # Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data
</pre>
<samp>
<img width="247" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/7549a942-6978-4089-b037-c4e3a7b405f7">
</samp>
<p>
 2.3 Creating Training Data
</p>
<pre>
 # Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape
</pre>
<samp>
 <img width="809" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/73e7cfa2-cc65-4072-b237-81b875af00fc">
</samp>
<p>2.4 Model Building</p>
<pre>
 from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
</pre>
<p>
2.5 Model Training
</p>
<pre>
# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
</pre>
<samp>
 <img width="464" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/34c84679-a517-4a94-bd34-cf82cc032c21">
</samp>
<p>
 2.6 Creating Testing Data
</p>
<pre>
 # Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
</pre>
<p>
 2.7 Making Predictions
</p>
<pre>
 # Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
</pre>
<Samp>
 <img width="974" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/959cf1ab-ca0c-4ea5-9b68-79c6f4b3be8c">
</Samp>
<p>
2.8 Model Evaluations 
</p>
<pre>
 # Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse
</pre>
<samp>
 <img width="362" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/dc0bf781-67cb-4e05-b8e8-2f9f457eb43b">
</samp>
<p>
 2.9 Visualization
</p>
<pre>
 # Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
</pre>
<samp>
 <img width="935" alt="image" src="https://github.com/anuragprasad95/analysis-and-prediction-of-stock-price-using-LSTM/assets/3609255/43051c6e-a1dd-4a04-a6d5-aac73e8cba39">
</samp>
