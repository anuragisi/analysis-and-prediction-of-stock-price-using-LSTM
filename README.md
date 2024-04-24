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
