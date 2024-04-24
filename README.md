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
