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
<code>
#!pip install datetime yfinance seaborn matplotlib
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
</code>
