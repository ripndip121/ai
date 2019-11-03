import yfinance as yf

# Get the data for the stock Apple by specifying the stock ticker, start date, and end date
data = yf.download('AAPL', '2019-01-01', '2019-11-02')
closing = data[['Adj Close']]
