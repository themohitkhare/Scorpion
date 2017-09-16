import quandl
from sklearn import preprocessing

x = preprocessing.MinMaxScaler()
print(x)
code = "500010"
quandl.ApiConfig.api_key = "tHhnk2LX1KrKyxKKyhaz"
data = quandl.get("BSE/BOM500010")
data.drop(['WAP', 'No. of Trades', 'Total Turnover', 'Deliverable Quantity', '% Deli. Qty to Traded Qty', 'Spread H-L',
           'Spread C-O'], 1, inplace=True)
data.rename(columns={'No. of Shares': 'Volume', 'Close': 'Adj Close'}, inplace=True)
data['date'] = data.index
data.set_index('date', inplace=True)
data['Pct'] = data['Adj Close'].pct_change()
print(x.fit_transform(data.Volume.values.reshape(-1, 1)))
# df = quandl.get('BSE/BOM' + stock_name)
# df = quandl.get_table('BSE.OIL.1')
# df.drop(['ticker', 'open', 'high', 'low', 'close', 'ex-dividend', 'volume', 'split_ratio'], 1, inplace=True)
# print(df, end=" ")
'''
df = quandl.get_table('WIKI/PRICES', ticker='GOOGL')
df.drop(['ticker', 'open', 'high', 'low', 'close', 'ex-dividend', 'volume', 'split_ratio'], 1, inplace=True)
df.set_index('date', inplace=True)

# Renaming all the columns so that we can use the old version code
df.rename(columns={'adj_open': 'Open', 'adj_high': 'High', 'adj_low': 'Low', 'adj_volume': 'Volume',
                   'adj_close': 'Adj Close'}, inplace=True)

# Percentage change
df['Pct'] = df['Adj Close'].pct_change()
print(df)
'''
