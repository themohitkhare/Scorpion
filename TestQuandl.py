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
data['Volume'] = x.fit_transform(data.Volume.values.reshape(-1, 1))


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    print("Amount of features = {}".format(amount_of_features))
    data = stock.as_matrix()
    sequence_length = seq_len + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length])  # index : index + 22days

    result = np.array(result)
    row = round(0.8 * result.shape[0])  # 80% split
    print("Amount of training data = {}".format(0.9 * result.shape[0]))
    print("Amount of testing data = {}".format(0.1 * result.shape[0]))

    train = result[:int(row), :]  # 90% date
    X_train = train[:, :-1]  # all data until day m
    y_train = train[:, -1][:, -1]  # day m + 1 adjusted close price

    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]


X_train, y_train, X_test, y_test = load_data(data, 22)

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
