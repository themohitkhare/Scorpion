import quandl

code = "500010"
quandl.ApiConfig.api_key = "tHhnk2LX1KrKyxKKyhaz"
data = quandl.get("BSE/BOM500010")
data.drop(['WAP', 'No. of Trades', 'Total Turnover', 'Deliverable Quantity', '% Deli. Qty to Traded Qty', 'Spread H-L',
           'Spread C-O'], 1, inplace=True)
print(data)
# df = quandl.get('BSE/BOM' + stock_name)
# df = quandl.get_table('BSE.OIL.1')
# df.drop(['ticker', 'open', 'high', 'low', 'close', 'ex-dividend', 'volume', 'split_ratio'], 1, inplace=True)
# print(df, end=" ")
