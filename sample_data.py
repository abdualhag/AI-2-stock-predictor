from iex import Stock


tgt_stock_label = "AAPL"
raw_data = Stock(tgt_stock_label).chart_table("1d")

x = raw_data.loc[:, "changeOverTime":"volume"]
y = raw_data["average"]

print(x)
print(y)
