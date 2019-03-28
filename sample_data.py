from iex import Stock
from pandas import DataFrame


tgt_stock_label = "AAPL"
raw_data = DataFrame()
date = "20190326"
while len(raw_data) < 50000 and int(date) > 20190300:
	raw_data = raw_data.append(Stock(tgt_stock_label).chart_table(date), ignore_index=True)
	date = str(int(date) - 1)
print(raw_data)
raw_data = raw_data[raw_data["average"] != -1]
x = raw_data.loc[:, "changeOverTime": "volumn"].as_matrix()
y = raw_data["average"].as_matrix()
