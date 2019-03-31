#
#
#
import os

import numpy as np
import matplotlib.pyplot as plt

from iex import Stock
from pandas import DataFrame


tgt_stock_label = "AAPL"
raw_data = DataFrame()
date = "20190326"

while len(raw_data) < 50000 and int(date) > 20190300:
	raw_data = raw_data.append(Stock(tgt_stock_label).chart_table(date), ignore_index=True)
	date = str(int(date) - 1)
	print(date)

print(raw_data)


## 0. average
## 1. changeOverTime  2.  close,         3. date,                  4. high,           5. label,
## 6. low             7. marketAverage   8. marketChangeOverTime   9. marketClose     10. marketHigh
## 11. marketLow      12. marketNotional 13. marketNumberOfTrades  14. marketOpen     15. marketVolume
## 16. minute         17. notional,      18. numberOfTrades        19. open           20. volume
##
data = []
average = []

for index, row in raw_data.iterrows():
	# print(index, row)

	temp = []
	for i in range(row.shape[0]):
		if i == 0:
			average.append(row[0])
		elif i == 3:  # date
			pass
		elif i == 5:  # label
			pass
		elif i == 16:  # minite
			pass
		else:
			temp.append(row[i])

	data.append(temp)

print('Finished! Nice work!')























