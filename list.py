import pandas as pd
from bewlib.text import Filter

train_df = pd.read_pickle("data/soma_goods_train.df")


cat_printed = {}
ft = Filter()

data = {}

i = 0
for each in train_df.iterrows() :
	i += 1
	if i % 100 != 0 : continue

	name = each[1]['name']
	print(name)
	print( ft.filter(name) )
	print(' ')

# for each in train_df.iterrows() :
# 	name = each[1]['name']
# 	cat = ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])


# 	if cat not in cat_printed :
# 		cat_printed[cat] = 1
# 	else :
# 		cat_printed[cat] += 1
# 	if cat_printed[cat] > 3 :
# 		continue



# 	name = ft.filter(name)

# 	if cat not in data :
# 		data[cat] = []
	
# 	data[cat].append( name )
	

# data = sorted(data.items())

# for cat, products in data :
# 	print('---------- ' + cat + ' ----------')

# 	for product in products :
# 		print(product)

# 	print(' ')

