import pandas as pd
from bewlib.text import Filter


train_df = pd.read_pickle("data/soma_goods_train.df")

ft = Filter()

i = 0

for each in train_df.iterrows() :
	i += 1
	if i % 100 != 0 : continue
	
	name = each[1]['name']
	print(name)
	#print(tw.nouns(name))
	#print(tw.phrases(name))
	print(ft.key_name(name))

	print('--------------------------------------------------------------------------------------------')