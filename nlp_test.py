import pandas as pd
from konlpy.tag import Twitter
from bewlib.text import Filter


train_df = pd.read_pickle("data/soma_goods_train.df")

tw = Twitter()
ft = Filter()

i = 0

for each in train_df.iterrows() :
	i += 1
	if i % 100 != 0 : continue
	
	name = each[1]['name']
	print(name)
	#print(tw.nouns(name))
	#print(tw.phrases(name))
	n = tw.nouns( ft.filter(name) )
	print(' '.join(n))

	print('--------------------------------------------------------------------------------------------')