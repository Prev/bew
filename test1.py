import pandas as pd

train_df = pd.read_pickle("soma_goods_train.df")


i = 0

categories = []

for each in train_df.iterrows() :
	# if i > 10 :	
	# 	break

	# print(each[1]['name'])
	# i += 1

	cat = each[1]['cate1'] + '\t' + each[1]['cate2'] + '\t\t' + each[1]['cate3']

	if cat not in categories :
		categories.append(cat)

#print( train_df )

categories.sort()

for cat in categories :
	print( cat )