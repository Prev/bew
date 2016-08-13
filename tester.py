import pandas as pd
import bewlib.predictor as pdt


def get_cate(each) :
	return ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])


train_df = pd.read_pickle("data/soma_goods_train.df")

success = 0
count = 0

for each in train_df.iterrows() :
	name = each[1]['name']
	cat = get_cate(each)
	cat_est = pdt.predict(name)

	count += 1

	if cat == cat_est :
		success += 1

	# if cat != cat_est:
		# print('pid:\t\t' + str(each[0]))
		# print('product:\t' + name)
		# print('real:\t\t' + cat)
		# print('est:\t\t' + cat_est)
		# print(' ')
		

print( str( success / count * 100 ) + '% : ' + str(success) + '/' + str(count) )


