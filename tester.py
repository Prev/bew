import pandas as pd
from sklearn.externals import joblib

clf = joblib.load('cache/classify.model')
cate_dict = joblib.load('cache/cate_dict.dat')
vectorizer = joblib.load('cache/vectorizer.dat')


cate_id_name_dict = dict( map( lambda k,v : (v,k), cate_dict.keys(), cate_dict.values() ) )


#str = '아동한복 여자아동 금나래 (파랑)'


def predict(str) :
	pred = clf.predict(vectorizer.transform([str]))[0]
	return cate_id_name_dict[pred]

def get_cate(each) :
	return ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])



train_df = pd.read_pickle("data/soma_goods_train.df")


for each in train_df.iterrows() :
	name = each[1]['name']
	cat = get_cate(each)

	#print(cat + '\t\t' + predict(name))


	if cat != predict(name):
		#print(cat + '\t\t' + predict(name))
		print('pid:\t\t' + str(each[0]))
		print('product:\t' + name)
		print('real:\t\t' + cat)
		print('est:\t\t' + predict(name))
		print(' ')
		


