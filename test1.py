import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV


train_df = pd.read_pickle("data/soma_goods_train.df")


vectorizer = CountVectorizer()


def get_cate(each) :
	return ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])


d_list = []
cate_list = []

for each in train_df.iterrows() :
	# if i > 1000: break
	# i += 1

	cate = get_cate(each)
	# print(cate)

	d_list.append(each[1]['name'])
	cate_list.append(cate)


cate_dict = dict(zip(list(set(cate_list)),range(len(set(cate_list)))))


y_list = []
for each in train_df.iterrows() :
	cate = get_cate(each)
	y_list.append(cate_dict[cate])


x_list = vectorizer.fit_transform(d_list)

print('x,y list extract completed')


# i = 0
# for y in y_list :
# 	print(str(cate_list[i]) + ' / ' + str(y))
# 	i += 1


svc_param = {'C':np.logspace(-2,0,20)}
gs_svc = GridSearchCV(LinearSVC(), svc_param, cv=5, n_jobs=4)
gs_svc.fit(x_list, y_list)

