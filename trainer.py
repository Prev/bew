import pandas as pd
import numpy as np
from bewlib.text import Filter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

train_df = pd.read_pickle("data/soma_goods_train.df")


def get_cate(each) :
	return ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])


# 이름 필터
ft = Filter()

# Count Vector 관리
# vectorizer = CountVectorizer(max_df=0.3)
vectorizer = CountVectorizer(max_df=0.3)


# 제품 이름 리스트
d_list = []

# 카테고리 리스트
cate_list = []


# d_list 및 cate_list 파싱
for each in train_df.iterrows() :
	name = each[1]['name']
	cate = get_cate(each)
	
	#name = ft.filter(name) + ' '.join( tw.nouns(name) )
	name = ft.key_name(name)

	d_list.append(name)
	cate_list.append(cate)


# 해당 카테고리가 몇번째 고유값을 갖고 있는지 저장
# ex) cate_dict['디지털/가전;PC부품;CPU'] = 1
print('creating cate_dict...')
cate_dict = dict(zip(list(set(cate_list)),range(len(set(cate_list)))))



# 실제 카테고리 저장값
# y_list[제품번호] = 카테고리번호
y_list = []

print('creating y_list...')

for each in train_df.iterrows() :
	cate = get_cate(each)
	y_list.append(cate_dict[cate])



# 제품 이름을 띄어쓰기를 기준으로 몇번째 단어가 몇개씩 있는지 fit하고 transform함
#
#  fit_transform 을 하게 되면, d_list 에 들어 있는 모든 단어들에 대해서, 단어-id 사전을 만드는 일을 먼저하고 (fit)
#  실제로 d_list 에 들어 있는 각 데이터들에 대해서 (단어id,빈도수) 형태의 데이터로 변환을 해준다. (transform)
print('creating x_list...')
x_list = vectorizer.fit_transform( d_list )


print('filtering x_list...')
x_list[x_list > 1] = 1



# SVM(Support Vector Machine) 을 사용한 분류 학습
print('creating gs_svc...')
svc_param = { 'C' : np.logspace(-2,0,20) }
gs_svc = GridSearchCV( LinearSVC(), svc_param, cv = 5, n_jobs = 4 )
gs_svc.fit( x_list, y_list )


# 가장 성능이 좋은 param 및 정확도 출력
print('gs_svc result: ')
print( gs_svc.best_params_, gs_svc.best_score_ )


# C param을 기준으로 SVM 모델 재생성
print('creating clf...')
clf = LinearSVC( C = gs_svc.best_params_['C'] )
clf.fit( x_list, y_list )


# 모델을 파일로 저장
joblib.dump(clf,'cache/classify.model',compress=3)
joblib.dump(cate_dict,'cache/cate_dict.dat',compress=3)
joblib.dump(vectorizer,'cache/vectorizer.dat',compress=3)

print('completed')




