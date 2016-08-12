
# coding: utf-8

# # 7기 마에스트로 백엔드 과제 - 상품 카테고리 자동 분류 서버 

# # 과제 개요

#  * 출제자 : 남상협 멘토 (justin@buzzni.com) / 버즈니 (http://buzzni.com) 대표 
#  * 배경 : 카테고리 분류 엔진은 실제로 많은 서비스에서 사용되는 중요한 기계학습의 한 분야이다. 본 과제는 버즈니 개발 인턴이자 마에스트로 6기 멘티가 아래와 나와 있는 기본 분류 모델을 기반으로 deep learning 기반의 feature 를 더해서 고도화된 분류 엔진을 만들어서 2016 한국 정보과학회 논문으로도 제출 했던 과제다. 기계학습에 대한 학습과, 실용성 두가지 측면에서 모두 도움이 될 것으로 보인다.
# 

# ## 과제 목표

#  * 입력 : 상품명, 상품 이미지
#  * 출력 : 카테고리
#  * 목표 : 가장 높은 정확도로 분류를 하는 분류 엔진을 개발

# ## 제약 조건
#  * 분류 엔진은 rest api 형태로 만들어야함 (샘플로 준 python server 사용해도 됨)
#  * rest api 는  http://서버주소:포트/classify?name=상품명&img=상품이미지주소 형식으로 호출이 가능해야함
#  * 위 rest api 를 이용해서 별도로 가지고 있는 데이터로 자동 성능 평가로 채점을 하게됨 (채점하는 데이터는 제공되지 않음)
#  * 데이터 리턴 형식은 {u'cate': u'\ud328\uc158\uc758\ub958;\uc544\ub3d9\uc758\ub958;\ud55c\ubcf5'} 이런식으로 cate 라는 키 값에 그에 해당되는 대,중,소 카테고리를 ; 로 연결한 형태로 반환하면 됨 

# ## 평가 항목
#  1. 성능 평가 (100%)

# ## 제출 항목
#  1. 채점 서버 호출이 필요함: 채점 서버 호출시 name 부분에 자신의 이름을 넣으면 됨.
#   - 자신이 개발한 서버 호출 형태 : http://서버주소:포트/classify?name=상품명&img=상품이미지주소
#   - 평가 서버 : http://somaeval.hoot.co.kr:8880/eval?url= 뒤에 자신의 서버 주소를 넣으면 됨 너무 자주 호출하면 서버가 죽을 수 있으니, mode=all 로는 꼭 필요할때만 호출하기 바람 , 평가 서버가 죽었을 시에는 justin@buzzni.com 으로 문의 
#   - 예) 샘플 테스트 :  http://somaeval.hoot.co.kr:8880/eval?url=http://somaeval.hoot.co.kr:18887
#   - 예) 전체 테스트 : http://somaeval.hoot.co.kr:8880/eval?url=http://somaeval.hoot.co.kr:18887&mode=all&name=베이스라인

# ## 개발 서버 URL 형태 
#  * name, img 를 파리미터로 호출한다.
#  * 호출 형태 : http://somaeval.hoot.co.kr:18887/classify?name=조끼&img=http://shopping.phinf.naver.net/main_8134935/8134935099.1.jpg 
#  

# ## 성능평가 테스트 서버 
#  * 아래 주소에서 url 에 자신의 분류기 모델 주소를 넣어주면 제공되지 않았던 데이터들을 이용해서 평가를 해준다.
#  * 샘플 테스트(카테고리별 2개만 가지고 테스트) : http://somaeval.hoot.co.kr:8880/eval?url=http://somaeval.hoot.co.kr:18887
#  * 전체 성능 평가 테스트 (mode=all) : http://somaeval.hoot.co.kr:8880/eval?url=http://somaeval.hoot.co.kr:18887&mode=all
#  

# ## 현재 보고 있는 IPython Notebook 사용법
#  * https://www.youtube.com/results?search_query=ipython+notebook+tutorial
#  * shift + enter 를 누르면 실행이 된다.

#  * cate1 는 대분류, cate2 는 중분류, cate3 는 소분류 
#  * 총 10000개의 학습 데이터

# ## 성능 향상 포인트
#  * 오픈된 형태소 분석기(예 - 은전한닢 http://eunjeon.blogspot.kr/ )를 써서, 단어 띄어쓰기를 의미 단위로 띄어서 학습하기 
#  * 상품명에서 분류하는데 도움이 되지 않는 stop word 제거하기 
#  * bigram, unigram, trigram 등 단어 feature 를 더 다양하게 추가하기 
#  * 이미지 데이터를 Deep Learning (CNN) 기반 방법으로 feature 를 추출해서 추가하기 
#   * 제일 기본적으로 https://github.com/irony/caffe-docker-classifier 이런 이미 만들어진 모델을 이용해서 feature 를 추출해서 추가하기도 가능함 
#   * DIGITS + caffe 를 이용해서 본 학습 모델에 맞는 이미지 자동 분류기를 별도로 학습해서 사용하는것도 가능함

# # 아래는 baseline 모델을 만드는 방법 

# In[3]:

import pandas as pd


# ## 데이터를 읽는다.
#  * 아래 id 에 해당되는 이미지 데이터 다운받기 https://www.dropbox.com/s/q0qmx3qlc6gfumj/soma_train.tar.gz
#  

# In[4]:

train_df = pd.read_pickle("soma_goods_train.df")


# In[5]:

train_df.shape


# In[6]:

train_df


# In[7]:

from sklearn.feature_extraction.text import CountVectorizer


#  * CountVectorizer 는 일반 text 를 이에 해당되는 숫자 id 와, 빈도수 형태의 데이터로 변환 해주는 역할을 해준다.
#  * 이 역할을 하기 위해서 모든 단어들에 대해서 id 를 먼저 할당한다.
#  * 그리고 나서, 학습 데이터에서 해당 단어들과, 그것의 빈도수로 데이터를 변환 해준다. (보통 이런 과정을 통해서 우리가 이해하는 형태를 컴퓨터가 이해할 수 있는 형태로 변환을 해준다고 보면 된다)
#  * 예를 들어서 '베네통키즈 키즈 러블리 키즈' 라는 상품명이 있고, 각 단어의 id 가 , 베네통키즈 - 1, 키즈 - 2, 러블리 -3 이라고 한다면 이 상품명은 (1,1), (2,2), (3,1) 형태로 변환을 해준다. (첫번째 단어 id, 두번째 빈도수)
#  

# In[8]:

vectorizer = CountVectorizer()


#  * 대분류, 중분류, 소분류 카테고리 명을 합쳐서 카테고리명을 만든다.  우리는 이 카테고리명을 예측하는 분류기를 만들게 된다.
#  * d_list 에는 학습하는 데이터(상품명) 을 넣고, cate_list 에는 분류를 넣는다.

# In[9]:

d_list = []
cate_list = []
for each in train_df.iterrows():
    cate = ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])
    d_list.append(each[1]['name'])
    cate_list.append(cate)


# In[10]:

print len(set(cate_list))


#  * 각 카테고리명에 대해서 serial 한 숫자 id 를 부여한다.
#  * cate_dict[카테고리명] = serial_id 형태이다. 

# In[11]:

cate_dict = dict(zip(list(set(cate_list)),range(len(set(cate_list)))))


# In[12]:

print cate_dict[u'디지털/가전;네트워크장비;KVM스위치']
print cate_dict[u'패션의류;남성의류;정장']


#  * y_list 에는 단어 형태의 카테고리명에 대응되는 serial_id 값들을 넣어준다.

# In[13]:

y_list = []
for each in train_df.iterrows():
    cate = ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])
    y_list.append(cate_dict[cate])


#  * fit_transform 을 하게 되면, d_list 에 들어 있는 모든 단어들에 대해서, 단어-id 사전을 만드는 일을 먼저하고 (fit)
#  * 실제로 d_list 에 들어 있는 각 데이터들에 대해서 (단어id,빈도수) 형태의 데이터로 변환을 해준다. (transform)

# In[14]:

x_list = vectorizer.fit_transform(d_list)


# In[15]:

# print y_list


#  * 여기서는 분류에서 가장 많이 사용하는 SVM(Support Vector Machine) 을 사용한 분류 학습을 한다. 

# In[16]:

from sklearn.svm import LinearSVC


# In[17]:

from sklearn.grid_search import GridSearchCV


# In[18]:

import numpy as np


# In[19]:

svc_param = {'C':np.logspace(-2,0,20)}


#  * grid search 를 통해서 최적의 c 파라미터를 찾는다.
#  * 5 cross validation 을 한다.

# In[20]:

gs_svc = GridSearchCV(LinearSVC(loss='l2'),svc_param,cv=5,n_jobs=4)


# In[25]:

gs_svc.fit(x_list, y_list)


#  * 현재 기본 baseline 성능은 64% 정도로 나온다. 이 성능을 높이는 것이 본 과제의 목표이다. 
#  * 위 grid search 로는 c 값을 찾고, 이렇게 찾은 c 값으로 다시 train 을 해서 최종 모델을 만든다.

# In[26]:

print gs_svc.best_params_, gs_svc.best_score_


# In[27]:

clf = LinearSVC(C=gs_svc.best_params_['C'])


# In[28]:

clf.fit(x_list,y_list)


# In[29]:

from sklearn.externals import joblib


#  * 만들어진 모델을 나중에도 쓰기 위해서 파일에 저장한다.
#  * 아래 형태로 저장하면, 추후에 손쉽게 load 할 수 있다.
#  * 이때 SVM 모델,  cate_name - cate_id 사전, 단어 - 단어_id,빈도수 변ㅎ

# In[30]:

joblib.dump(clf,'classify.model',compress=3)
joblib.dump(cate_dict,'cate_dict.dat',compress=3)
joblib.dump(vectorizer,'vectorizer.dat',compress=3)


#  * 여기까지 모델을 만든다음에 classify_server 노트북을 열고, 쭉 실행을 시키면 서버가 뜬다.
#  * 서버가 뜨면 아래처럼 실행을 시킬 수 가 있다.

# In[31]:

import requests


# In[32]:

name='[신한카드5%할인][예화-좋은아이들] 아동한복 여아 1076 빛이나노랑'
img=''


# In[33]:

u='http://localhost:8887/classify?name=%s&img=%s'


# In[34]:

r = requests.get(u%(name,img)).json()
# classify_server 이 노트북을 먼저 실행하고 나서 해야 동작한다.


# In[35]:

print r


# In[ ]:



