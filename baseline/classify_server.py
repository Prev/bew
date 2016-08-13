
# coding: utf-8

# 

# In[1]:

from sklearn.externals import joblib


# In[2]:


clf = joblib.load('classify.model')
cate_dict = joblib.load('cate_dict.dat')
vectorizer = joblib.load('vectorizer.dat')


# In[3]:

joblib.dump(clf,'n_classify.model')


# In[4]:

joblib.dump(cate_dict,'n_cate_dict.dat')
joblib.dump(vectorizer,'n_vectorizer.dat')


# In[5]:

cate_id_name_dict = dict(map(lambda (k,v):(v,k),cate_dict.items()))


# In[6]:

pred = clf.predict(vectorizer.transform(['[신한카드5%할인][서우한복] 아동한복 여자아동 금나래 (분홍)']))[0]
print cate_id_name_dict[pred]


# In[ ]:

from bottle import route, run, template,request,get, post


import  time
from threading import  Condition
_CONDITION = Condition()
@route('/classify')
def classify():
    print "classify called"
    img = request.GET.get('img','')
    name = request.GET.get('name', '')
    pred = clf.predict(vectorizer.transform([name]))[0]
    return {'cate':cate_id_name_dict[pred]}


run(host='0.0.0.0', port=8887)


#  * 추후 여기 docker 에서 뭔가 python package 설치할게 있으면 
#  * /opt/conda/bin/pip2 install bottle 이런식으로 설치 가능

# In[ ]:




# In[ ]:



