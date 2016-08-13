import bewlib.text as btext
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

d_list = [
	"인텔 intel 코어4세대 하스웰 i3-4160",
	"[4%즉시할인쿠폰]인텔 제온 E3-1226V3 (하스웰) (정품)",
	"인텔 펜티엄 하스웰 G3220",
	"베네통키즈 멀티프린트경량신발주머니(QCBG23511.PK)",
	"[현대백화점 V관] 파코라반베이비 룰라니트가디건 PP1-43204 핑크"
]

y_list = [0, 0, 0, 1, 1]


vectorizer = CountVectorizer()

x_list = vectorizer.fit_transform( d_list )

print(x_list.toarray())



svc_param = {'C':np.logspace(-2,0,20)}
gs_svc = GridSearchCV(LinearSVC(), svc_param)

gs_svc.fit(x_list, y_list)

print(gs_svc.best_params_, gs_svc.best_score_)


clf = LinearSVC(C=gs_svc.best_params_['C'])
clf.fit(x_list,y_list)




str = '[현대백화점][신한카드5%할인][서우한복] 아동한복 여자아동 금나래 (분홍)'

print( vectorizer.fit_transform([str]).toarray() )


# pred = clf.predict(
# 	vectorizer.transform([str])
# )[0]


# print(pred)