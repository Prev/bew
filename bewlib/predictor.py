import pandas as pd
import os 
from sklearn.externals import joblib

dir_path = os.path.dirname(os.path.realpath(__file__))

clf = joblib.load(dir_path + '/../cache/classify.model')
cate_dict = joblib.load(dir_path + '/../cache/cate_dict.dat')
vectorizer = joblib.load(dir_path + '/../cache/vectorizer.dat')


cate_id_name_dict = dict( map( lambda k,v : (v,k), cate_dict.keys(), cate_dict.values() ) )


def predict(str) :
	pred = clf.predict(vectorizer.transform([str]))[0]
	return cate_id_name_dict[pred]

