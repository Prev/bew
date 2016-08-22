# -*- coding: utf-8 -*-
#
# BEW (back-end-work) project
# author: YoungSoo Lee(prevdev@gmail.com)

from flask import Flask
from flask import request
from bewlib.text import Filter
import bewlib.predictor as pdt
import json
import jpype


app = Flask(__name__)
ft = Filter()

@app.route('/classify', methods=['GET'])
def classify():
	jpype.attachThreadToJVM()

	name = request.args.get('name')

	# f = open('req.txt', 'a')
	# f.write(name + '\n')
	# f.close()

	name = ft.key_name(name)
	name = ft.repetition_removal(name)

	output = {
		'cate' : pdt.predict(name)
	}

	return json.dumps(output), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
	app.run(debug = True, port=18887, host='0.0.0.0')