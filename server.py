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
	name = ft.key_name(name)

	output = {
		'cate' : pdt.predict(name)
	}

	return json.dumps(output), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
	app.run(debug = True, port=18887, host='0.0.0.0')
	#app.run(debug = True, port=18887, host='localhost')