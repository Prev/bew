from flask import Flask
from flask import request
import bewlib.predictor as pdt
import json

app = Flask(__name__)


@app.route('/classify', methods=['GET'])
def classify():
	product = request.args.get('name')

	output = {
		'cate' : pdt.predict(product)
	}

	return json.dumps(output), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
	app.run(debug = True, port=18887, host='localhost')