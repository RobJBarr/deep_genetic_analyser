import os
import time

from flask import Flask, render_template, request, Response, send_file
from learning import train_model
from mutation_map import get_mutation_map, generate_mutation_map_graph
from util import read_file

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join('./static/for_server', f.filename))


@app.route('/process_train/<file>')
def process_train(file):
    response = Response(train_model(file), mimetype='application/octet-stream')
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/upload_model')
def upload_model():
    return render_template('upload_model.html', download="False")


@app.route('/generate_map/<file>/<sequence>', methods=['POST','GET'])
def generate_map(file, sequence):
    print("here")
    time.sleep(3)

    model = read_file(file)
    mutation_map = get_mutation_map(sequence, model)
    generate_mutation_map_graph(sequence, mutation_map)
    time.sleep(3)
    response = Response(mutation_map, mimetype='image/png')
    response.headers.add('Access-Control-Allow-Origin', '*')
    print("here2")
    return response


if __name__ == '__main__':
    app.run(port=3000)
