import os
import time
from flask import Flask, render_template, request, Response, send_file
from flask_cors import CORS, cross_origin
from learning import train_model
from mutation_map import get_mutation_map, generate_mutation_map_graph
from util import read_file

app = Flask(__name__)
cors = CORS(app)


@app.route('/process_sequence', methods = ['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def process_sequence():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join('./', f.filename))
        return "success"

@app.route('/process_pickle', methods = ['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def process_pickle():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join('./', f.filename))
        return "success"

@app.route('/process_train/<file>', methods = ['GET'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def process_train(file):
    time.sleep(3)
    response = Response(train_model(file), mimetype='text/event-stream')
    return response


@app.route('/upload_model')
def upload_model():
    return render_template('upload_model.html', download="False")


@app.route('/generate_map/<file>/<sequence>', methods=['POST','GET'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def generate_map(file, sequence):
    time.sleep(3)
    model = read_file(file)
    os.remove(file)
    mutation_map = get_mutation_map(sequence, model)
    base64_data = generate_mutation_map_graph(sequence, mutation_map)
    response = Response(base64_data)
    return response

if __name__ == '__main__':
    app.run(port=3000)
