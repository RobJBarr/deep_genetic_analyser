from flask import Flask, request, Response

from learning import TrainingObserver, train_model

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        f.save(f.filename)


@app.route('/process_train/<file>')
def process_train(file):
    response = Response(train_model(file, TrainingObserver()), mimetype='text/event-stream')
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(port=3000)
