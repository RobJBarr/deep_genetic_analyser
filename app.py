from flask import Flask, render_template, request, Response

from learning import TrainingObserver, train_model

app = Flask(__name__)


@app.route('/')
def file_upload():
    return render_template('file_upload.html')


@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        return render_template('progress.html', file=f.filename, task_id=0)


@app.route('/process_train/<file>/<task_id>')
def process_train(file, task_id):
    return Response(train_model(file, TrainingObserver(task_id)), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run()
