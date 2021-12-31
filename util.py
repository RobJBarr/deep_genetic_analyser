import pickle


def read_file(file_name):
    with open(file_name, 'rb') as target:
        trained_model = pickle.load(target)
    print("Loaded model from: " + file_name)
    return trained_model
