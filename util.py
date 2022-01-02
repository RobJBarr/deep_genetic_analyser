import pickle
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_file(file_path):
    with open(file_path, 'rb') as target:
        trained_model = pickle.load(target)
    print("Loaded model from: " + file_path)
    return trained_model.to(device)
