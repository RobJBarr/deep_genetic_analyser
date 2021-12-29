import numpy as np
from prediction import Predictor


def save_file(model):
    hidden_layer = model.hidden_layer
    max_pool = model.max_pool
    dropout_value = model.dropout_value
    convolution_weights = model.convolution_weights.detach().numpy()
    rectification_weights = model.rectification_weights.detach().numpy()
    neural_weights = model.neural_weights.detach().numpy()
    neural_bias = model.neural_bias.detach().numpy()
    hidden_weights = None if not hidden_layer else model.hidden_weights.detach().numpy()
    hidden_bias = None if not hidden_layer else model.hidden_bias.detach().numpy()

    # Saving param values into txt and npz files
    with open("params.txt", "w") as f:
        f.write(str(hidden_layer) + "\n")
        f.write(str(max_pool) + "\n")
        f.write(str(dropout_value))

    if hidden_layer:
        np.savez('weights.npz', convolution_weights=convolution_weights, rectification_weights=rectification_weights,
                 neural_weights=neural_weights, neural_bias=neural_bias, hidden_weights=hidden_weights,
                 hidden_bias=hidden_bias)

    else:
        np.savez('weights.npz', convolution_weights=convolution_weights, rectification_weights=rectification_weights,
                 neural_weights=neural_weights, neural_bias=neural_bias)


def read_file():
    hidden_weights = None
    hidden_bias = None

    with open("params.txt", "r") as f:
        lines = f.read().splitlines()
        hidden_layer = True if lines[0] == 'True' else False
        max_pool = True if lines[1] == 'True' else False
        dropout_value = float(lines[2])

    data = np.load('mat.npz')
    convolution_weights = data['convolution_weights']
    rectification_weights = data['rectification_weights']
    neural_weights = data['neural_weights']
    neural_bias = data['neural_bias']

    if hidden_layer:
        hidden_weights = data['hidden_weights']
        hidden_bias = data['hidden_bias']

    model = Predictor(convolution_weights=convolution_weights, rectification_weights=rectification_weights,
                      hidden_layer=hidden_layer, max_pool=max_pool, neural_weights=neural_weights,
                      neural_bias=neural_bias, hidden_weights=hidden_weights, hidden_bias=hidden_bias,
                      dropout_value=dropout_value)

    return model
