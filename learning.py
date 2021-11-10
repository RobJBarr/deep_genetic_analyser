import math
import numpy as np
import random
import torch
import torch.nn.functional as F

from file_parser import load_as_tensor, parse_file
from prediction import ConvNet
from sklearn import metrics


def log_sampler(a, b):
    x = np.random.uniform(low=0, high=1)
    y = 10 ** ((math.log10(b) - math.log10(a)) * x + math.log10(a))
    return y


def sqrt_sampler(a, b):
    x = np.random.uniform(low=0, high=1)
    y = (b - a) * math.sqrt(x) + a
    return y


device = torch.device('cpu')


def train_model(file_path):
    # Load the data from the file_path as 20 segments of tensors for training
    segments = parse_file(file_path)
    for segment in segments:
        first_train, first_valid, second_train, second_valid, third_train, third_valid = segment
        first_train = load_as_tensor(first_train, 64)
        second_train = load_as_tensor(second_train, 64)
        third_train = load_as_tensor(third_train, 64)
        first_valid = load_as_tensor(first_valid, 64)
        second_valid = load_as_tensor(second_valid, 64)
        third_valid = load_as_tensor(third_valid, 64)
        training_sets = [first_train, second_train, third_train]
        validation_sets = [first_valid, second_valid, third_valid]

        for n in range(5):
            # Make random choices as calibration parameters for the ConvNet model
            max_pool = random.choice([True, False])
            hidden_layer = random.choice([True, False])
            dropout_value = random.choice([0.5, 0.75, 1.0])
            learning_rate = log_sampler(0.0005, 0.05)
            learning_momentum = sqrt_sampler(0.95, 0.99)
            initial_weight = log_sampler(10 ** -7, 10 ** -3)
            neural_weight = log_sampler(10 ** -5, 10 ** -2)
            weight_decay1 = log_sampler(10 ** -10, 10 ** -3)
            weight_decay2 = log_sampler(10 ** -10, 10 ** -3)
            weight_decay3 = log_sampler(10 ** -10, 10 ** -3)

            model_auc = [[], [], []]

            for i in range(3):
                # Initialise the model
                model = ConvNet(num_motifs=16, motif_len=24, max_pool=max_pool,
                                hidden_layer=hidden_layer, training_mode=True,
                                dropout_value=dropout_value, learning_rate=learning_rate,
                                learning_momentum=learning_momentum, initial_weight=initial_weight,
                                neural_weight=neural_weight, weight_decay1=weight_decay1, weight_decay2=weight_decay2,
                                weight_decay3=weight_decay3).to(device)

                # Check if model has 'one hidden layer with 32 rectified-linear units'
                if model.hidden_layer:
                    optimiser = torch.optim.SGD(
                        params=[model.convolution_weights, model.rectification_weights, model.neural_weights,
                                model.weights_neural_bias, model.hidden_weights, model.wHiddenBias],
                        lr=model.learning_rate, momentum=model.learning_momentum, nesterov=True)
                else:
                    optimiser = torch.optim.SGD(
                        params=[model.convolution_weights, model.rectification_weights, model.neural_weights,
                                model.weights_neural_bias],
                        lr=model.learning_rate, momentum=model.learning_momentum, nesterov=True)

                # Select the training and validation set to use
                train = training_sets[i]
                valid = validation_sets[i]
                learning_steps = 0
                while learning_steps <= 20000:
                    auc = []
                    for _, (t_data, t_target) in enumerate(train):
                        t_data = t_data.to(device)
                        t_target = t_target.to(device)

                        # Feed-forward on the model using data
                        output = model(t_data)

                        # Loss function output depends on whether the model has a hidden layer or not
                        if model.hidden_layer:
                            loss = F.binary_cross_entropy(input=torch.sigmoid(output), target=t_target)
                            + model.weight_decay1 * model.convolution_weights.norm() \
                            + model.weight_decay2 * model.hidden_weights.norm() \
                            + model.weight_decay3 * model.neural_weights.norm()

                        else:
                            loss = F.binary_cross_entropy(input=torch.sigmoid(output), target=t_target) \
                                + model.weight_decay1 * model.convolution_weights.norm() \
                                + model.weight_decay3 * model.neural_weights.norm()

                        optimiser.zero_grad()
                        loss.backward()
                        optimiser.step()
                        learning_steps += 1

                        if learning_steps % 4000 == 0:
                            with torch.no_grad():
                                model.training_mode = False
                                auc = []
                                for _, (v_data, v_target) in enumerate(valid):
                                    v_data = v_data.to(device)
                                    v_target = v_target.to(device)

                                    # Feed-forward on the model
                                    output = model(v_data)
                                    pred_sig = torch.sigmoid(output)
                                    pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                                    labels = v_target.cpu().numpy().reshape(output.shape[0])
                                    auc.append(metrics.roc_auc_score(labels, pred))

                                model_auc[i].append(np.mean(auc))


file = r'C:\Users\prash\deep_genetic_analyser\ELK1_GM12878_ELK1_(1277-1)_Stanford_AC.seq.gz'
train_model(file)
