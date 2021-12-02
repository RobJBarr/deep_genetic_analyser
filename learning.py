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


def train_model(file_path, observer):
    observer.update(0)

    device = torch.device('cpu')
    num_epochs = 5
    learning_steps_list = [4000, 8000, 12000, 16000, 20000]
    best_AUC = 0

    # Load the data from the file_path
    first_train, first_valid, second_train, second_valid, third_train, third_valid = parse_file(file_path)

    # Load the data as tensors
    first_train = load_as_tensor(first_train, 64)
    second_train = load_as_tensor(second_train, 64)
    third_train = load_as_tensor(third_train, 64)
    first_valid = load_as_tensor(first_valid, 64)
    second_valid = load_as_tensor(second_valid, 64)
    third_valid = load_as_tensor(third_valid, 64)
    training_sets = [first_train, second_train, third_train]
    validation_sets = [first_valid, second_valid, third_valid]

    # Iterate over the training_sets and validation_sets num_epochs times
    for n in range(num_epochs):
        # Choose randomly when there is a choice for a parameter
        max_pool = random.choice([True, False])
        hidden_layer = random.choice([True, False])
        dropout_value = random.choice([0.5, 0.75, 1.0])
        # Get values for parameter using log and square root sampler
        learning_rate = log_sampler(0.0005, 0.05)
        learning_momentum = sqrt_sampler(0.95, 0.99)
        initial_weight = log_sampler(10 ** -7, 10 ** -3)
        neural_weight = log_sampler(10 ** -5, 10 ** -2)
        weight_decay1 = log_sampler(10 ** -10, 10 ** -3)
        weight_decay2 = log_sampler(10 ** -10, 10 ** -3)
        weight_decay3 = log_sampler(10 ** -10, 10 ** -3)

        # Store the auc performance for each fold and each 4000 learning step interval
        # E.g model_auc[0][2] is mean auc for model trained on 1st train at 12000 learning steps on 1st valid
        model_auc = [[], [], []]

        # Iterate over 3 different folds
        for fold in range(3):
            # Initialise the model
            model = ConvNet(num_motifs=16, motif_len=24, max_pool=max_pool,
                            hidden_layer=hidden_layer, training_mode=True,
                            dropout_value=dropout_value, learning_rate=learning_rate,
                            learning_momentum=learning_momentum, initial_weight=initial_weight,
                            neural_weight=neural_weight, weight_decay1=weight_decay1, weight_decay2=weight_decay2,
                            weight_decay3=weight_decay3).to(device)

            # Select the training and validation set to use
            train = training_sets[fold]
            valid = validation_sets[fold]

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

            # For the train set selected for this fold, train the model for 20000 update steps
            learning_steps = 0
            while learning_steps <= 20000:
                # At each update step, go through the data in the train set selected
                for _, (t_data, t_target) in enumerate(train):
                    t_data = t_data.to(device)
                    t_target = t_target.to(device)

                    # Feed-forward on the model using the data
                    output = model(t_data)

                    # Calculate the loss function output, which depends on whether the model has a hidden layer
                    if model.hidden_layer:
                        loss = F.binary_cross_entropy(input=torch.sigmoid(output), target=t_target)
                        + model.weight_decay1 * model.convolution_weights.norm() \
                        + model.weight_decay2 * model.hidden_weights.norm() \
                        + model.weight_decay3 * model.neural_weights.norm()

                    else:
                        loss = F.binary_cross_entropy(input=torch.sigmoid(output), target=t_target) \
                               + model.weight_decay1 * model.convolution_weights.norm() \
                               + model.weight_decay3 * model.neural_weights.norm()

                    # Perform back propagation
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    learning_steps += 1

                    # Every 4000 steps, we evaluate the current performance of the trained model on the valid set.
                    # So this will be done at 5 times at [4000, 8000, 12000, 16000 and 20000] steps
                    if learning_steps % 4000 == 0:
                        with torch.no_grad():
                            # Set the training mode of the model to false
                            model.training_mode = False
                            auc = []
                            # Iterate through the valid set
                            for _, (v_data, v_target) in enumerate(valid):
                                v_data = v_data.to(device)
                                v_target = v_target.to(device)

                                # Feed-forward on the model
                                output = model(v_data)
                                # Compute the sigmoid function on the output
                                pred_sig = torch.sigmoid(output)
                                pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                                labels = v_target.cpu().numpy().reshape(output.shape[0])
                                # Find and add the auc performance metric for the data
                                auc.append(metrics.roc_auc_score(labels, pred))

                            # Get the mean auc performance metric for the valid set on this fold and learning step
                            model_auc[fold].append(np.mean(auc))
                            print('AUC performance when training fold number ', fold + 1, 'learning steps = ',
                                  learning_steps_list[len(model_auc[fold]) - 1], 'is ', np.mean(auc))

        for num in range(5):
            # Get the mean auc between the three folds for each learning interval
            AUC = (model_auc[0][num] + model_auc[1][num] + model_auc[2][num]) / 3

            # Update the best_AUC if the mean auc is higher
            if AUC > best_AUC:
                best_AUC = AUC
                best_learning_steps = learning_steps_list[num]
                best_learning_rate = model.learning_rate
                best_learning_momentum = model.learning_momentum
                is_hidden_layer = model.hidden_layer
                is_max = model.max_pool
                best_initial_weight = model.initial_weight
                best_dropout_value = model.dropout_value
                best_neural_weight = model.neural_weight
                best_weight_decay1 = model.weight_decay1
                best_weight_decay2 = model.weight_decay2
                best_weight_decay3 = model.weight_decay3

        #yield "data: {}\n\n".format((n + 1) * 18)
        observer.update((n + 1) * 18)  # At this point we should be 20%, 40%, 60%, 80% or 100% done

    # Return model with the best parameters which will be used for predicting other sequences
    optimal_model = ConvNet(num_motifs=16, motif_len=24, max_pool=is_max,
                            hidden_layer=is_hidden_layer, training_mode=False,
                            dropout_value=best_dropout_value, learning_rate=best_learning_rate,
                            learning_momentum=best_learning_momentum, initial_weight=best_initial_weight,
                            neural_weight=best_neural_weight, weight_decay1=best_weight_decay1,
                            weight_decay2=best_weight_decay2,
                            weight_decay3=best_weight_decay3).to('cpu')
    #yield "data: {}\n\n".format(100)
    observer.update(100)


class TrainingObserver:
    def __init__(self, task_id):
        self.current_percentage = 0
        self.task_id = task_id

    def update(self, percentage):
        yield "data: {}\n\n".format(percentage)
        self.current_percentage = percentage
