import torch
from torch import nn
import torchmetrics

def train_step(model, dataloader: torch.utils.data.DataLoader,
               optimizer,
               loss_fn,
               acc_fn: torchmetrics.Accuracy,
               device = 'cpu'):
    """
    Trains the model for 1 epoch.
    Returns a dictionary containing the loss and accuracy values.
    """
    model.train() # Put model in training mode
    average_accuracy, average_loss = 0, 0
    for X_train, y_train in dataloader:
        # 0. Send data to target device
        X_train, y_train = X_train.to(device), y_train.to(device)
        # 1. Forward pass
        y_preds = model(X_train)
        # 2. Calculate the loss and accuracy
        loss = loss_fn(y_preds, y_train)
        average_loss += loss
        acc_fn = acc_fn.to(device)  # Send metric to target device
        acc = acc_fn(y_preds, y_train)
        average_accuracy += acc
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Backpropagation
        loss.backward()
        # 5. Gradient descent
        optimizer.step()

    average_accuracy /= len(dataloader)
    average_loss /= len(dataloader)
    return {'Loss': float(loss), 'Accuracy': float(average_accuracy) * 100}

def test_step(model,
              dataloader: torch.utils.data.DataLoader,
              loss_fn,
              acc_fn: torchmetrics.Accuracy,
              device = 'cpu'):
    """
    Tests the model using a test dataloader.
    Returns a dictionary containing the loss and accuracy values.
    """
    model.eval() # Put model in evaluation mode
    average_accuracy, average_loss = 0, 0
    for X_test, y_test in dataloader:
        # 0. Send data to target device
        X_test, y_test = X_test.to(device), y_test.to(device)
        # 1. Forward pass
        y_preds = model(X_test)
        # 2. Calculate the loss and accuracy
        loss = loss_fn(y_preds, y_test)
        average_loss += loss
        acc_fn = acc_fn.to(device)  # Send metric to target device
        acc = acc_fn(y_preds, y_test)
        average_accuracy += acc
    average_accuracy /= len(dataloader)
    average_loss /= len(dataloader)
    return {'Loss': float(loss), 'Accuracy': float(average_accuracy) * 100}

def calculateConvLayersOutputPixels(input_shape, *layers_sequence):
    """
    Returns the result of multiplying the output's height and width to be used as input for linear layers.
    """
    for layers in layers_sequence:
        for layer in range(len(layers)):
            if isinstance(layers[layer], nn.Conv2d):
                input_shape = [
                    (input_shape[-2] + 2* layers[layer].padding[0] - layers[layer].dilation[0] * (layers[layer].kernel_size[0] - 1) - 1) / layers[layer].stride[0] + 1,
                    (input_shape[-1] + 2* layers[layer].padding[1] - layers[layer].dilation[1] * (layers[layer].kernel_size[1] - 1) - 1) / layers[layer].stride[1] + 1
                ]
            elif isinstance(layers[layer], nn.MaxPool2d):
                input_shape = [
                    (input_shape[-2] + 2* layers[layer].padding - layers[layer].dilation * (layers[layer].kernel_size - 1) - 1) / layers[layer].stride + 1,
                    (input_shape[-1] + 2* layers[layer].padding - layers[layer].dilation * (layers[layer].kernel_size - 1) - 1) / layers[layer].stride + 1
                ]

    return int(input_shape[0] * input_shape[1])