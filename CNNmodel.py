# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

from image_dataset import Image_Dataset
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os

class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Try the hyperparameters
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 5, padding = 2), # in, out, kernel size, padding;
            # in has 3 because we have RGB - 3 features...?
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 5),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(16, 16, 5),
            # torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1600, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim = 1)
        )
    
    def forward(self, features):
        return self.layers(features)

def train(model, train_loader, validation_loader, test_loader, optimizer = torch.optim.SGD, lr = 0.1, epochs = 10):

    writer = SummaryWriter()
    optimizer = optimizer(model.parameters(), lr = lr)
    batch_idx = 0

    for epoch in range(epochs):
        for (features, labels) in train_loader: # Load the batch
            # Train the model with train_dataset
            prediction = model(features) # gives predicted labels
            loss = F.cross_entropy(prediction, labels)
            loss.backward() # Essentially grad += d(loss)/dx
            optimizer.step() # Update the model weights & biases with the optimizer
            print(f"Epoch{epoch}, batch{batch_idx}: Loss = {loss.item()}")
            optimizer.zero_grad() # otherwise loss.backward() accumulates gradients from all previous batches
            
            writer.add_scalar("Loss", loss.item(), batch_idx)
            batch_idx += 1


            

        
    return model

def evaluate(model, dataloader):
    '''
    Evaluate the prediction of a model with a validation set.
    Returns average loss and the accuracy of predictions.

    Parameters
    ----------
    model: torch.nn.Module
        The model to be evaluated.
    dataloader: torch.utils.data.dataloader.DataLoader
        The dataloader containing the data of the validation set.
    '''

    losses = []
    correct = 0
    example_count = 0
    for (features, labels) in dataloader:
        prediction = model(features)
        loss = F.cross_entropy(prediction, labels)
        losses.append(loss.detach()) # detach tensor from graph
        predicted_category = torch.argmax(prediction, dim = 1)
        correct += int(torch.sum(predicted_category == labels))
        example_count += len(labels)
    return np.mean(losses), correct/example_count

def split_train_test(dataset, fractions: list):
    '''
    Split a dataset into train, validation and test sets.
    Returns a list of datasets for training, validation and testing respectively.

    Parameters
    ----------
    model: torch.nn.Module
        The model to be evaluated.
    fractions: list of float
        The fraction of elements in train and validation sets respectively.
        The rest are allocated to the test set.
    '''
    # if len(fractions) != 2:
    #     raise Exception(f'2 elements in "fractions" expected, but got {len(fractions)}')
    # elif sum(fractions) > 1:
    #     raise ValueError('Fractions of train and validation sets sums to more than 1')
    train_set_len = int(fractions[0]*len(dataset))
    validation_set_len = int(fractions[1]*len(dataset))
    test_set_len = len(dataset) - train_set_len - validation_set_len
    train_set, validation_set, test_set = \
    torch.utils.data.random_split(dataset,[train_set_len, validation_set_len, test_set_len])
    return [train_set, validation_set, test_set]


if __name__ == '__main__':
    dataset = Image_Dataset(use_cuda = True)
    model = CNN()
    batch_size = 16
    epoch = 3

    # Load datasets and train the model
    train_set, validation_set, test_set = split_train_test(dataset, fractions = [0.7, 0.15])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    model = train(model, train_loader, validation_loader, test_loader, lr = 0.1, epochs = epoch)

    # Save the model
    dir = 'model_evaluation'
    model_name = 'trained_CNNmodel.pt'
    if os.path.exists(dir) == False:
        os.mkdir(dir)
    path = os.path.join(dir, model_name)
    torch.save(model, path) # Save only the model's parameters as a dictionary.
