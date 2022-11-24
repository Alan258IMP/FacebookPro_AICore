# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

from CNNmodel import CNN
from image_dataset import Image_Dataset
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
import datetime

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

def split_train_test(dataset, fractions):
    '''
    Split a dataset into train, validation and test sets.
    Returns a list of datasets for training, validation and testing respectively.

    Parameters
    ----------
    model: torch.nn.Module
        The model to be evaluated.
    fractions: list/tuple of float, length 2
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
    # Convert from Subset object to Dataset object
    
    return [train_set, validation_set, test_set]

def train(model, train_loader, validation_loader, test_loader, model_name = None, optimizer = torch.optim.SGD, lr = 0.1, epochs = 10):
    '''
    Train and evaluate a model for a certain number of epoch, and save the model 
    It trains the model, evaluates it on the validation dataset after each epoch, and saves the model
    for each epoch
    
    Parameters
    ----------
    model: torch.nn.Module
        The model to be trained.
    train_loader, validation_loader, test_loader: torch.utils.data.dataloader.DataLoader
        The dataloader containing the datasets for training, validation and testing respectively.
    model_name: str
        The name of the model. This will be used to create a directory for the model.
    optimizer: type in torch.optim.Optimizer
        The optimizer to use for training.
    lr: float
        The learning rate. Default 0.1
    epochs: int
        Number of times the model will be trained over. Default 10
    
    Returns
    -------
        The model with the trained parameters
    
    '''

    writer = SummaryWriter()
    optimizer = optimizer(model.parameters(), lr = lr)
    batch_idx = 0
    if not model_name:
        model_name = 'my_model'
    
    # Make directory for the training session
    datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir = f'model_evaluation/{model_name}_{datetime_now}'
    os.makedirs(dir)

    for epoch in range(epochs):
        for (features, labels) in train_loader: # Load the batch

            #print(features[0]) # This is the picture in torch.Tensor
            #print(labels) # These are certainly the correct labels

            # Train the model with train_dataset
            prediction = model(features) # gives predicted labels. TODO: DEBUG

            loss = F.cross_entropy(prediction, labels)
            loss.backward() # Essentially grad += d(loss)/dx
            optimizer.step() # Update the model weights & biases with the optimizer
            print(f"Epoch{epoch}, batch{batch_idx}: Loss = {loss.item()}")
            optimizer.zero_grad() # otherwise loss.backward() accumulates gradients from all previous batches
            
            writer.add_scalar("Loss-Train", loss.item(), batch_idx)
            batch_idx += 1
        
        # Evaluate the model on the validation dataset
        val_loss, val_accuracy = evaluate(model, validation_loader)
        writer.add_scalar("Loss-Val", val_loss, val_accuracy)

        # Save the model for each epoch
        torch.save(model.state_dict(), os.path.join(dir, f'epoch{epoch}.pt')) # Save only the model's parameters as a dictionary
    
    print("Model training complete - Evaluating on the test dataset")
    test_loss, test_accuracy = evaluate(model, test_loader)
    print(f"Test loss: {test_loss}, accuracy: {test_accuracy}")
    # Need to write this on tensorboard?
    return model

if __name__ == '__main__':
    dataset = Image_Dataset(use_cuda = False)
    # print(type(dataset)) # 'image_dataset.Image_Dataset'
    model = CNN()
    model_name = 'CNN'
    batch_size = 16
    epoch = 1

    # Load datasets
    train_set, validation_set, test_set = split_train_test(dataset, fractions = [0.7, 0.15])
    # print(type(train_set)) # 'torch.utils.data.dataset.Subset'
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)

    # Train the model
    model = train(model, train_loader, validation_loader, test_loader, model_name = model_name, lr = 0.1, epochs = epoch)


