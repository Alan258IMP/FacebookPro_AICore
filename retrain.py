# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved
import torch
from models import CNN, Transfer_Resnet50
from train import split_train_test, evaluate, train

if __name__ == '__main__':
    # # Continue training with existing model?
    # continue_training = True
    # model_path = 'final_models/TransferResnet50_24epochs.pt'
    # if continue_training:
    #     model.load_state_dict(torch.load(model_path))
    # model.to(device)
    # Why this does not work?
    pass
