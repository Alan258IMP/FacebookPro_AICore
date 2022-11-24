# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

import torch

class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 5, padding = 2), # in, out, kernel size, padding;
            # in has 3 because we have RGB - 3 features...?
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 5),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(8, 16, 5),
            # torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(508032, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 13), # We have 13 categories
            torch.nn.ReLU(),
            torch.nn.Softmax(dim = 1)
        )
    
    def forward(self, features):
        return self.layers(features) 
        # TODO: DEBUG
        # line 459, in _conv_forward:
        # return F.conv2d(input, weight, bias, self.stride,
        # Exception has occurred: RuntimeError
        # Input type (unsigned char) and bias type (float) should be the same
