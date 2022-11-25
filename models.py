# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved

import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 5, stride = 2, padding = 1), # in, out, kernel size, padding;
            # in has 3 because we have RGB - 3 features...?
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 5, stride = 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 5, stride = 2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(5408, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 13), # We have 13 categories
            torch.nn.ReLU(),
            torch.nn.Softmax(dim = 1)
        )
    
    def forward(self, features):
        return self.layers(features)


class Transfer_Resnet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        resnet50.fc = torch.nn.Sequential(
            torch.nn.Linear(resnet50.fc.in_features, 13), # Change final layer - should have 13 output features
            torch.nn.ReLU(),
            torch.nn.Softmax(dim = 1)
        )
        self.layers = resnet50

    def forward(self, features):
        return self.layers(features)
