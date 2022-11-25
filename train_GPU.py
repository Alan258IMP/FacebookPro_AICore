# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved
import torch
from models import CNN, Transfer_Resnet50

# TODO: Try to train with GPU
# May not be possible / too expensive on cloud computers, so save for later

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check cuda availability
    # if not torch.cuda.is_available():
    #     raise Exception("Cuda is not available on your device. Run train.py to proceed with CPU.")
    
    




