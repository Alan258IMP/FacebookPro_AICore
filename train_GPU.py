# Alan (Wentao Li), Imperial College London
# AICore 2022, All rights reserved
import torch

# TODO: Try to train with GPU
# May not be possible / too expensive on cloud computers, so save for later

if __name__ == '__main__':
    # Check cuda availability
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("CUDA is not supported on your machine. Proceed with CPU...")




