import torch

print("Is CUDA available: ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)
print("cuDNN version: ", torch.backends.cudnn.version())