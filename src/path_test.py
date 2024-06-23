import torch
print(torch.cuda.is_available())

checkpoints_path = '../models/'
for epoch in range(15):
    print(f"{checkpoints_path}epoch_{epoch+1}.pth")