import sys
import os
import time
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from src.dataset import ChestXrayDataset
from src.model import get_model
from src.logger import setup_logger

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(data_csv, img_dir, num_classes, num_epochs, batch_size, learning_rate, weight_decay, momentum):
    
    save_path = '../models'

    # Initialize dataset and data loader
    dataset = ChestXrayDataset(csv_file=data_csv, root_dir=img_dir, transform=T.ToTensor())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Set device to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Get the model and move it to the device
    model = get_model(num_classes)
    model.to(device)

    # Initialize optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()

        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(data_loader)}")

        # Save the model checkpoint after each epoch
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch + 1}.pth"))

    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))
