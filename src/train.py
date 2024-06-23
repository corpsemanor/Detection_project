import sys
import os
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

    checkpoints_path = '../models/'
    logger = setup_logger('train_script', '../logs/train.log')
    logger.info('Starting training script...')

    dataset = ChestXrayDataset(csv_file=data_csv, root_dir=img_dir, transform=T.ToTensor())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 5 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")
            i += 1

        lr_scheduler.step()
        torch.save(model.state_dict(), f"{checkpoints_path}epoch_{epoch}.pth")
    
    logger.info('Training script completed.')
