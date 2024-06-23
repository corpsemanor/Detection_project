import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from src.dataset import ChestXrayDataset
from src.model import get_model
from torch.utils.data import DataLoader
import torchvision.transforms as T

def evaluate_model(data_csv, img_dir, num_classes):
    dataset = ChestXrayDataset(csv_file=data_csv, root_dir=img_dir, transform=T.ToTensor())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes)
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                predictions = {
                    'boxes': output['boxes'].cpu().numpy(),
                    'labels': output['labels'].cpu().numpy(),
                    'scores': output['scores'].cpu().numpy()
                }
                all_predictions.append(predictions)

    print('Model evaluation complete.')
    return all_predictions