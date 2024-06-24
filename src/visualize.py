import sys
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torchvision.transforms as T
from src.model import get_model
from src.logger import setup_logger
from src.dataset import ChestXrayDataset

def visualize_results(model_path, image_path, num_classes, label_map, threshold=0.5):

    logger = setup_logger('visualise_script', '../logs/visualise.log')
    logger.info(f'Starting prediction script for image: {image_path}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        return
    
    model.to(device)
    model.eval()

    # Print model weights for debugging
    for name, param in model.named_parameters():
        logger.info(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")

    try:
        image = Image.open(image_path).convert("RGB")
        logger.info("Image loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading the image: {e}")
        return
    
    transform = T.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    logger.info(f"Image transformed to tensor with shape: {image_tensor.shape}")

    with torch.no_grad():
        outputs = model(image_tensor)
    print(outputs)
    logger.info(f"Model outputs: {outputs}")

    if outputs and 'boxes' in outputs[0] and 'labels' in outputs[0] and 'scores' in outputs[0]:
        boxes = outputs[0]['boxes'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')

        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # display_label = label - 1
                disease_name = label_map[label]
                plt.text(x_min, y_min, f"{disease_name}: {score:.2f}", bbox=dict(facecolor='white', alpha=0.5), fontsize=12, color='red')

        visualization_dir = '../visualization'
        os.makedirs(visualization_dir, exist_ok=True)
        output_path = os.path.join(visualization_dir, f"visualized_{os.path.basename(image_path)}")
        
        plt.savefig(output_path)
        logger.info(f'Visualization saved to {output_path}')
        
        plt.show()
    else:
        logger.info("No valid outputs found or expected keys are missing in the outputs.")

    logger.info('Prediction script completed.')
