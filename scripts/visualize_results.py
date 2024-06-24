import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualize import visualize_results
from src.dataset import ChestXrayDataset
import torchvision.transforms as T

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_results.py <path_to_image>")
        sys.exit(1)

    data_csv = "../data/CXR8/BBox_List_2017.csv"
    img_dir = "../data/CXR8/images/images"
    model_path = "../models/final_model.pth"
    num_classes = 8

    image_path = sys.argv[1]
    
    dataset = ChestXrayDataset(csv_file=data_csv, root_dir=img_dir, transform=T.ToTensor())
    label_map = dataset.get_label_map()
    print(label_map)
    visualize_results(model_path, image_path, num_classes, label_map)
