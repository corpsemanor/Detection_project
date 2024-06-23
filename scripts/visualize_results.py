import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualize import visualize_results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_results.py <path_to_image>")
        sys.exit(1)
    
    model_path = "../models/model.pth"
    num_classes = 8

    image_path = sys.argv[1]
    visualize_results(model_path, image_path, num_classes)