import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluate import evaluate_model

if __name__ == "__main__":
    data_csv = "../data/CXR8/BBox_List_2017.csv"
    img_dir = "../data/CXR8/images/images"
    num_classes = 8

    predictions = evaluate_model(data_csv, img_dir, num_classes)