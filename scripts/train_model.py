import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import train_model


if __name__ == "__main__":
    data_csv = "../data/CXR8/BBox_List_2017.csv"
    img_dir = "../data/CXR8/images/images"
    num_classes = 8 
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.005
    weight_decay = 0.0005
    momentum = 0.9

    train_model(data_csv, img_dir, num_classes, num_epochs, batch_size, learning_rate, weight_decay, momentum)