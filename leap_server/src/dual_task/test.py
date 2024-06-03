import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.dual_task.train_and_test import train
from config import *
import pickle
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help="The number of training epochs.")
    parser.add_argument('--mode', type=str, default='card', help="The number of training epochs.")
    args = parser.parse_args()

    parameters = pickle.load(open(parameters_path, "rb"))
    model_path = card_model_path if args.mode == 'card' else cost_model_path

    train(args.epochs, parameters, training_data_path, test_data_path,
          model_path, args.mode, eval_data_path)
