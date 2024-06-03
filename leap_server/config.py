import torch

# Change to absolute path of this directory
data_path = "/home/yjh/spark_tune/spark_ltr_qo/leap_server"
device = torch.device("cpu")

dataset_name = "stack"

parameters_path = f"{data_path}/data/{dataset_name}/parameters.pkl"
card_model_path = f"{data_path}/data/{dataset_name}/card_model"
cost_model_path = f"{data_path}/data/{dataset_name}/cost_model"
statistics_path = f'{data_path}/data/{dataset_name}/stats.json'
original_training_data_path = f'{data_path}/data/{dataset_name}/train.json'
training_data_path = f'{data_path}/data/{dataset_name}/train_organized.json'
test_data_path = f'{data_path}/data/{dataset_name}/test_organized.json'
eval_data_path = f'{data_path}/data/{dataset_name}/eval_organized.json'
historical_window_path = f'{data_path}/data/{dataset_name}/historical_window.pkl'


def update_paths(dataset_name):
    global parameters_path, card_model_path, cost_model_path, statistics_path
    global original_training_data_path, training_data_path, test_data_path
    global eval_data_path, historical_window_path

    parameters_path = f"{data_path}/data/{dataset_name}/parameters.pkl"
    card_model_path = f"{data_path}/data/{dataset_name}/card_model"
    cost_model_path = f"{data_path}/data/{dataset_name}/cost_model"
    statistics_path = f'{data_path}/data/{dataset_name}/stats.json'
    original_training_data_path = f'{data_path}/data/{dataset_name}/train.json'
    training_data_path = f'{data_path}/data/{dataset_name}/train_organized.json'
    test_data_path = f'{data_path}/data/{dataset_name}/test_organized.json'
    eval_data_path = f'{data_path}/data/{dataset_name}/eval_organized.json'
    historical_window_path = f'{data_path}/data/{dataset_name}/historical_window.pkl'

