import torch

# 改成本目录的绝对路径
data_path = "/home/yjh/spark_tune/spark_ltr_qo/spark_card_est_sbt_new"
dataset_name = "stack"

# 下面的一般不需要改
samples_path = f"{data_path}/data/{dataset_name}/samples"
parameters_path = f"{data_path}/data/{dataset_name}/parameters.pkl"
card_model_path = f"{data_path}/data/{dataset_name}/card_model"
cost_model_path = f"{data_path}/data/{dataset_name}/cost_model"
statistics_path = f'{data_path}/data/{dataset_name}/stats.json'
original_training_data_path = f'{data_path}/data/{dataset_name}/train.json'
training_data_path = f'{data_path}/data/{dataset_name}/train_organized.json'
test_data_path = f'{data_path}/data/{dataset_name}/test_organized.json'
eval_data_path = f'{data_path}/data/{dataset_name}/eval_organized.json'
historical_window_path = f'{data_path}/data/{dataset_name}/historical_window.pkl'

device = torch.device("cpu")
