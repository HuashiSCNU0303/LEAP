import pickle
import json
import math
from src.internal_parameters import Parameters
from config import *


logical_ops_id = {'Inner': 1, 'Filter': 2, 'Aggregate': 3, ') Inner': 4}
compare_ops_id = {'=': 1, '>': 2, '<': 3, '!=': 4, '>=': 5, '<=': 6, 'isnull': 7, 'isnotnull': 8,
                  'Contains': 9, 'EndsWith': 10, 'StartsWith': 11, 'LIKE': 12, 'IN': 13}
join_compare_ops_id = {'=': 1, '>': 2, '<': 3, '!=': 4, '>=': 5, '<=': 6}
bool_ops_id = {'AND': 1, 'OR': 2, 'NOT': 3, ') AND': 4, ') OR': 5, ') NOT': 6}


def load_statistics(path):
    with open(path, 'r') as f:
        statistics = json.loads(f.read())
    return statistics


def obtain_upper_bound_query_size(path):
    card_label_max = 0.0
    card_label_min = 9999999999.0
    plans = []
    with open(path, 'r') as f:
        for plan in f.readlines():
            plan = json.loads(plan)
            plans.append(plan)
            sequence = plan['seq']
            for node in sequence:
                if node is not None:
                    cardinality = node['cardinality']
                    if cardinality > card_label_max:
                        card_label_max = cardinality
                    elif cardinality < card_label_min:
                        card_label_min = cardinality
    card_label_min, card_label_max = math.log(max(1, card_label_min)), math.log(card_label_max)
    return card_label_min, card_label_max


def gen_col_dtypes(statistics):
    col_dtypes = {}
    for tab, col_stats in statistics.items():
        for col, stat in col_stats.items():
            if 'max' in stat:
                col_dtypes[f"{tab}.{col}"] = 'float64'
            else:
                col_dtypes[f"{tab}.{col}"] = 'object'
    return col_dtypes


def prepare_dataset(statistics):
    tables_id = {}
    columns_id = {}
    table_id = 1
    column_id = 1
    # 所有ID从1开始，后面encode node的时候注意id - 1
    for tab, col_stats in statistics.items():
        tables_id[tab] = table_id
        table_id += 1
        for col, stat in col_stats.items():
            columns_id[f"{tab}.{col}"] = column_id
            column_id += 1

    return tables_id, columns_id


if __name__ == "__main__":
    statistics = load_statistics(statistics_path)
    tables_id, columns_id = prepare_dataset(statistics)
    col_types = gen_col_dtypes(statistics)
    print('statistics')

    card_label_min, card_label_max = obtain_upper_bound_query_size(original_training_data_path)
    print(card_label_min)
    print(card_label_max)

    # Parameters存pickle然后读取
    parameters = Parameters(tables_id, columns_id, logical_ops_id, compare_ops_id, join_compare_ops_id,
                            bool_ops_id, col_types, statistics, card_label_min, card_label_max)

    pickle.dump(parameters, open(parameters_path, "wb"))