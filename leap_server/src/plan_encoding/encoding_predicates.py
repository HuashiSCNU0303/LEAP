import re

import numpy as np
import math
import xxhash
from src.plan_encoding.encoding_histogram import scale_value


# 将string按照字符串统一进行哈希
def get_str_representation(value, parameters):
    hash_num = xxhash.xxh64(value, seed=1).intdigest()
    return np.array([(hash_num >> i) & 1 for i in range(parameters.string_encoding_dim - 1, -1, -1)])


def get_histogram(stat, bin_count):
    # bin_count + 1
    histogram = stat['histogram']
    min_, max_ = stat['min'], stat['max']
    scaled_min, scaled_max = scale_value(min_), scale_value(max_)
    lo_values = np.zeros(bin_count + 3)
    lo_values[0], lo_values[1], lo_values[-1] = scaled_min, scaled_max, 1
    if max_ != min_:
        lo_values[2: -1] = [(_['lo'] - min_) / (max_ - min_) for _ in histogram]
    else:
        lo_values[2: -1] = [1 for _ in histogram]
    return lo_values


def encode_condition_op(condition_op, parameters):
    # bool_operator + left_value + compare_operator + right_value
    vec = np.zeros(parameters.condition_op_dim, dtype=float)
    if condition_op is None:
        return vec
    elif condition_op['op_type'] == 'Bool':
        idx = parameters.bool_ops_id[condition_op['operator']]
        vec[idx - 1] = 1
    else:
        operator = condition_op['operator']
        left_value = condition_op['left_value']
        relation_name = left_value.split('.')[0]
        column_name = left_value.split('.')[1]

        left_value_vec = np.zeros(parameters.column_total_num)
        left_value_vec[parameters.columns_id[left_value] - 1] = 1

        right_value = condition_op['right_value']

        # join条件
        if right_value in parameters.columns_id:
            if parameters.columns_id[left_value] >= parameters.columns_id[right_value]:
                if operator == '>':
                    operator = '<'
                elif operator == '<':
                    operator = '>'
                elif operator == '>=':
                    operator = '<='
                elif operator == '<=':
                    operator = '>='
            left_value_vec[parameters.columns_id[right_value] - 1] = 1
            right_value_vec = np.array([0])
            condition_op['histogram'] = get_histogram(parameters.statistics[relation_name][column_name],
                                                      parameters.histogram_bin_count)
            condition_op['selected_bins'] = np.zeros(parameters.histogram_bin_count)
            condition_op['selectivity'] = 0
        # isnull / isnotnull
        elif right_value == "" and operator in ['isnull', 'isnotnull']:
            right_value_vec = np.array([1] if operator == 'isnull' else [0])
        # 数值型
        elif parameters.col_dtypes[f"{relation_name}.{column_name}"] in ['int64', 'float64']:
            right_value = float(right_value)
            value_max = parameters.statistics[relation_name][column_name]['max']
            value_min = parameters.statistics[relation_name][column_name]['min']
            right_value_vec = np.array([(right_value - value_min) / (value_max - value_min) if value_max != value_min
                                        else 1])
        # 其余字符串算子
        else:
            right_value_vec = np.array([0])

        operator_vec = np.zeros(parameters.compare_ops_total_num)
        operator_vec[parameters.compare_ops_id[operator] - 1] = 1

        start_index = parameters.bool_ops_total_num
        vec[start_index: start_index + len(left_value_vec)] = left_value_vec
        start_index += len(left_value_vec)
        vec[start_index: start_index + len(operator_vec)] = operator_vec
        start_index += len(operator_vec)
        vec[start_index: start_index + len(right_value_vec)] = right_value_vec
        start_index += len(right_value_vec)
        vec[start_index: start_index + len(condition_op['histogram'])] = condition_op['histogram']
        start_index += len(condition_op['histogram'])
        vec[start_index: start_index + len(condition_op['selected_bins'])] = condition_op['selected_bins']
        start_index += len(condition_op['selected_bins'])
        vec[-1] = condition_op['selectivity']

    return vec


def encode_condition(condition, parameters):
    if len(condition) == 0:
        vecs = np.zeros((1, parameters.condition_op_dim))
    else:
        vecs = np.zeros((len(condition), parameters.condition_op_dim))
        for index, condition_op in enumerate(condition):
            vecs[index, :] = encode_condition_op(condition_op, parameters)
    return vecs
