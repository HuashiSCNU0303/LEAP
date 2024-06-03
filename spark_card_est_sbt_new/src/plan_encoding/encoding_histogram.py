import xxhash
import math
import re
import numpy as np
from config import *


def get_filtered_bins(cond, stat, bin_count):
    op = cond['operator']
    value = cond['right_value']
    selectivity = 1.0
    selected_bins = np.zeros(bin_count)
    if op == 'isnull':
        selectivity = stat['null_count'] / stat['count']
        return selected_bins, selectivity
    elif op == 'isnotnull':
        selectivity = 1 - stat['null_count'] / stat['count']
        selected_bins = np.ones(bin_count)
        return selected_bins, selectivity

    if 'histogram' in stat:
        bins = stat['histogram']
        value = float(value)
        if op == '<':
            # [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], < 2.5
            selectivity = 0.0
            for i, bin in enumerate(bins):
                if bin['hi'] <= value:
                    portion = 1.0  # 默认全选
                    if bin['hi'] == value:
                        portion -= 1.0 / bin['ndv']
                    selected_bins[i] = portion
                    selectivity += portion / bin_count
                elif bin['lo'] >= value:
                    break
                else:
                    portion = (value - bin['lo']) / (bin['hi'] - bin['lo']) if bin['hi'] != bin['lo'] else 1.0
                    selected_bins[i] = portion
                    selectivity += portion / bin_count
        elif op == '<=':
            # [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], <= 2.5
            selectivity = 0.0
            for i, bin in enumerate(bins):
                if bin['hi'] <= value:
                    portion = 1.0  # 默认全选
                    selected_bins[i] = portion
                    selectivity += portion / bin_count
                elif bin['lo'] > value:
                    break
                else:
                    portion = (value - bin['lo']) / (bin['hi'] - bin['lo']) if bin['hi'] != bin['lo'] else 1.0
                    if portion == 0.0:
                        portion = 1.0 / bin['ndv']
                    selected_bins[i] = portion
                    selectivity += portion / bin_count
        elif op == '>':
            # [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], > 2.5
            selectivity = 0.0
            for idx, bin in enumerate(reversed(bins)):
                i = len(bins) - 1 - idx
                if bin['lo'] >= value:
                    portion = 1.0  # 默认全选
                    if bin['lo'] == value:
                        portion -= 1.0 / bin['ndv']
                    selected_bins[i] = portion
                    selectivity += portion / bin_count
                elif bin['hi'] <= value:
                    break
                else:
                    portion = (bin['hi'] - value) / (bin['hi'] - bin['lo']) if bin['hi'] != bin['lo'] else 1.0
                    selected_bins[i] = portion
                    selectivity += portion / bin_count
        elif op == '>=':
            # [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], >= 2.5
            selectivity = 0.0
            for idx, bin in enumerate(reversed(bins)):
                i = len(bins) - 1 - idx
                if bin['lo'] >= value:
                    portion = 1.0  # 默认全选
                    selected_bins[i] = portion
                    selectivity += portion / bin_count
                elif bin['hi'] < value:
                    break
                else:
                    portion = (bin['hi'] - value) / (bin['hi'] - bin['lo']) if bin['hi'] != bin['lo'] else 1.0
                    if portion == 0.0:
                        portion = 1.0 / bin['ndv']
                    selected_bins[i] = portion
                    selectivity += portion / bin_count
        elif op == '=':
            selectivity = 0.0
            for i, bin in enumerate(bins):
                if bin['hi'] < value:
                    continue
                elif bin['lo'] > value:
                    break
                else:
                    portion = 1.0 / bin['ndv']
                    selected_bins[i] = portion
                    selectivity += portion / bin_count
    else:
        # 字符串类型，or op ==
        # LIKE之类的，暂时不支持，直接返回个全选的
        if op == '=':
            # 计算hash值，然后取余，落在哪个桶，就加1 / ndv
            bin_index = xxhash.xxh64(value, seed=1).intdigest() % bin_count
            if value in stat['mcv_list']:
                selectivity = stat['mcv_list'][value] / stat['count']  # 直接就是选择率
            else:
                selectivity = 1.0 / stat['distinct_count']
            selected_bins[bin_index] = selectivity

    return selected_bins, selectivity


def scale_value(value):
    domain = [-2, 10]
    if value == 0:
        return 0
    elif value > 0:
        # 映射到(0, 1]
        log_value = math.log10(value)
        return (log_value - domain[0]) / (domain[1] - domain[0])
    else:
        # 映射到[-1, 0)
        log_value = math.log10(-value)
        return -(log_value - domain[0]) / (domain[1] - domain[0])


def process_in(in_values, tab_col, bin_count, stat):
    new_conditions = [{'op_type': 'Compare',
                       'operator': '=',
                       'left_value': tab_col,
                       'right_value': value} for value in in_values]
    selected_bins, selectivity = np.zeros(bin_count), 0
    for cond in new_conditions:
        sub_selected_bins, sub_selectivity = get_filtered_bins(cond, stat, bin_count)
        selectivity += sub_selectivity
        selected_bins += np.array(sub_selected_bins)
    return selected_bins, selectivity


def sql_like_to_regex(sql_pattern):
    sql_pattern = sql_pattern.replace('_', '\\_')
    sql_pattern = sql_pattern.replace('(', '\\(')
    sql_pattern = sql_pattern.replace(')', '\\)')
    sql_pattern = sql_pattern.replace('%', '.*')
    regex_pattern = '^' + sql_pattern + '$'
    return regex_pattern


def add_histograms(plan, stats, bin_count=64):
    columns = set()
    for table, table_stat in stats.items():
        for col in table_stat.keys():
            columns.add(f"{table}.{col}")
    for node in plan:
        if 'condition' in node:
            for condition in node['condition']:
                if condition is not None and condition['op_type'] == 'Compare' and condition['right_value'] not in columns:
                    tab_col = condition['left_value']
                    tab, col = tab_col.split(".")[0], tab_col.split(".")[1]
                    stat = stats[tab][col]
                    if dataset_name == "stack" and col.find('date') != -1 and condition['operator'] not in ['isnull', 'isnotnull']:
                        # 纳秒变成秒
                        condition['right_value'] = str(float(condition['right_value']) // 1e6)
                    if 'histogram' in stat:
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
                        condition['histogram'] = lo_values
                    else:
                        condition['histogram'] = np.zeros(bin_count + 3)

                    # LIKE系列的谓词，基于MCV转变为IN，如果MCV内没有，就不管了
                    if condition['operator'] in ['Contains', 'StartsWith', 'EndsWith', 'LIKE']:
                        value = condition['right_value']
                        mcvs = stat['mcv_list'].keys()
                        if condition['operator'] == 'Contains':
                            qualified_mcvs = [cv for cv in mcvs if cv.find(value) != -1]
                        elif condition['operator'] == 'StartsWith':
                            qualified_mcvs = [cv for cv in mcvs if cv.startswith(value)]
                        elif condition['operator'] == 'EndsWith':
                            qualified_mcvs = [cv for cv in mcvs if cv.endswith(value)]
                        elif condition['operator'] == 'LIKE':
                            qualified_mcvs = [cv for cv in mcvs if re.fullmatch(sql_like_to_regex(value), cv)]
                        else:
                            qualified_mcvs = []
                        if len(qualified_mcvs) == 0:
                            # 假定剩下的全都被选中
                            selected_bins = np.zeros(bin_count)
                            selectivity = (stat['count'] - stat['mcvs_total_count']) / stat['count']
                        else:
                            # 变成IN
                            selected_bins, selectivity = process_in(qualified_mcvs, tab_col, bin_count, stat)
                            # 返回值全部乘上一个系数，假定剩下的也均匀分布
                            ratio = max(1, stat['mcvs_total_count']) / stat['count']
                            selectivity /= ratio
                            selected_bins = [_ / ratio for _ in selected_bins]

                    # 如果是IN，拆成若干个等于，然后加起来
                    elif condition['operator'] == 'IN':
                        # 去除头尾的双引号（忘记加双引号了）
                        in_values = [_[1: -1] for _ in condition['right_value'].split("|[,]|")]
                        selected_bins, selectivity = process_in(in_values, tab_col, bin_count, stat)
                    else:
                        selected_bins, selectivity = get_filtered_bins(condition, stat, bin_count)
                    condition['selected_bins'] = selected_bins
                    condition['selectivity'] = selectivity
    return plan
