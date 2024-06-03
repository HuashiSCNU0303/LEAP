import copy
import random
from src.plan_encoding.encoding_histogram import get_filtered_bins, scale_value, process_in
from config import *
import numpy as np


class TreeNode(object):
    def __init__(self, current_vec, parent):
        self.item = current_vec
        self.parent = parent
        self.children = []

    def get_parent(self):
        return self.parent

    def get_item(self):
        return self.item

    def get_children(self):
        return self.children

    def add_child(self, child):
        self.children.append(child)

    def __str__(self):
        return f"item = {self.item}"


def recover_tree(sequence):
    root = TreeNode(None, None)
    stack = [root]

    for item in sequence:
        if not item['node_type'].startswith(') ') and item['node_type'] != 'Filter':
            new_node = TreeNode(item, parent=stack[-1])
            stack[-1].add_child(new_node)
            stack.append(new_node)
        elif item['node_type'].startswith(") "):
            stack.pop()
        else:
            leaf_node = TreeNode(item, parent=stack[-1])
            stack[-1].add_child(leaf_node)

    return root.children[0]


def filter_columns(columns, exclude_patterns):
    return [col for col in columns if all(pattern not in col for pattern in exclude_patterns)]


# 生成新的谓词
def gen_new_cond(parameters, table, bin_count):
    cols = parameters.statistics[table]
    if dataset_name == 'tpch':
        exclude_patterns = ["key", "phone", "comment", "clerk", "s_name", "p_name", "c_name", "address"]
    else:
        exclude_patterns = ["id", "body", "title", "about_me", "date", "tagstring"]
    qualified_columns = filter_columns(cols.keys(), exclude_patterns)

    if len(qualified_columns) == 0:
        return None
    random_col = random.choice(qualified_columns)
    stat = cols[random_col]
    if 'max' in stat and len(stat['histogram']) > 0:
        operator = random.choice(['=', '>', '<', '>=', '<='])
        value = random.randint(int(stat['min']), int(stat['max']))
        min_, max_ = stat['min'], stat['max']
        scaled_min, scaled_max = scale_value(min_), scale_value(max_)
        lo_values = np.zeros(bin_count + 3)
        lo_values[0], lo_values[1], lo_values[-1] = scaled_min, scaled_max, 1
        if max_ != min_:
            lo_values[2: -1] = [(_['lo'] - min_) / (max_ - min_) for _ in stat['histogram']]
        else:
            lo_values[2: -1] = [1 for _ in stat['histogram']]
        histogram = lo_values
    elif 'mcv_list' in stat and len(stat['mcv_list']) > 0:
        operator = random.choice(['IN', 'LIKE', 'StartsWith', 'EndsWith', 'Contains', '='])
        histogram = np.zeros(bin_count + 3)
        value = random.choice(list(stat['mcv_list'].keys()))
    else:
        return None

    new_cond = {
        'op_type': 'Compare',
        'operator': operator,
        'left_value': f"{table}.{random_col}",
        'right_value': value,
        'histogram': histogram
    }
    if operator in ['IN', 'LIKE', 'StartsWith', 'EndsWith', 'Contains']:
        in_values = random.sample(list(stat['mcv_list'].keys()), random.randint(1, min(8, len(stat['mcv_list'].keys()))))
        selected_bins, selectivity = process_in(in_values, new_cond['left_value'], bin_count, stat)
    else:
        selected_bins, selectivity = get_filtered_bins(new_cond, stat, bin_count)
    new_cond['selected_bins'] = selected_bins
    new_cond['selectivity'] = selectivity
    return new_cond


def alter_join_order(plan):
    if len(plan) == 1:
        return {'seq': plan}
    has_replaced = False

    def random_pre_order_traversal(node):
        nonlocal has_replaced
        nodes = [node.item]
        children = node.children
        if len(node.children) == 2 and random.random() < 0.4:
            has_replaced = True
            children = [node.children[1], node.children[0]]
        for child in children:
            nodes.extend(random_pre_order_traversal(child))
        if node.item['node_type'] != 'Filter':
            new_node = copy.deepcopy(node.item)
            new_node['node_type'] = f") {new_node['node_type']}"
            nodes.append(new_node)
        return nodes

    plan_tree = recover_tree(plan)
    seq = random_pre_order_traversal(plan_tree)
    return {'seq': seq} if has_replaced else None


def gen_new_plan(plan, is_bigger, parameters):
    has_replaced = False
    alter_filter_cnt_prob = 0.7
    bin_count = parameters.histogram_bin_count
    new_plan = alter_join_order(plan['seq'])
    if new_plan is None:
        return None
    new_plan = copy.deepcopy(new_plan['seq'])
    for node in new_plan:
        if node['node_type'] == 'Filter':
            table = node['relation_name']
            for cond_i, cond in enumerate(node['condition']):
                # 前面一个condition不能是NOT
                if cond['op_type'] == 'Compare' and (cond_i == 0 or node['condition'][cond_i - 1]['operator'] != 'NOT'):
                    left_value = cond['left_value']
                    tab, col = left_value.split(".")[0], left_value.split(".")[1]
                    # 数值型
                    if cond['operator'] in ['>', '>=', '<', '<=', '='] and 'max' in parameters.statistics[tab][col]:
                        stat = parameters.statistics[tab][col]
                        right_value = float(cond['right_value'])
                        min_, max_ = stat['min'], stat['max']
                        new_op, new_right_value = cond['operator'], right_value
                        if cond['operator'] in ['>', '>=']:
                            # >/>= a, bigger: >/>= [min, a); smaller, >/>=/= (a, max]
                            if is_bigger:
                                new_op, new_right_value = random.choice(['>', '>=']), np.random.uniform(min_, right_value)
                            else:
                                new_op, new_right_value = random.choice(['>', '>=', '=']), np.random.uniform(right_value + 1, max_)
                        elif cond['operator'] in ['>', '>=']:
                            # </<= a, bigger: </<= (a, max]; smaller, </<=/= [min, a)
                            if is_bigger:
                                new_op, new_right_value = random.choice(['<', '<=']), np.random.uniform(right_value + 1, max_)
                            else:
                                new_op, new_right_value = random.choice(['<', '<=', '=']), np.random.uniform(min_, right_value)
                        elif cond['operator'] == '=':
                            if is_bigger:
                                # = a, bigger: >/>= [min, a), </<= (a, max]; smaller: 没有
                                if random.random() < 0.5:
                                    new_op, new_right_value = random.choice(['>', '>=']), np.random.uniform(min_, right_value)
                                else:
                                    new_op, new_right_value = random.choice(['<', '<=']), np.random.uniform(right_value + 1, max_)
                        cond['operator'] = new_op
                        cond['right_value'] = str(int(new_right_value))
                        selected_bins, selectivity = get_filtered_bins(cond, stat, bin_count)
                        cond['selected_bins'] = selected_bins
                        cond['selectivity'] = selectivity
                        # print(cond)
                        # print("————————")
                        has_replaced = True
                    elif cond['operator'] in ['IN', 'StartsWith', 'EndsWith', 'Contains', 'LIKE']:
                        min_bin, max_bin = min(cond['selected_bins']), max(cond['selected_bins'])
                        stat = parameters.statistics[tab][col]
                        ndv = 1.0 / stat['distinct_count']
                        if is_bigger:
                            # 生成一个更大的谓词
                            for i in range(len(cond['selected_bins'])):
                                selected_bin = cond['selected_bins'][i]
                                if selected_bin == 0:
                                    # 无中生有，要稀疏的随机添加
                                    if random.random() < 0.2:
                                        cond['selected_bins'][i] = random.uniform(ndv, 0.5 * max_bin)
                                else:
                                    # 原来就有，加一点
                                    cond['selected_bins'][i] = random.uniform(selected_bin, max_bin)
                        else:
                            # 去除部分桶的取值
                            for i in range(len(cond['selected_bins'])):
                                selected_bin = cond['selected_bins'][i]
                                if selected_bin != 0:
                                    # 有的再减小一点
                                    if ndv < selected_bin and random.random() < 0.5:
                                        cond['selected_bins'][i] = random.uniform(ndv, selected_bin)
                                    else:
                                        cond['selected_bins'][i] = 0
                        # 变化拉大一点
                        cond['operator'] = random.choice(['IN', 'StartsWith', 'EndsWith', 'Contains', 'LIKE'])
                        cond['selectivity'] = min(1.0, sum(cond['selected_bins']))
                        has_replaced = True
            if random.random() < alter_filter_cnt_prob:
                and_cond_node = {'op_type': 'Bool', 'operator': 'AND'}
                end_and_cond_node = {'op_type': 'Bool', 'operator': ') AND'}
                if is_bigger:
                    # 减少谓词
                    if len(node['condition']) >= 4:
                        # 如果有OR，直接跳过；如果没有，把所有Compare拿出来，随机选择一部分拼成左深
                        is_disjunctive = any(cond['operator'] == 'OR' for cond in node['condition'])
                        if not is_disjunctive:
                            # 把所有compare拿出来，随机选择一部分拼成左深树
                            all_compares = [cond for i, cond in enumerate(node['condition'])
                                            if cond['op_type'] == 'Compare' and node['condition'][i - 1]['operator'] != 'NOT']
                            sampled_compares = random.sample(all_compares, random.randint(1, max(1, len(all_compares) - 1)))
                            # 基于sampled_compares构建左深树，然后替代node['condition']
                            # AND AND AND A B )AND C )AND D )AND
                            ands = [and_cond_node] * (len(sampled_compares) - 1)
                            ands.append(sampled_compares[0])
                            for cond in sampled_compares[1:]:
                                ands.append(cond)
                                ands.append(end_and_cond_node)
                            node['condition'] = ands
                            has_replaced = True
                else:
                    # 增加谓词
                    new_cond = gen_new_cond(parameters, table, bin_count)
                    if new_cond is None:
                        continue

                    # 任选一个Compare C，将C替换为(C AND 新生成谓词)
                    compare_ids = [i for i, cond in enumerate(node['condition']) if cond['op_type'] == 'Compare']
                    selected_compare_id = random.choice(compare_ids)
                    # 前面不能是NOT
                    while selected_compare_id != 0 and node['condition'][selected_compare_id - 1]['operator'] == 'NOT':
                        selected_compare_id = random.choice(compare_ids)
                    selected_compare = node['condition'][selected_compare_id]
                    new_compare = [and_cond_node, selected_compare, new_cond, end_and_cond_node]
                    new_conditions = node['condition'][: selected_compare_id] + new_compare + node['condition'][selected_compare_id + 1:]
                    node['condition'] = new_conditions
                    has_replaced = True

    return {'seq': new_plan} if has_replaced else None
