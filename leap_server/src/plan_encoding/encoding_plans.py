import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.plan_encoding.encoding_predicates import *


class TreeNode(object):
    def __init__(self, current_vec, parent, idx, level_id):
        self.item = current_vec
        self.idx = idx
        self.level_id = level_id
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

    def get_idx(self):
        return self.idx

    def __str__(self):
        return f"item = {self.item}"
        # return 'level_id: ' + str(self.level_id) + '; idx: ' + str(self.idx)


def recover_tree(vecs, parent, start_idx):
    if len(vecs) == 0:
        return vecs, start_idx
    if vecs[0] == None:
        return vecs[1:], start_idx + 1
    node = TreeNode(vecs[0], parent, start_idx, -1)
    while True:
        vecs, start_idx = recover_tree(vecs[1:], node, start_idx + 1)
        parent.add_child(node)
        if len(vecs) == 0:
            return vecs, start_idx
        if vecs[0] == None:
            return vecs[1:], start_idx + 1
        node = TreeNode(vecs[0], parent, start_idx, -1)


def print_tree(node, indent=""):
    if node is not None:
        print(indent + str(node))
        indent += "  "
        for child in node.children:
            print_tree(child, indent)


def encode_node_job(node, parameters):
    operator_vec = np.zeros(parameters.logical_op_total_num)
    condition1_vec = np.zeros((1, parameters.condition_op_dim))
    table_one_hot_vec = np.zeros(parameters.table_total_num)
    cardinality = 1

    if node is not None:
        operator = node['node_type']
        operator_vec[parameters.logical_ops_id[operator] - 1] = 1

        if operator in ['Inner', ') Inner']:
            condition1_vec = encode_condition(node['condition'], parameters)
            if len(node['condition']) != 0:
                for cond in node['condition']:
                    if cond['op_type'] == 'Compare':
                        table_one_hot_vec[parameters.tables_id[cond['left_value'].split(".")[0]] - 1] = 1
                        if cond['right_value'] in parameters.columns_id:
                            table_one_hot_vec[parameters.tables_id[cond['right_value'].split(".")[0]] - 1] = 1
        elif operator == 'Filter':
            relation_name = node['relation_name']
            condition1_vec = encode_condition(node['condition'], parameters)
            table_one_hot_vec[parameters.tables_id[relation_name] - 1] = 1
            cardinality = node['table_cardinality']
    return condition1_vec, cardinality, table_one_hot_vec, operator_vec


def encode_plan_job(plan, parameters):
    num_nodes = len(plan)

    num_filter_conds = np.array([max(1, len(node['condition'])) for node in plan])
    filter_conds = np.zeros((num_nodes, max(num_filter_conds), parameters.condition_op_dim), dtype=float)
    tables = np.zeros((num_nodes, parameters.table_total_num), dtype=float)
    cards = np.zeros((num_nodes, 1), dtype=float)
    operators = np.zeros((num_nodes, parameters.logical_op_total_num), dtype=float)

    # i肯定是偶数
    for i in range(0, len(plan)):
        node = plan[i]
        condition, cardinality, table_one_hot, operator = encode_node_job(node, parameters)

        # 其余部分不设置，保持为0
        filter_conds[i, :condition.shape[0], :] = condition
        tables[i, :] = table_one_hot
        cards[i, :] = np.array([cardinality])
        operators[i, :] = operator

    return filter_conds, tables, cards, num_filter_conds, operators
