import copy
import numpy as np
import random

from src.plan_encoding.tensor_util import pad_batch_2D, pad_batch_3D, pad_batch_4D, normalize_label
from src.plan_encoding.encoding_plans import encode_plan_job
from src.dual_task.model import DualComparator, device
from src.dual_task.data_augmentor import gen_new_plan
from config import dataset_name
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassFBetaScore

import torch
import time
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from collections import defaultdict
import xxhash


label_smoothing_factor = 0.01
epoch_id = 0


class PlanDataset(Dataset):
    def __init__(self, plans):
        self.plans = plans

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, idx):
        return self.plans[idx]


def recover_plan_tree(plan, start_index, layer_dict, level):
    # (A (B C D B) (E F G E) A) -> {0: A, 1: [B, E], 2: [C, D, F, G]}
    layer_dict[level].append(start_index)  # plan[start_index], 改为存索引
    if plan[start_index]['node_type'] == 'Filter':
        # 叶结点
        return start_index
    else:
        # 非叶结点
        end_index = -1
        for i in range(start_index + 1, len(plan)):
            if 'join_id' in plan[i] and plan[i]['join_id'] == plan[start_index]['join_id']:
                end_index = i
                break
        if end_index == -1:
            raise ValueError

        # 处理左孩子
        left_child_end_index = recover_plan_tree(plan, start_index + 1, layer_dict, level + 1)
        # 处理右孩子
        right_child_end_index = recover_plan_tree(plan, left_child_end_index + 1, layer_dict, level + 1)

        return end_index


def gen_label(plan, mode):
    if mode == 'cost':
        # 传进来的是{'seq': [], 'label': {}}
        # 每一层的基数的最大值加起来
        label = plan['label']
    elif mode == 'card':
        label = plan['seq'][0]['cardinality']
    else:
        label = 0
    return label


def make_data_job(plans, parameters, mode):
    card_batch = np.zeros(len(plans), dtype=object)
    filter_conds_batch = np.zeros(len(plans), dtype=object)
    join_conds_batch = np.zeros(len(plans), dtype=object)
    tables_batch = np.zeros(len(plans), dtype=object)
    operators_batch = np.zeros(len(plans), dtype=object)
    num_filter_conds_batch = np.zeros(len(plans), dtype=object)
    num_join_conds_batch = np.zeros(len(plans), dtype=object)
    filter_cond_masks_batch = np.zeros(len(plans), dtype=object)
    target_batch = []

    for index, plan in enumerate(plans):
        filter_conds, join_conds, tables, cards, num_filter_conds, num_join_conds, operators, filter_cond_masks = encode_plan_job(
            plan['seq'], parameters)
        tables_batch[index] = tables
        filter_conds_batch[index] = filter_conds
        join_conds_batch[index] = join_conds
        target_batch.append(gen_label(plan, mode))
        num_filter_conds_batch[index] = num_filter_conds
        num_join_conds_batch[index] = num_join_conds
        operators_batch[index] = operators
        filter_cond_masks_batch[index] = filter_cond_masks

        cards = normalize_label(cards, parameters.card_label_min, parameters.card_label_max)
        card_batch[index] = cards

    return card_batch, filter_conds_batch, tables_batch, num_filter_conds_batch, operators_batch, target_batch


def gen_batch_data(cards_batch, filter_conds_batch, tables_batch, num_filter_conds_batch, operator_batch):
    return {
        'num_steps_batch': [len(_) for _ in cards_batch],
        'num_filter_conds_batch': pad_batch_2D(num_filter_conds_batch, padding_value=1),
        # batch_size × num_steps × num_conditions × condition_dim
        'filter_conds_batch': pad_batch_4D(filter_conds_batch),
        # batch_size × num_steps × data_dim
        'tables_batch': pad_batch_3D(tables_batch),
        'cards_batch': pad_batch_3D(cards_batch),
        'operators_batch': pad_batch_3D(operator_batch)
    }


def gen_card_batch(batch, mapping, type, parameters):
    batch_1, batch_2 = [], []
    max_len = max(list(mapping.keys()))
    min_len = min(list(mapping.keys()))
    for plan in list(batch):
        card = gen_label(plan, 'card')
        card_len = len(str(card))
        if type != 'train':
            sample = random.choice(mapping[6])
            batch_1.append(plan)
            batch_2.append(sample)
        else:
            index = 3 + (plan['id'] + epoch_id) % 7  # [3, 9]
            # index = random.randint(min_len, max_len)
            sample = random.choice(mapping[index])
            batch_1.append(plan)
            batch_2.append(sample)

            sample_card = gen_label(sample, 'card')

            if card >= sample_card:
                new_plan = gen_new_plan(plan, True, parameters)
                new_sample = gen_new_plan(sample, False, parameters)
                if new_plan is not None and new_sample is not None: 
                    batch_1.append(new_plan)
                    batch_2.append(new_sample)
            else:
                new_plan = gen_new_plan(plan, False, parameters)
                new_sample = gen_new_plan(sample, True, parameters)
                if new_plan is not None and new_sample is not None: 
                    batch_1.append(new_plan)
                    batch_2.append(new_sample)

    return batch_1, batch_2


def gen_cost_batch(batch, mapping, type, parameters):
    batch_1, batch_2 = [], []
    max_len = max(list(mapping.keys()))
    min_len = min(list(mapping.keys()))
    for plan in list(batch):
        card = gen_label(plan, 'cost')
        card_len = len(str(card))
        if type != 'train':
            index = len(plan['seq'])
            sample = random.choice(mapping[index])
            sample_card = gen_label(sample, 'cost')

            i = 0
            while max(sample_card / card, card / sample_card) < 5:
                index = random.choice(list(mapping.keys()))
                sample = random.choice(mapping[index])
                sample_card = gen_label(sample, 'cost')
                i += 1
                if i >= 20:
                    break
            if i >= 20:
                continue

            batch_1.append(plan)
            batch_2.append(sample)
        else:
            index = random.choice(list(mapping.keys()))
            sample = random.choice(mapping[index])

            lens = [4, 5, 6, 7, 8, 9, 10]  # [10, 12]
            sample_len = lens[(plan['id'] + epoch_id) % len(lens)]

            sample_card = gen_label(sample, 'cost')
            i = 0
            while len(str(sample_card)) != sample_len:
                index = random.choice(list(mapping.keys()))
                sample = random.choice(mapping[index])
                sample_card = gen_label(sample, 'cost')
                i += 1
                if i >= 10000:
                    break
            if i >= 10000:
                index = random.choice(list(mapping.keys()))
                sample = random.choice(mapping[index])
                sample_card = gen_label(sample, 'cost')

            if plan['id'] == 300:
                print(f"id = 300, label = {card}, sample = {sample_card}, sample_len = {sample_len}")

            batch_1.append(plan)
            batch_2.append(sample)

            if card >= sample_card:
                new_plan = gen_new_plan(plan, True, parameters)
                new_sample = gen_new_plan(sample, False, parameters)
                if new_plan is not None and new_sample is not None:
                    batch_1.append({'seq': new_plan['seq'], 'label': plan['label']})
                    batch_2.append({'seq': new_sample['seq'], 'label': sample['label']})
            else:
                new_plan = gen_new_plan(plan, False, parameters)
                new_sample = gen_new_plan(sample, True, parameters)
                if new_plan is not None and new_sample is not None:
                    batch_1.append({'seq': new_plan['seq'], 'label': plan['label']})
                    batch_2.append({'seq': new_sample['seq'], 'label': sample['label']})

    return batch_1, batch_2


def custom_collate(batch, parameters, mapping, mode, type='train'):
    if mode == 'card':
        batch_1, batch_2 = gen_card_batch(batch, mapping, type, parameters)
    else:
        batch_1, batch_2 = gen_cost_batch(batch, mapping, type, parameters)

    cards_batch, filter_conds_batch, tables_batch, num_filter_conds_batch, operator_batch, target_batch_1 = make_data_job(batch_1, parameters, mode)
    batch_1_data = gen_batch_data(cards_batch, filter_conds_batch, tables_batch, num_filter_conds_batch, operator_batch)

    cards_batch, filter_conds_batch, tables_batch, num_filter_conds_batch, operator_batch, target_batch_2 = make_data_job(batch_2, parameters, mode)
    batch_2_data = gen_batch_data(cards_batch, filter_conds_batch, tables_batch, num_filter_conds_batch, operator_batch)

    tensor_1, tensor_2 = torch.tensor(target_batch_1), torch.tensor(target_batch_2)
    orders = torch.where(tensor_1 >= tensor_2, torch.tensor(1 - label_smoothing_factor), torch.tensor(label_smoothing_factor)).float()
    return batch_1_data, batch_2_data, orders, tensor_1, tensor_2


def load_plans(path, parameters):
    plans = []
    with open(path, 'r') as f:
        for idx, seq in enumerate(f.readlines()):
            plan = json.loads(seq)
            plans.append(plan)
    extended_plans = extend_plans(plans, parameters)
    print(f"original plans = {len(plans)}, extended plans = {len(extended_plans)}")
    return extended_plans


def extend_plans(plans, parameters):
    def get_join_id(value):
        return xxhash.xxh64(value, seed=1).intdigest()

    extended_plans = []
    table_plan_strs = set()
    for plan in plans:
        plan = plan['seq']

        # 添加join id
        for i in range(len(plan)):
            if plan[i]['node_type'] == 'Inner':
                plan[i]['join_id'] = get_join_id(json.dumps(plan[i]))
            elif plan[i]['node_type'] == ') Inner':
                forward_node = copy.deepcopy(plan[i])
                forward_node['node_type'] = forward_node['node_type'][2:]  # 去除') '
                plan[i]['join_id'] = get_join_id(json.dumps(forward_node))

        # 不需要还原为树结构。然后去重
        for i in range(len(plan)):
            if plan[i]['node_type'] == 'Filter':
                single_table_str = json.dumps(plan[i])
                if single_table_str not in table_plan_strs:
                    extended_plans.append({'seq': [plan[i]]})
                    table_plan_strs.add(single_table_str)
            elif plan[i]['node_type'] == 'Inner':
                for j in range(i + 1, len(plan)):
                    if plan[j]['node_type'] == ') Inner' and plan[j]['join_id'] == plan[i]['join_id']:
                        plan_str = json.dumps({'seq': plan[i: j + 1]})
                        if plan_str not in table_plan_strs:
                            extended_plans.append({'seq': plan[i: j + 1]})
                            table_plan_strs.add(plan_str)
                        break

    return extended_plans


def gen_mapping(plans, mode):
    mapping = {}
    if mode == 'cost':
        for plan in plans:
            index = len(plan['seq'])
            if index not in mapping:
                mapping[index] = []
            mapping[index].append(plan)
    else:
        for plan in plans:
            if len(plan['seq']) != 1:
                continue
            index = len(str(gen_label(plan, 'card')))
            if index not in mapping:
                mapping[index] = []
            mapping[index].append(plan)
    return mapping


def save_model(model, path):
    torch.save({'model_state_dict': model.state_dict()}, path)


def get_weights(order, values, mode):
    weights = torch.full(values.shape, 1.0, dtype=torch.float)
    if mode == 'card':
        # 1.2 / 25.0
        weights[order == 1 - label_smoothing_factor] = 1.2 if dataset_name != "stack" else 50.0
    return weights


def organize_plans_for_cost(plans):
    # 去除单表计划
    plans = [_ for _ in plans if len(_['seq']) > 1]
    # 提前recover plan tree
    mapping = {}
    for plan in plans:
        layer_dict = defaultdict(list)
        recover_plan_tree(plan['seq'], 0, layer_dict, 0)        
        label = 1
        for level, node_indices in layer_dict.items():
            nodes = [plan['seq'][i] for i in node_indices]
            nodes = [_ for _ in nodes if _['node_type'] in ['Inner']]
            max_card = max(node['cardinality'] for node in nodes) if len(nodes) > 0 else 0
            label += max_card
        plan['label'] = label
        if len(str(label)) not in mapping:
            mapping[len(str(label))] = 0
        mapping[len(str(label))] += 1
    return plans


def test_epoch(epoch, model, dataloader, mode, is_eval=False):
    loss_total, batch_num = 0, 0
    f1 = MulticlassFBetaScore(beta=2.0, num_classes=2, average=None).to(device)
    precision = MulticlassPrecision(num_classes=2, average=None).to(device)
    recall = MulticlassRecall(num_classes=2, average=None).to(device)

    if is_eval:
        print("********eval********")
    for batch_idx, batch_data in enumerate(dataloader):
        model.eval()
        with torch.no_grad():
            batch_start = time.time()
            batch_1, batch_2, order, _1, _2 = batch_data
            score = model(batch_1, batch_2)

            # weights = get_weights(order, _1, mode)
            loss_function = nn.BCELoss(reduction='none')
            loss = loss_function(score, order.to(device))

            for _1, _2, _score, _loss in zip(_1, _2, score.tolist(), loss.tolist()):
                print(f"{_1} {_2} {_score} {_loss}")

            loss = loss.mean()
            loss_total += loss.item()

            comparison_result = (score >= 0.5).int()
            abs_order = torch.where(order == 1 - label_smoothing_factor, torch.tensor(1), torch.tensor(0)).to(device)
            f1.update(comparison_result, abs_order)
            precision.update(comparison_result, abs_order)
            recall.update(comparison_result, abs_order)

            print(f"eval batch {batch_idx}, loss = {loss.item()}, "
                  f"forward time = {(time.time() - batch_start) * 1000} ms")

        batch_num += 1

    eval_loss = loss_total / batch_num
    f1_value, precision_value, recall_value = f1.compute(), precision.compute(), recall.compute()
    print(f"Epoch {epoch}, evaluation loss: {eval_loss}, "
          f"f1: {f1_value}, precision: {precision_value}, recall: {recall_value}")
    criteria = -f1_value[1]
    return criteria


def train(num_epochs, parameters, train_directory, test_directory, model_path, mode, eval_directory=None):
    global epoch_id
    hidden_dim = 48
    hid_dim = 48
    lr = 0.0001 if dataset_name == "imdb_10x" else 0.0002
    wd = 0.1 if dataset_name == "stack" else 0.1
    model = DualComparator(parameters, hidden_dim, hid_dim, mode).to(device)
    print(f"num of params = {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr if mode == 'card' else lr, weight_decay=wd)

    lambda1 = lambda epoch: 1 if epoch < 10 else (0.5 if epoch < 15 else 0.25)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    training_plans = load_plans(train_directory, parameters)
    training_plans_augmented = training_plans
    if mode == 'cost':
        training_plans_augmented = organize_plans_for_cost(training_plans_augmented)
    for id, _ in enumerate(training_plans_augmented):
        _['id'] = id
    training_mapping = gen_mapping(training_plans_augmented, mode)
    training_dataset = PlanDataset(training_plans_augmented)
    training_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=32, num_workers=0,
                                     collate_fn=lambda batch: custom_collate(batch, parameters, training_mapping, mode))

    test_plans = load_plans(test_directory, parameters)
    if mode == 'cost':
        test_plans = organize_plans_for_cost(test_plans)
        test_mapping = gen_mapping(test_plans, mode)
    else:
        test_plans = [_ for _ in test_plans if _['seq'][0]['cardinality'] < 1e5 or _['seq'][0]['cardinality'] > 1e6]
        test_mapping = gen_mapping(training_plans, mode)
    test_plans = test_plans * 3
    for id, _ in enumerate(test_plans):
        _['id'] = id
    test_dataset = PlanDataset(test_plans)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=64, num_workers=0,
                                 collate_fn=lambda batch: custom_collate(batch, parameters, test_mapping, mode,
                                                                         type='test'))

    if eval_directory:
        eval_plans = load_plans(eval_directory, parameters)
        if mode == 'cost':
            eval_plans = organize_plans_for_cost(eval_plans)
            mapping = gen_mapping(eval_plans, mode)
        else:
            eval_plans = [_ for _ in eval_plans if _['seq'][0]['cardinality'] < 1e5 or _['seq'][0]['cardinality'] > 1e6]
            mapping = test_mapping
        eval_plans = eval_plans * 3
        for id, _ in enumerate(eval_plans):
            _['id'] = id
        eval_dataset = PlanDataset(eval_plans)
        eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=128, num_workers=0,
                                     collate_fn=lambda batch: custom_collate(batch, parameters, mapping,
                                                                             mode, type='eval'))

    start = time.time()
    best_eval_loss = 1e8
    for epoch in range(num_epochs):
        epoch_id = epoch
        loss_total = 0.
        batch_num = 0
        for batch_idx, batch_data in enumerate(training_dataloader):
            batch_start = time.time()
            model.train()
            optimizer.zero_grad()
            batch_1, batch_2, order, _1, _2 = batch_data
            score = model(batch_1, batch_2)

            weights = get_weights(order, _1, mode)
            loss_function = nn.BCELoss(reduction='none', weight=weights.to(device))

            loss = loss_function(score, order.to(device)).mean()
            loss.backward()
            optimizer.step()

            print(f"train batch {batch_idx}, loss = {loss.item()}, "
                  f"forward time = {(time.time() - batch_start) * 1000} ms")
            loss_total += loss.item()

            batch_num += 1

        print(f"Epoch {epoch}, training loss: {loss_total / batch_num}")

        eval_loss = test_epoch(epoch, model, test_dataloader, mode)
        if eval_loss < best_eval_loss:
            print(f"In epoch {epoch}, Best evaluation loss update to: {eval_loss}")
            best_eval_loss = eval_loss
        save_model(model, model_path)

        if eval_directory is not None:
            test_epoch(epoch, model, eval_dataloader, mode, is_eval=True)

        lr_scheduler.step()

    end = time.time()
    print(end - start)
    return model
