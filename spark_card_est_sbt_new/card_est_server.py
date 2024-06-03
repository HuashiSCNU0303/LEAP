import grpc
import time
import argparse
from concurrent import futures
from comparator_service import comparator_pb2, comparator_pb2_grpc

from src.dual_task.model import DualComparator
from src.plan_encoding.encoding_histogram import add_histograms
from src.plan_encoding.encoding_plans import encode_plan_job
from collections import defaultdict
from card_est import custom_collate
from config import *
import joblib
import pickle
import numpy as np
import os
import json
import torch
import bisect


class ComparatorService(comparator_pb2_grpc.ComparatorServiceServicer):

    def __init__(self, paths):
        self.parameters = pickle.load(open(parameters_path, "rb"))

        self.cost_model = self.load_model(paths['cost_model_path'], "cost")
        self.card_model = self.load_model(paths['card_model_path'], "card")
        self.historical_plan_window = self.load_historical_plans(paths['training_data_path'], paths['historical_window_path'])
        self.card_keys = sorted(self.historical_plan_window.keys())
        self.c = paths['c']
        self.epsilon = paths['epsilon']
        print(f"We have {len(self.card_keys)} historical plans, c: {self.c}, epsilon: {self.epsilon}, paths: {paths}")

    def load_model(self, path, mode):
        if not os.path.exists(path):
            return None
        hidden_dim = 48
        hid_dim = 48
        model = DualComparator(self.parameters, hidden_dim, hid_dim, mode).to(device)

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def extend_plans(self, plans):
        plan_strs = set()
        extended_plans = []
        for plan in plans:
            plan = plan['seq']

            for i in range(len(plan)):
                if plan[i]['node_type'] == 'Filter':
                    single_table_str = json.dumps(plan[i])
                    if single_table_str not in plan_strs:
                        extended_plans.append({'seq': [plan[i]]})
                        plan_strs.add(single_table_str)
        return extended_plans

    def gen_batch_representation(self, plan_batch, model):
        encoded_plan_batch = []
        for plan in plan_batch:
            filter_conds, tables, cards, num_filter_conds, operators = encode_plan_job(plan['seq'], self.parameters)
            encoded_plan_batch.append({
                'filter_conds': filter_conds,
                'tables': tables,
                'cards': cards,
                'num_filter_conds': num_filter_conds,
                'operators': operators
            })
        batch = custom_collate(encoded_plan_batch, self.parameters)
        # batch_size × result_dim
        start = time.time()
        model.eval()
        with torch.no_grad():
            # 转换为numpy数组
            representation_batch = model.representation(batch).cpu().numpy()
        print(f"gen_representation() time = {(time.time() - start) * 1000} ms")
        return representation_batch

    def load_historical_plans(self, plan_path, embedding_path):
        if os.path.exists(embedding_path):
            return joblib.load(open(embedding_path, "rb"))

        if not os.path.exists(plan_path):
            return None

        with open(plan_path, 'r') as f:
            plans = [json.loads(seq) for seq in f.readlines()]
        extended_plans = self.extend_plans(plans)
        window = defaultdict(list)
        batch_size = 64
        for i in range(0, len(extended_plans), batch_size):
            plan_batch = extended_plans[i: i + batch_size]
            representation_batch = self.gen_batch_representation(plan_batch, self.card_model)
            for plan, embedding in zip(plan_batch, representation_batch):
                card = plan['seq'][0]['cardinality']
                window[card].append(embedding)
            print(f"Processed {i} historical plans...")
        joblib.dump(window, open(embedding_path, "wb"))
        return window

    def transform_and_encode_plans(self, plan_strs, model):
        plans = np.empty((len(plan_strs), model.result_dim), dtype=np.float32)
        batch_size = 32
        for i in range(0, len(plan_strs), batch_size):
            plan_str_batch = plan_strs[i: i + batch_size]
            plan_batch = [{'seq': add_histograms(json.loads(plan_str)['seq'], self.parameters.statistics)}
                          for plan_str in plan_str_batch]
            start = time.time()
            representation_batch = self.gen_batch_representation(plan_batch, model)
            plans[i: i + batch_size] = representation_batch
            print(f"transform_and_encode_plans(), single batch time: {(time.time() - start) * 1000} ms")
        return plans

    def top_k_closest(self, candidate_plans, plan_1, k):
        candidate_plans_array = np.array([plan for _, plan in candidate_plans])
        distances = np.linalg.norm(candidate_plans_array - plan_1.reshape(1, -1), axis=1)
        top_k_indices = np.argsort(distances)[:k]
        top_k_elements = np.array([candidate_plans[idx][1] for idx in top_k_indices])
        top_k_keys = np.array([candidate_plans[idx][0] for idx in top_k_indices], dtype=np.float32)

        return top_k_elements, top_k_keys

    def CompareCost(self, request, context):
        if request.plans[0] == "test":
            return comparator_pb2.DataResponse(result=[False])

        start_time = time.time()
        plans = self.transform_and_encode_plans([_ for _ in request.plans] + [request.pivot], self.cost_model)
        print(f"transform_and_encode_plans() time: {(time.time() - start_time) * 1000} ms")

        batch_size = 32
        plan_embeddings = plans[:-1]  # 已经是numpy了
        pivot_embedding = plans[-1]
        total_cost_compare_result = []
        for start_index in range(0, len(plan_embeddings), batch_size):
            sub_plans_2 = torch.from_numpy(plan_embeddings[start_index: start_index + batch_size]).to(device)
            sub_plans_1 = torch.from_numpy(np.tile(pivot_embedding, (len(sub_plans_2), 1))).to(device)
            self.cost_model.eval()
            with torch.no_grad():
                cost_compare_scores = self.cost_model.compare_cost(sub_plans_2, sub_plans_1)
            total_cost_compare_result += (cost_compare_scores < 0.5).bool().tolist()
        print(f"compareCost() all time: {(time.time() - start_time) * 1000} ms")
        print("————————————————")

        return comparator_pb2.DataResponse(result=total_cost_compare_result)

    def CanBeBroadcast(self, request, context):
        start_time = time.time()
        # 已经是embedding形式了（依然是numpy数组）
        plans_1 = self.transform_and_encode_plans(request.data, self.card_model)
        compare_results = []
        key_range_ratio = self.epsilon
        num_candidates = self.c
        for index, (plan_1, threshold) in enumerate(zip(plans_1, request.threshold)):
            single_table_start = time.time()
            # 根据threshold，从历史里找一个合适的计划
            # bisect_left()返回 >= 的第一个下标，那应该 - 1
            idx = bisect.bisect_left(self.card_keys, threshold) - 1
            closest_key = self.card_keys[idx]

            # 从[key_min, key_max]范围内的plan做收集
            key_min, key_max = (1 - key_range_ratio) * closest_key, (1 + key_range_ratio) * closest_key
            candidate_plans = [(key, plan) for key in self.card_keys if key_min < key < key_max for plan in self.historical_plan_window[key]]

            # 选距离最小的num_candidates个
            selection_start = time.time()
            candidate_plans, top_k_keys = self.top_k_closest(candidate_plans, plan_1, num_candidates)
            selection_time = (time.time() - selection_start) * 1000

            sub_plans_1 = torch.from_numpy(np.tile(plan_1, (len(candidate_plans), 1))).to(device)
            sub_plans_2 = torch.from_numpy(candidate_plans).to(device)
            self.card_model.eval()
            with torch.no_grad():
                card_compare_scores = self.card_model.compare_card(sub_plans_1, sub_plans_2)
            # soft voting
            avg_score = sum(card_compare_scores.cpu().tolist()) / len(candidate_plans)
            final_result = avg_score < 0.5
            compare_results.append(final_result)
            print(f"threshold = {threshold}, selected_key = {closest_key}, avg = {avg_score}, score = {card_compare_scores}, "
                  f"final_result = {final_result}, single table selection time = {selection_time}, "
                  f"all time = {(time.time() - single_table_start) * 1000}")

        print(f"canBeBroadcast() all time: {(time.time() - start_time) * 1000} ms")
        print("————————————————")
        return comparator_pb2.DataResponse(result=compare_results)

    def get_top_k(self, indices, plans, low, high, k):
        def partition(p_low, p_high):
            pivot_embedding = plans[indices[p_high]]
            total_compare_results = []
            cand_embeddings = plans[indices[p_low: p_high]]
            batch_size = 32
            for start_index in range(0, len(cand_embeddings), batch_size):
                sub_plans_2 = torch.from_numpy(cand_embeddings[start_index: start_index + batch_size]).to(device)
                sub_plans_1 = torch.from_numpy(np.tile(pivot_embedding, (len(sub_plans_2), 1))).to(device)
                # 全部和pivot比
                self.cost_model.eval()
                with torch.no_grad():
                    cost_compare_scores = self.cost_model.compare_cost(sub_plans_2, sub_plans_1)
                cost_compare_result = (cost_compare_scores < 0.5).bool().tolist()
                total_compare_results += cost_compare_result
            print(f"total_compare_results = {total_compare_results}")
            i, j = p_low, p_high - 1
            while i <= j:
                if total_compare_results[i - p_low]:
                    i += 1
                elif not total_compare_results[j - p_low]:
                    j -= 1
                else:
                    temp = indices[i]
                    indices[i] = indices[j]
                    indices[j] = temp
                    i += 1
                    j -= 1
            temp = indices[i]
            indices[i] = indices[p_high]
            indices[p_high] = temp
            return i

        if low < high:
            i = partition(low, high)
            cnt = i - low + 1
            if cnt < k:
                # 在右边找第k - cnt个
                self.get_top_k(indices, plans, i + 1, high, k - cnt)
            elif cnt > k:
                # 在左边找第k个
                self.get_top_k(indices, plans, low, i - 1, k)

    def GetTopKPlans(self, request, context):
        start_time = time.time()
        # 已经编码好，变成numpy数组了
        plans = self.transform_and_encode_plans(request.plans, self.cost_model)
        k = int(request.k)
        # 返回top-k对应的索引
        indices = list(range(len(plans)))
        self.get_top_k(indices, plans, 0, len(plans) - 1, k)
        top_k_indices = indices[:k]
        print(f"top_k_indices = {top_k_indices}, k = {k}, getTopKPlans() all time: {(time.time() - start_time) * 1000} ms")
        print("————————————————")

        return comparator_pb2.TopKResponse(result=top_k_indices)


def serve(port, paths):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    comparator_pb2_grpc.add_ComparatorServiceServicer_to_server(ComparatorService(paths), server)
    server.add_insecure_port(f'localhost:{port}')
    print(f"Init finished. You can estimate cardinality via localhost:{port}...")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9009, help='Port for the cardinality estimation RPC server.')
    parser.add_argument('--c', type=int, default=16)
    parser.add_argument('--epsilon', type=float, default=0.5)
    opt = parser.parse_args()
    paths = {
        "cost_model_path": cost_model_path,
        "card_model_path": card_model_path,
        "training_data_path": training_data_path,
        "historical_window_path": historical_window_path,
        "c": opt.c,
        "epsilon": opt.epsilon
    }

    serve(opt.port, paths)
