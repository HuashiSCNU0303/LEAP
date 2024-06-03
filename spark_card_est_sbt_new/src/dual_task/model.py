import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device
from torch.nn.utils.rnn import pack_padded_sequence


class ConditionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConditionLayer, self).__init__()

        self.conds_norm = nn.LayerNorm(input_dim)
        self.conds_mlp_1 = nn.Linear(input_dim, input_dim // 3)
        self.conds_lstm = nn.GRU(input_dim // 3, hidden_dim, batch_first=True)
        self.conds_mlp = nn.Linear(hidden_dim, output_dim)
        self.conds_dropout = nn.Dropout(0.3)

    def forward(self, conds, num_conds):
        batch_size, max_steps, max_condition, condition_dim = conds.shape
        conds = conds.view(batch_size * max_steps, max_condition, condition_dim)
        conds = self.conds_norm(conds)
        conds = F.relu(self.conds_mlp_1(conds))
        num_conds = num_conds.view(batch_size * max_steps)
        conds_packed = pack_padded_sequence(conds, num_conds, batch_first=True, enforce_sorted=False)
        out, hid = self.conds_lstm(conds_packed)
        out = hid[0]
        conds_output = self.conds_dropout(F.leaky_relu(self.conds_mlp(out)))
        conds_output = conds_output.view(batch_size, max_steps, -1)
        return conds_output


class Representation(nn.Module):
    def __init__(self, parameters, hidden_dim, result_dim):
        super(Representation, self).__init__()
        self.hidden_dim = hidden_dim

        self.filter_conds_layer = ConditionLayer(parameters.condition_op_dim, hidden_dim, result_dim)
        print(f"num of filter_conds_layer params = {sum(p.numel() for p in self.filter_conds_layer.parameters())}")

        self.representation_dim = parameters.logical_op_total_num + parameters.table_total_num + 1 + result_dim
        self.sequence_input_norm = nn.LayerNorm(self.representation_dim)
        self.sequence_mlp = nn.Linear(self.representation_dim, self.representation_dim // 2)
        self.sequence_lstm = nn.GRU(self.representation_dim // 2, result_dim, batch_first=True, bidirectional=True)
        self.sequence_post_mlp = nn.Linear(2 * result_dim, result_dim)
        self.sequence_dropout = nn.Dropout(0.3)

    def forward(self, batch):
        filter_conds = torch.from_numpy(batch['filter_conds_batch']).to(device)
        tables = torch.from_numpy(batch['tables_batch']).to(device)
        cards = torch.from_numpy(batch['cards_batch']).to(device)
        operators = torch.from_numpy(batch['operators_batch']).to(device)  # batch_size × num_nodes × operator_dim
        num_steps = batch['num_steps_batch']
        num_filter_conds = torch.from_numpy(batch['num_filter_conds_batch'])

        # batch_size × max_node × result_dim
        filter_conds_output = self.filter_conds_layer(filter_conds, num_filter_conds)
        conds_output = filter_conds_output

        out = torch.cat((operators, tables, cards, conds_output), 2)
        out = self.sequence_input_norm(out)
        out = F.relu(self.sequence_mlp(out))
        out_packed = pack_padded_sequence(out, num_steps, batch_first=True, enforce_sorted=False)
        lstm_out, hid = self.sequence_lstm(out_packed)

        out = torch.cat((hid[0], hid[1]), dim=1)
        out = self.sequence_dropout(F.relu(self.sequence_post_mlp(out)))
        return out


class Comparator(nn.Module):
    def __init__(self, result_dim):
        super(Comparator, self).__init__()

        self.distance_hidden = nn.Linear(result_dim, result_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.comparator = nn.Linear(result_dim // 2, 1)

    def forward(self, representation_1, representation_2):
        distance_hidden = self.distance_hidden(representation_1 - representation_2)
        hidden = self.dropout(distance_hidden)
        hidden = self.comparator(hidden)
        return torch.sigmoid(hidden).squeeze(1)


class DualComparator(nn.Module):
    def __init__(self, parameters, hidden_dim, result_dim, mode):
        super(DualComparator, self).__init__()

        self.result_dim = result_dim
        self.mode = mode
        self.representation = Representation(parameters, hidden_dim, result_dim)
        self.comparator = Comparator(result_dim)

    def forward(self, batch_1, batch_2):
        hid_1 = self.representation(batch_1)
        hid_2 = self.representation(batch_2)

        score_1_aligned, score_2_aligned = hid_1, hid_2
        prediction = self.comparator(score_1_aligned, score_2_aligned)

        return prediction

    def compare_card(self, batch_1, batch_2):
        return self.comparator(batch_1, batch_2)

    def compare_cost(self, batch_1, batch_2):
        return self.comparator(batch_1, batch_2)