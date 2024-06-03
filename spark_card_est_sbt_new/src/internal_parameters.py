
class Parameters:
    def __init__(self, tables_id, columns_id, logical_ops_id, compare_ops_id, join_compare_ops_id,
                 bool_ops_id, col_types, statistics, card_label_min, card_label_max):
        self.tables_id = tables_id
        self.columns_id = columns_id
        self.logical_ops_id = logical_ops_id
        self.compare_ops_id = compare_ops_id
        self.bool_ops_id = bool_ops_id
        self.join_compare_ops_id = join_compare_ops_id
        self.column_total_num = len(columns_id)
        self.table_total_num = len(tables_id)
        self.logical_op_total_num = len(logical_ops_id)
        self.bool_ops_total_num = len(bool_ops_id)
        self.compare_ops_total_num = len(compare_ops_id)
        self.join_compare_ops_total_num = len(join_compare_ops_id)
        self.histogram_bin_count = 64
        self.string_encoding_dim = 64
        self.condition_op_dim = self.bool_ops_total_num + self.compare_ops_total_num + self.column_total_num \
                                + 1 + self.string_encoding_dim \
                                + 2 * self.histogram_bin_count + 4
        self.join_condition_op_dim = self.bool_ops_total_num + 2 * (self.column_total_num + 3 + self.histogram_bin_count) + self.join_compare_ops_total_num
        self.col_dtypes = col_types
        self.statistics = statistics
        self.card_label_min = card_label_min
        self.card_label_max = card_label_max