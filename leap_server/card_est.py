from src.plan_encoding.tensor_util import pad_batch_2D, pad_batch_3D, pad_batch_4D, normalize_label


def make_data_job(plans, parameters):
    card_batch = []
    filter_conds_batch = []
    tables_batch = []
    num_filter_conds_batch = []
    operators_batch = []

    for plan in plans:
        tables_batch.append(plan['tables'])
        filter_conds_batch.append(plan['filter_conds'])
        num_filter_conds_batch.append(plan['num_filter_conds'])
        operators_batch.append(plan['operators'])

        cards = normalize_label(plan['cards'], parameters.card_label_min, parameters.card_label_max)
        card_batch.append(cards)

    return card_batch, filter_conds_batch, tables_batch, num_filter_conds_batch, operators_batch


def custom_collate(batch, parameters):
    batch_1 = list(batch)
    cards_batch, filter_conds_batch, tables_batch, num_filter_conds_batch, operators_batch = make_data_job(batch_1, parameters)

    return {
        'num_steps_batch': [len(_) for _ in cards_batch],
        'num_filter_conds_batch': pad_batch_2D(num_filter_conds_batch, padding_value=1),
        # batch_size × num_steps × num_conditions × condition_dim
        'filter_conds_batch': pad_batch_4D(filter_conds_batch),
        # batch_size × num_steps × data_dim
        'tables_batch': pad_batch_3D(tables_batch),
        'cards_batch': pad_batch_3D(cards_batch),
        'operators_batch': pad_batch_3D(operators_batch)
    }
