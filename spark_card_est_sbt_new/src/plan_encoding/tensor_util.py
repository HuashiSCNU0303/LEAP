from src.plan_encoding.encoding_plans import *
import math


def pad_batch_2D(batch, padding_value=0, dtype=np.float32):
    max_2nd_D = max([len(x) for x in batch])

    if padding_value == 0:
        tensor = np.zeros((len(batch), max_2nd_D), dtype=dtype)
    else:
        tensor = np.full((len(batch), max_2nd_D), fill_value=padding_value, dtype=dtype)

    for i, row in enumerate(batch):
        tensor[i, :len(row)] = row

    return tensor


def pad_batch_3D(batch):
    max_2nd_D = max(x.shape[0] for x in batch)
    max_3rd_D = max(x.shape[1] for x in batch)

    tensor = np.zeros((len(batch), max_2nd_D, max_3rd_D), dtype=np.float32)

    for dim_1, _1 in enumerate(batch):
        rows, cols = _1.shape
        tensor[dim_1, :rows, :cols] = _1

    return tensor


def pad_batch_4D(batch):
    max_2nd_D = max(x.shape[0] for x in batch)
    max_3rd_D = max(x.shape[1] for x in batch)
    max_4th_D = max(x.shape[2] for x in batch)

    tensor = np.zeros((len(batch), max_2nd_D, max_3rd_D, max_4th_D), dtype=np.float32)

    for dim_1, _1 in enumerate(batch):
        dim_2, dim_3, dim_4 = _1.shape
        tensor[dim_1, :dim_2, :dim_3, :dim_4] = _1

    return tensor


def normalize_label(labels, mini, maxi):
    labels = np.where(labels >= 1, np.log(labels), 0)
    labels_norm = (labels - mini) / (maxi - mini)
    labels_norm = np.clip(labels_norm, 0, 1)
    return labels_norm

