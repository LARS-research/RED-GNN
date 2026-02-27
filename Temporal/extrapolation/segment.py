import time
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

def _segment_id2sparse_block_diag_matrix_coordinate(segment_ids):
    """
    segment_ids is a ascending 1d numpy array, dtype int, e.g. [0,0,0,1,2,2,3, ...]
    we want to create a sparse block digonal matrix from segment_ids,
    i-th block (square) has a shape (number_of_i_in_segment_ids, number_of_i_in_segment_ids)
    and each block is filled with 1
    e.g. [0,0,0,1,2,2] -->
    [[1,1,1,0,0,0],
     [1,1,1,0,0,0],
     [1,1,1,0,0,0],
     [0,0,0,1,0,0],
     [0,0,0,0,1,1],
     [0,0,0,0,1,1]]
    Attention!: But we don't return the matrix, we return the index of nonzero in this matrix
    in the form of a numpy array of shape 2 x N, first row is row index, second row is col index
    """
    mask = segment_ids[:-1] != segment_ids[1:]
    segment_start = np.concatenate([np.array([0]),
                                    np.arange(1, len(segment_ids))[mask],
                                    np.array([len(segment_ids)])])
    segment_len = np.diff(segment_start)

    row_idx = []
    col_idx = []
    shift = 0
    for i, slen in enumerate(segment_len):
        shift += i and segment_len[i - 1]
        col_idx.append(np.tile(np.arange(slen), slen) + shift)
        row_idx.append(np.repeat(np.arange(slen), slen) + shift)
    col_idx = np.concatenate(col_idx)
    row_idx = np.concatenate(row_idx)
    return np.stack([row_idx, col_idx], axis=0)


def segment_softmax_op(logits, segment_ids, tc=None):
    """
    logits is a 1d tensor of attention score (refer to DPMPN paper),
    i-th  node has attention score logits[i] which is in the subgraph developed for the query segment_ids[i]
    This function try to calculate the softmax of the nodes in the same subgraph

    :param logits: 1d Tensor
    :param segment_ids: id numpy.array eg_idx, sorted
    :return:
    softmax for logtis with same segment_id
    """
    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    len_logits = len(segment_ids)
    if tc:
        t_start = time.time()
    sparse_index_np = _segment_id2sparse_block_diag_matrix_coordinate(segment_ids)
    if tc:
        tc['model']['DP_attn_softmax_trans_matrix'] = time.time() - t_start
    sparse_index = torch.LongTensor(sparse_index_np)
    sparse_value = torch.ones(sparse_index_np.shape[1], dtype=torch.float)
    trans_matrix_sparse_th = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                      torch.Size([len_logits, len_logits])).to(device)
    softmax_den = torch.squeeze(torch.sparse.mm(trans_matrix_sparse_th, torch.exp(logits).unsqueeze(1)))
    logits_segment_softmax = torch.exp(logits) / softmax_den
    return logits_segment_softmax


def segment_sum(logits, segment_ids, keep_length=True):
    """

    :param logits:
    :param segment_ids:
    :param keep_length: if True, return a Tensor with the same length as logits
    out[i] is the sum of segments of segment_ids[i]
    else, return a Tensor with the length of segment_ids
    :return:
    """
    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    logits_len = len(segment_ids)
    num_segments = max(segment_ids) + 1

    # calculate summation of logits value for each group
    sparse_index = torch.LongTensor(np.stack([segment_ids, np.arange(logits_len)]))
    sparse_value = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                   torch.Size([num_segments, logits_len])).to(device)
    seg_sum = torch.sparse.mm(trans_matrix_sparse, logits.unsqueeze(1))
    if not keep_length:
        return seg_sum

    # repeat summation to have the same length as logits
    sparse_index2 = torch.LongTensor(np.stack([np.arange(logits_len), segment_ids]))
    sparse_value2 = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse2 = torch.sparse.FloatTensor(sparse_index2, sparse_value2,
                                                    torch.Size([logits_len, num_segments])).to(device)
    seg_sum_repeat = torch.sparse.mm(trans_matrix_sparse2, seg_sum)
    return torch.squeeze(seg_sum_repeat)


def segment_max(logits, segment_ids, keep_length=True):
    """

    :param logits:
    :param segment_ids:
    :param keep_length:
    if True, return a Tensor with the same length as logits
    out[i] is the sum of segments of segment_ids[i]
    else, return a Tensor with the length of segment_ids
    :return:
    1d Tensor
    """
    device = logits.get_device()
    n_logits = len(segment_ids)
    mask = segment_ids[1:] != segment_ids[:-1]
    seg_head_ids = np.concatenate([np.array([0]),
                                   np.arange(1, n_logits)[mask],
                                   np.array([n_logits])]).astype(np.int64)
    if keep_length:
        seg_max_ind = torch.cat([(torch.argmax(logits[torch.arange(head, tail).to(torch.int64).to(device)]) + torch.tensor([head]).to(torch.int64).to(device)).repeat(tail - head) for head, tail in zip(seg_head_ids[:-1], seg_head_ids[1:])])
    else:
        seg_max_ind = torch.cat([torch.argmax(logits[torch.arange(head, tail).to(torch.int64).to(device)]) + torch.tensor([head]).to(torch.int64).to(device) for head, tail in zip(seg_head_ids[:-1], seg_head_ids[1:])])
    return logits[seg_max_ind]


def segment_softmax_op_v2(logits, segment_ids, tc=None):
    """

    :param logits:
    :param segment_ids: numpy array, same length as logits, logits[i] belongs to segment segment_ids[i]
    logits in the same segment should aranged in a continuous block
    :param tc:
    :return:
    """
    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    if tc:
        t_start = time.time()

    logits_len = len(segment_ids)
    num_segments = max(segment_ids) + 1
    # numerical stable softmax
    logits = logits - segment_max(logits, segment_ids, keep_length=True)
    logits_exp = torch.exp(logits).unsqueeze(1)  # e^{logit} N x 1

    # calculate summation of exponential of logits value for each group
    sparse_index = torch.LongTensor(np.stack([segment_ids, np.arange(logits_len)]))
    sparse_value = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                   torch.Size([num_segments, logits_len])).to(device)
    softmax_den = torch.sparse.mm(trans_matrix_sparse, logits_exp)

    # repeat softmax denominator to have the same length as logits
    sparse_index2 = torch.LongTensor(np.stack([np.arange(logits_len), segment_ids]))
    sparse_value2 = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse2 = torch.sparse.FloatTensor(sparse_index2, sparse_value2,
                                                    torch.Size([logits_len, num_segments])).to(device)
    softmax_den_repeat = torch.sparse.mm(trans_matrix_sparse2, softmax_den)

    out = torch.squeeze(logits_exp / softmax_den_repeat)
    if tc:
        tc['model']['DP_attn_softmax_v2'] += time.time() - t_start
    return out


def segment_norm_l1_ordered(logits, segment_ids, tc=None):
    """
    segment_ids has to be ordered
    logits: Tensor 1d
    segment_ids: numpy.array 1d
    """
    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    len_logits = len(segment_ids)
    if tc:
        t_start = time.time()
    sparse_index_np = _segment_id2sparse_block_diag_matrix_coordinate(segment_ids)
    if tc:
        tc['model']['DP_attn_softmax_trans_matrix'] = time.time() - t_start
    sparse_index = torch.LongTensor(sparse_index_np)
    sparse_value = torch.ones(sparse_index_np.shape[1], dtype=torch.float)

    trans_matrix_sparse_th = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                      torch.Size([len_logits, len_logits])).to(device)

    norm_den = torch.squeeze(torch.sparse.mm(trans_matrix_sparse_th, logits.unsqueeze(1)))
    return logits / norm_den


def segment_norm_l1(logits, segment_ids):
    """
    segment_ids doesn't have to be ordered
    :param logits: Tensor
    :param segment_ids: 1-d numpy array
    :return:
    """
    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    N_segment = max(segment_ids) + 1
    # get denominator by multiplication logits with a matrix
    # get a 1-d tensor with a length of N_segment
    sparse_index = torch.LongTensor(np.vstack([segment_ids, np.arange(len(segment_ids))]))
    sparse_value = torch.ones(len(segment_ids), dtype=torch.float)
    trans_matrix_sparse_th = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                      torch.Size([N_segment, len(segment_ids)])).to(device)
    norm_den = torch.sparse.mm(trans_matrix_sparse_th, logits.unsqueeze(1))

    # copy denominator so that it has the same lenghth as the logits and the dominator lies in the same position
    # ie den[i] is the denominator for segment_ids[i]
    sparse_index = torch.LongTensor(np.vstack([np.arange(len(segment_ids)), segment_ids]))
    sparse_value = torch.ones(len(segment_ids), dtype=torch.float)
    trans_matrix_sparse_th = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                      torch.Size([len(segment_ids), N_segment])).to(device)
    den = torch.squeeze(torch.sparse.mm(trans_matrix_sparse_th, norm_den))
    res = logits / den
    res[res != res] = 0  # res != res inidcates where NaNs (0/0) are
    return res


def segment_norm_l1_part(logits, logits_ids, segment_ids, tc=None):
    """
    apply segment l1 norm on logits[start:]
    :param logits_ids: apply l1 norm on which logits
    :param logits:
    :param start:
    :param end:
    :param segment_idx: segment indicator for logits specified by logits_ids
    :param tc:
    :return:
    """
    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    len_logits = len(logits)
    if tc:
        t_start = time.time()

    mask = segment_ids[1:] != segment_ids[:-1]
    segment_start = np.concatenate([np.array([0]),
                                    np.arange(1, len(segment_ids))[mask],
                                    np.array([len(segment_ids)])])
    sparse_index_l = []
    for start, end in zip(segment_start[:-1], segment_start[1:]):
        sparse_index_l += [[x, y] for x, y in product(logits_ids[start:end], repeat=2)]

    sparse_index_np = np.array(sparse_index_l).T

    if tc:
        tc['model']['DP_attn_softmax_trans_matrix'] = time.time() - t_start

    sparse_index = torch.LongTensor(sparse_index_np)
    sparse_value = torch.ones(sparse_index_np.shape[1], dtype=torch.float)

    trans_matrix_sparse_th = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                      torch.Size([len_logits, len_logits])).to(device)

    norm_den = torch.squeeze(torch.sparse.mm(trans_matrix_sparse_th, logits.unsqueeze(1)))
    res = logits / norm_den
    res[res != res] = 0 # res != res inidcates where NaNs (0/0) are
    return res


def segment_topk(t, segment_idx, k, sorted=False):
    """
    compute topk along segments of a tensor
    params:
        t: Tensor, 1d, dtype=torch.float32
        segment_idx: numpy.array, 1d, dtype=numpy.int32, sorted
        k: k largest values
    return:
        values[i]: Tensor of topk of segment i
        indices[i]: numpy.array of position of topk elements of segment i in original Tensor t
    """
    mask = segment_idx[1:] != segment_idx[:-1]
    key_idx = np.concatenate([np.array([0], dtype=np.int32),
                              np.arange(1, len(segment_idx))[mask],
                              np.array([len(segment_idx)])])
    values = []
    indices = []
    for s, e in zip(key_idx[:-1], key_idx[1:]):
        if e - s < k:
            if sorted:
                sorted_value, sorted_indices = torch.sort(t[s:e], descending=True)
                values.append(sorted_value)
                indices.append(s + sorted_indices.cpu().numpy())
            else:
                values.append(t[s:e])
                indices.append(np.arange(s, e))
        else:
            segment_values, segment_indices = torch.topk(t[s:e], k, sorted=sorted)
            values.append(segment_values)
            indices.append(s + segment_indices.cpu().numpy())
    return values, indices


def segment_rank(t, entities, target_idx_l):
    """
    compute rank of ground truth (target_idx_l) in prediction according to score, i.e. t
    :param t: prediction score
    :param entities: 2-d numpy array, (segment_idx, entity_idx)
    :param target_idx_l: 1-d numpy array, (batch_size, )
    :return:
    """
    mask = entities[1:, 0] != entities[:-1, 0]
    key_idx = np.concatenate([np.array([0], dtype=np.int32),
                              np.arange(1, len(entities))[mask],
                              np.array([len(entities)])])
    rank = []
    found_mask = []
    for i, (s, e) in enumerate(zip(key_idx[:-1], key_idx[1:])):
        arg_target = np.nonzero(entities[s:e, 1] == target_idx_l[i])[0]
        if arg_target.size > 0:
            found_mask.append(True)
            rank.append(torch.sum(t[s:e] > t[s:e][torch.from_numpy(arg_target)]).item() + 1)
        else:
            found_mask.append(False)
            rank.append(1e9) # MINERVA set rank to +inf if not in path, we follow this scheme
    return np.array(rank), found_mask


def segment_rank_fil(t, entities, target_idx_l, sp2o, spt2o, queries_sub, queries_pre, queries_ts):
    """
    compute rank of ground truth (target_idx_l) in prediction according to score, i.e. t
    :param sp2o:
    :param t: prediction score
    :param entities: 2-d numpy array, (segment_idx, entity_idx)
    :param target_idx_l: 1-d numpy array, (batch_size, )
    :return:
    """
    mask = entities[1:, 0] != entities[:-1, 0]
    key_idx = np.concatenate([np.array([0], dtype=np.int32),
                              np.arange(1, len(entities))[mask],
                              np.array([len(entities)])])
    rank = []
    rank_fil = []
    rank_fil_t = []
    found_mask = []
    for i, (s, e) in enumerate(zip(key_idx[:-1], key_idx[1:])):
        arg_target = np.nonzero(entities[s:e, 1] == target_idx_l[i])[0]
        if arg_target.size > 0:
            found_mask.append(True)
            sub, pre, ts = queries_sub[i], queries_pre[i], queries_ts[i]
            obj_exist = sp2o[(sub, pre)]
            obj_exist_t = spt2o[(sub, pre, ts)]
            rank_pred_com1 = torch.sum(t[s:e] > t[s:e][torch.from_numpy(arg_target)]).item()
            rank_pred_com2 = torch.sum(t[s:e] == t[s:e][torch.from_numpy(arg_target)]).item()
            rank.append(rank_pred_com1 + ((rank_pred_com2 - 1) / 2) + 1)
            setdiff1d_array = np.setdiff1d(obj_exist, [target_idx_l[i]])
            fil = [ent not in setdiff1d_array for ent in entities[s:e, 1]]
            setdiff1d_array_t = np.setdiff1d(obj_exist_t, [target_idx_l[i]])
            fil_t = [ent not in setdiff1d_array_t for ent in entities[s:e, 1]]
            rank_pred_com1_fil = torch.sum(t[s:e][fil] > t[s:e][torch.from_numpy(arg_target)]).item()
            rank_pred_com2_fil = torch.sum(t[s:e][fil] == t[s:e][torch.from_numpy(arg_target)]).item()
            rank_fil.append(rank_pred_com1_fil + ((rank_pred_com2_fil - 1) / 2) + 1)
            rank_pred_com1_fil_t = torch.sum(t[s:e][fil_t] > t[s:e][torch.from_numpy(arg_target)]).item()
            rank_pred_com2_fil_t = torch.sum(t[s:e][fil_t] == t[s:e][torch.from_numpy(arg_target)]).item()
            rank_fil_t.append(rank_pred_com1_fil_t + ((rank_pred_com2_fil_t - 1) / 2) + 1)
        else:
            found_mask.append(False)
            rank.append(1e9)  # MINERVA set rank to +inf if not in path, we follow this scheme
            rank_fil.append(1e9)
    return np.array(rank), found_mask, np.array(rank_fil), np.array(rank_fil_t)
