import torch
from torch import Tensor
from .common import act_func


# Python variant functions

def irevrnn_py_fwd(z: Tensor, h_0: Tensor, c_0: Tensor, ind_weights: Tensor, hidden_weights: Tensor,
                           cell_weights: Tensor, res_len: int, ind_act_typ: int, res_act_typ: int):
    rev_h = torch.zeros_like(z)
    rev_c = torch.zeros_like(z)
    h_t = h_0
    c_t = c_0
    out = torch.zeros_like(z)
    seq_len = z.size(0)
    ind_act = act_func[ind_act_typ][0]
    res_act = act_func[res_act_typ][0]
    for t in range(seq_len):
        h_t = ind_weights * h_t
        for i in range(res_len):
            c_t = c_t + hidden_weights[i] * res_act(h_t)
            h_t = h_t + cell_weights[i] * res_act(c_t)
        rev_h[t] = h_t
        rev_c[t] = c_t
        h_t = ind_act(z[t] + h_t)
        out[t] = h_t
    return out, rev_h, rev_c


def irevrnn_py_bwd(dout: Tensor, dh_last: Tensor, dc_last: Tensor, z: Tensor, rev_h: Tensor, rev_c: Tensor,
                           ind_weights: Tensor, hidden_weights: Tensor, cell_weights: Tensor, res_len: int,
                           ind_act_typ: int, res_act_typ: int):
    dz = torch.zeros_like(z)
    dh_t = torch.zeros_like(rev_h[0])
    dc_t = torch.zeros_like(rev_c[0])
    dh_t += dh_last
    dc_t += dc_last
    dind_weights = torch.zeros_like(ind_weights)
    dhidden_weights = torch.zeros_like(hidden_weights)
    dcell_weights = torch.zeros_like(cell_weights)
    res_act = act_func[res_act_typ][0]
    ind_back = act_func[ind_act_typ][1]
    res_back = act_func[res_act_typ][1]
    seq_len = z.size(0)
    for t in range(seq_len - 1, -1, -1):
        dh_t = dh_t + dout[t]
        h_t = rev_h[t]
        c_t = rev_c[t]
        dz[t] = dh_t * ind_back(z[t] + h_t)
        dh_t = dz[t]
        for i in range(res_len - 1, -1, -1):
            dcell_weights[i] = dcell_weights[i] + torch.sum(dh_t * res_act(c_t), 0)
            dc_t = dc_t + dh_t * cell_weights[i] * res_back(c_t)
            h_t = h_t - cell_weights[i] * res_act(c_t)
            dhidden_weights[i] = dhidden_weights[i] + torch.sum(dc_t * res_act(h_t), 0)
            dh_t = dh_t + dc_t * hidden_weights[i] * res_back(h_t)
            c_t = c_t - hidden_weights[i] * res_act(h_t)
        h_t = h_t / ind_weights
        dind_weights = dind_weights + torch.sum(dh_t * h_t, 0)
        dh_t = dh_t * ind_weights
    return dz, dh_t, dc_t, dind_weights, dhidden_weights, dcell_weights
