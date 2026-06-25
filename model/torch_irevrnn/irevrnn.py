from typing import Optional
import torch
import sys
from torch import nn, Tensor
from torch.autograd import Function
from .irevrnn_py import irevrnn_py_fwd, irevrnn_py_bwd
from . import irevrnn_cpp, irevrnn_cuda


# # JIT loading of cpp and cuda modules
# from torch.utils.cpp_extension import load
# irevrnn_cpp = load(
#     name='irevrnn_cpp', sources=['irevrnn_cpp.cpp'], verbose=True)
# if torch.cuda.is_available():
#     irevrnn_cuda = load(name='irevrnn_cuda',
#                        sources=['irevrnn_cuda.cpp', 'irevrnn_cuda_kernel.cu'],
#                        verbose=True)
# else:
#     import warnings
#     warnings.warn("cuda acceleration not available for irevrnn")
#     irevrnn_cuda = None


# Python baselines


class _IRevRNNFuncPy(Function):
    @staticmethod
    def forward(ctx, z: Tensor, h_0: Tensor, c_0: Tensor, ind_weights: Tensor, hidden_weights: Tensor,
                cell_weights: Tensor, rev_len: int, ind_act_typ: int, res_act_typ: int):
        z = z.contiguous()
        out, rev_h, rev_c = irevrnn_py_fwd(z, h_0, c_0, ind_weights, hidden_weights,
                                                   cell_weights, rev_len, ind_act_typ, res_act_typ)
        ctx.save_for_backward(z, rev_h, rev_c, ind_weights, hidden_weights, cell_weights)
        ctx.saved_nontensors = (rev_len, ind_act_typ, res_act_typ)
        h_last = out[-1]
        c_last = rev_c[-1]
        return out, h_last, c_last

    @staticmethod
    def backward(ctx, dout: Tensor, dh_last: Tensor, dc_last: Tensor):
        dout = dout.contiguous()
        z, rev_h, rev_c, ind_weights, hidden_weights, cell_weights = ctx.saved_tensors
        rev_len, ind_act_typ, res_act_typ = ctx.saved_nontensors
        dz, dh_t, dc_t, dind_weights, dhidden_weights, dcell_weights = \
            irevrnn_py_bwd(dout, dh_last, dc_last, z, rev_h, rev_c, ind_weights,
                                   hidden_weights, cell_weights, rev_len, ind_act_typ, res_act_typ)
        return dz, dh_t, dc_t, dind_weights, dhidden_weights, dcell_weights, None, None, None


# C++ implementations


class _IRevRNNFuncCPP(Function):
    @staticmethod
    def forward(ctx, z: Tensor, h_0: Tensor, c_0: Tensor, ind_weights: Tensor, hidden_weights: Tensor,
                cell_weights: Tensor, rev_len: int, ind_act_typ: int, res_act_typ: int):
        z = z.contiguous()
        out, rev_h, rev_c = irevrnn_cpp.fwd(z, h_0, c_0, ind_weights, hidden_weights,
                                                    cell_weights, rev_len, ind_act_typ, res_act_typ)
        ctx.save_for_backward(z, rev_h, rev_c, ind_weights, hidden_weights, cell_weights)
        ctx.saved_nontensors = (rev_len, ind_act_typ, res_act_typ)
        h_last = out[-1]
        c_last = rev_c[-1]
        return out, h_last, c_last

    @staticmethod
    def backward(ctx, dout: Tensor, dh_last: Tensor, dc_last: Tensor):
        dout = dout.contiguous()
        z, rev_h, rev_c, ind_weights, hidden_weights, cell_weights = ctx.saved_tensors
        rev_len, ind_act_typ, res_act_typ = ctx.saved_nontensors
        dz, dh_t, dc_t, dind_weights, dhidden_weights, dcell_weights = \
            irevrnn_cpp.bwd(dout, dh_last, dc_last, z, rev_h, rev_c, ind_weights,
                                    hidden_weights, cell_weights, rev_len, ind_act_typ, res_act_typ)
        return dz, dh_t, dc_t, dind_weights, dhidden_weights, dcell_weights, None, None, None


# Custom CUDA implementations


class _IRevRNNFuncCUDA(Function):
    @staticmethod
    def forward(ctx, z: Tensor, h_0: Tensor, c_0: Tensor, ind_weights: Tensor, hidden_weights: Tensor,
                cell_weights: Tensor, rev_len: int, ind_act_typ: int, res_act_typ: int):
        z = z.contiguous()
        out, rev_h, rev_c = irevrnn_cuda.fwd(z, h_0, c_0, ind_weights, hidden_weights,
                                                     cell_weights, rev_len, ind_act_typ, res_act_typ)
        ctx.save_for_backward(z, rev_h, rev_c, ind_weights, hidden_weights, cell_weights)
        ctx.saved_nontensors = (rev_len, ind_act_typ, res_act_typ)
        h_last = out[-1]
        c_last = rev_c[-1]
        return out, h_last, c_last

    @staticmethod
    def backward(ctx, dout: Tensor, dh_last: Tensor, dc_last: Tensor):
        dout = dout.contiguous()
        z, rev_h, rev_c, ind_weights, hidden_weights, cell_weights = ctx.saved_tensors
        rev_len, ind_act_typ, res_act_typ = ctx.saved_nontensors
        dz, dh_t, dc_t, dind_weights, dhidden_weights, dcell_weights = \
            irevrnn_cuda.bwd(dout, dh_last, dc_last, z, rev_h, rev_c, ind_weights,
                                     hidden_weights, cell_weights, rev_len, ind_act_typ, res_act_typ)
        return dz, dh_t, dc_t, dind_weights, dhidden_weights, dcell_weights, None, None, None


# public API

# independent reversible RNN layer module

class IRevRNN(nn.Module):
    def __init__(self, hidden_size: int, rev_len: int, ind_act_typ_str: str = 'relu', res_act_typ_str: str = 'tanh'):
        super(IRevRNN, self).__init__()
        assert hidden_size > 0
        self.hidden_size = hidden_size
        self.rev_len = rev_len
        act_typ = {'relu': 0, 'tanh': 1, 'sigmoid': 2}
        assert ind_act_typ_str in act_typ
        assert res_act_typ_str in act_typ
        self.ind_act_typ = act_typ[ind_act_typ_str]
        self.res_act_typ = act_typ[res_act_typ_str]
        self.register_parameter(name="ind_weights", param=nn.Parameter(torch.empty(1, hidden_size),
                                                                       requires_grad=True))
        self.register_parameter(name="hidden_weights", param=nn.Parameter(torch.empty(self.rev_len, hidden_size),
                                                                          requires_grad=True))
        self.register_parameter(name="cell_weights", param=nn.Parameter(torch.empty(self.rev_len, hidden_size),
                                                                        requires_grad=True))
        if torch.cuda.is_available():
            self.ind_weights.cuda()
            self.hidden_weights.cuda()
            self.cell_weights.cuda()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.ind_weights, a=0, b=1)
        nn.init.normal_(self.hidden_weights, mean=0.0, std=1e-8)
        nn.init.normal_(self.cell_weights, mean=0.0, std=1e-8)

    def reset_states(self, z, h_0, c_0, flatten):
        if h_0 is None:
            h_0 = torch.zeros_like(z[0])
            h_0.normal_(mean=0, std=1e-4)
        if c_0 is None:
            c_0 = torch.zeros_like(z[0])
            c_0.normal_(mean=0, std=1e-4)
        if flatten:
            zshape = z.size()
            z = z.view(*zshape[:2], -1)
            h_0 = h_0.view(h_0.size(0), -1)
            c_0 = c_0.view(c_0.size(0), -1)
        else:
            zshape = None
        return z, zshape, h_0, c_0

    def forward(self, z: Tensor, h_0: Optional[Tensor] = None, c_0: Optional[Tensor] = None, flatten: bool = False):
        """nn.functional style implementation of irevrnn for tensor
        z (flattened): shape (seq_len, batch_size, hidden_size).
        If specified, h_0: shape (batch_size, hidden_size), c_0: shape (batch_size, hidden_size).
        If x_input has more than one feature dimensions, use flatten=True."""
        z, zshape, h_0, c_0 = self.reset_states(z, h_0, c_0, flatten)
        if z.is_cuda:
            out, h_last, c_last = _IRevRNNFuncCUDA.apply(z, h_0, c_0, self.ind_weights, self.hidden_weights,
                                                                self.cell_weights, self.rev_len, self.ind_act_typ,
                                                                self.res_act_typ)
        else:
            out, h_last, c_last = _IRevRNNFuncCPP.apply(z, h_0, c_0, self.ind_weights, self.hidden_weights,
                                                               self.cell_weights, self.rev_len, self.ind_act_typ,
                                                               self.res_act_typ)
        if flatten:
            return out.view(zshape)
        else:
            return out
