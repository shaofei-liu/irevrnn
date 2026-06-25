#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include "cuda_activation.cuh"

/* NOTE:
replace "torch::PackedTensorAccessor32" to "torch::PackedTensorAccessor64" and
".packed_accessor32" to ".packed_accessor64" if index overflow might occur.
Might be slower though. See:
https://pytorch.org/cppdocs/notes/tensor_basics.html
*/

namespace {

    // forward kernel for the external independently RNN

    template <typename scalar_t, class ind_act, class res_act>
    __launch_bounds__(1024)
        __global__ void irevrnn_forward_kernel(
            torch::PackedTensorAccessor32<scalar_t, 3> out,
            torch::PackedTensorAccessor32<scalar_t, 3> z,
            torch::PackedTensorAccessor32<scalar_t, 2> h_0,
            torch::PackedTensorAccessor32<scalar_t, 2> c_0,
            torch::PackedTensorAccessor32<scalar_t, 3> rev_h,
            torch::PackedTensorAccessor32<scalar_t, 3> rev_c,
            torch::PackedTensorAccessor32<scalar_t, 2> ind_weights,
            torch::PackedTensorAccessor32<scalar_t, 2> hidden_weights,
            torch::PackedTensorAccessor32<scalar_t, 2> cell_weights,
            int seq_len,
            int res_len)
    {
        const auto batch_size = z.size(1);
        const auto hidden_size = z.size(2);
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        // return if out of range
        if (c >= batch_size * hidden_size) return;
        // get current batch and state index numbers
        const int bs = c / hidden_size;
        const int idx = c % hidden_size;
        // run irevrnn for this thread
        scalar_t h_t = h_0[bs][idx];
        scalar_t c_t = c_0[bs][idx];
        for (int t = 0; t < seq_len; ++t)
        {
            h_t = ind_weights[0][idx] * h_t;
            for (int i = 0; i < res_len; ++i)
            {
                c_t = c_t + hidden_weights[i][idx] * res_act::forward(h_t);
                h_t = h_t + cell_weights[i][idx] * res_act::forward(c_t);
            }
            rev_h[t][bs][idx] = h_t;
            rev_c[t][bs][idx] = c_t;
            h_t = ind_act::forward(z[t][bs][idx] + h_t);
            out[t][bs][idx] = h_t;
        }
    }


    // backward kernel for the external independently RNN

    template <typename scalar_t, class ind_act, class res_act>
    __launch_bounds__(1024)
        __global__ void irevrnn_backward_kernel(
            torch::PackedTensorAccessor32<scalar_t, 3> dout,
            torch::PackedTensorAccessor32<scalar_t, 3> z,
            torch::PackedTensorAccessor32<scalar_t, 3> rev_h,
            torch::PackedTensorAccessor32<scalar_t, 3> rev_c,
            torch::PackedTensorAccessor32<scalar_t, 2> ind_weights,
            torch::PackedTensorAccessor32<scalar_t, 2> hidden_weights,
            torch::PackedTensorAccessor32<scalar_t, 2> cell_weights,
            torch::PackedTensorAccessor32<scalar_t, 3> dz,
            torch::PackedTensorAccessor32<scalar_t, 2> dh_t,
            torch::PackedTensorAccessor32<scalar_t, 2> dc_t,
            torch::PackedTensorAccessor32<scalar_t, 3> dind_weights,
            torch::PackedTensorAccessor32<scalar_t, 3> dhidden_weights,
            torch::PackedTensorAccessor32<scalar_t, 3> dcell_weights,
            int seq_len,
            int res_len)
    {
        const auto batch_size = z.size(1);
        const auto hidden_size = z.size(2);
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        // return if out of range
        if (c >= batch_size * hidden_size) return;
        // get current batch and state index numbers
        const int bs = c / hidden_size;
        const int idx = c % hidden_size;
        // run irevrnn for this thread
        for (int t = seq_len - 1; t > -1; --t)
        {
            dh_t[bs][idx] = dh_t[bs][idx] + dout[t][bs][idx];
            scalar_t h_t = rev_h[t][bs][idx];
            scalar_t c_t = rev_c[t][bs][idx];
            dz[t][bs][idx] = dh_t[bs][idx] * ind_act::backward(z[t][bs][idx] + h_t);
            dh_t[bs][idx] = dz[t][bs][idx];
            for (int i = res_len - 1; i > -1; --i)
            {
                dcell_weights[bs][i][idx] = dcell_weights[bs][i][idx] + dh_t[bs][idx] * res_act::forward(c_t);
                dc_t[bs][idx] = dc_t[bs][idx] + dh_t[bs][idx] * cell_weights[i][idx] * res_act::backward(c_t);
                h_t = h_t - cell_weights[i][idx] * res_act::forward(c_t);
                dhidden_weights[bs][i][idx] = dhidden_weights[bs][i][idx] + dc_t[bs][idx] * res_act::forward(h_t);
                dh_t[bs][idx] = dh_t[bs][idx] + dc_t[bs][idx] * hidden_weights[i][idx] * res_act::backward(h_t);
                c_t = c_t - hidden_weights[i][idx] * res_act::forward(h_t);
            }
            h_t = h_t / ind_weights[0][idx];
            dind_weights[bs][0][idx] = dind_weights[bs][0][idx] + dh_t[bs][idx] * h_t;
            dh_t[bs][idx] = dh_t[bs][idx] * ind_weights[0][idx];
        }
    }

}// namespace

// cuda host functions for irevrnn


std::vector<torch::Tensor> irevrnn_cuda_forward(
    torch::Tensor z,
    torch::Tensor h_0,
    torch::Tensor c_0,
    torch::Tensor ind_weights,
    torch::Tensor hidden_weights,
    torch::Tensor cell_weights,
    int res_len,
    int ind_act_typ,
    int res_act_typ)
{
    const auto batch_size = z.size(1);
    const auto hidden_size = z.size(2);
    const int threads = 1024;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;
    auto rev_h = torch::zeros_like(z);
    auto rev_c = torch::zeros_like(z);
    auto out = torch::zeros_like(z);
    int seq_len = z.size(0);
    AT_DISPATCH_FLOATING_TYPES(z.type(), "irevrnn_forward", ([&] {
        DISPATCH_INDEPENDENTLY_ACTIVATION(ind_act_typ, ([&] {
            DISPATCH_RESIDUAL_ACTIVATION(res_act_typ, ([&] {
                irevrnn_forward_kernel<scalar_t, ind_act, res_act> << <blocks, threads >> > (
                    out.packed_accessor32<scalar_t, 3>(),
                    z.packed_accessor32<scalar_t, 3>(),
                    h_0.packed_accessor32<scalar_t, 2>(),
                    c_0.packed_accessor32<scalar_t, 2>(),
                    rev_h.packed_accessor32<scalar_t, 3>(),
                    rev_c.packed_accessor32<scalar_t, 3>(),
                    ind_weights.packed_accessor32<scalar_t, 2>(),
                    hidden_weights.packed_accessor32<scalar_t, 2>(),
                    cell_weights.packed_accessor32<scalar_t, 2>(),
                    seq_len,
                    res_len);
                }));
            }));
        }));
    return { out, rev_h, rev_c };
}


std::vector<torch::Tensor> irevrnn_cuda_backward(
    torch::Tensor dout,
    torch::Tensor dh_last,
    torch::Tensor dc_last,
    torch::Tensor z,
    torch::Tensor rev_h,
    torch::Tensor rev_c,
    torch::Tensor ind_weights,
    torch::Tensor hidden_weights,
    torch::Tensor cell_weights,
    int res_len,
    int ind_act_typ,
    int res_act_typ)
{
    const auto batch_size = z.size(1);
    const auto hidden_size = z.size(2);
    const int threads = 1024;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;
    auto dz = torch::zeros_like(z);
    auto dh_t = torch::zeros_like(rev_h[0]);
    auto dc_t = torch::zeros_like(rev_c[0]);
    dh_t += dh_last;
    dc_t += dc_last;
    auto dind_weights = torch::zeros_like(ind_weights).repeat({ batch_size, 1, 1 });
    auto dhidden_weights = torch::zeros_like(hidden_weights).repeat({ batch_size, 1, 1 });
    auto dcell_weights = torch::zeros_like(cell_weights).repeat({ batch_size, 1, 1 });
    int seq_len = z.size(0);
    AT_DISPATCH_FLOATING_TYPES(z.type(), "irevrnn_backward", ([&] {
        DISPATCH_INDEPENDENTLY_ACTIVATION(ind_act_typ, ([&] {
            DISPATCH_RESIDUAL_ACTIVATION(res_act_typ, ([&] {
                irevrnn_backward_kernel<scalar_t, ind_act, res_act> << <blocks, threads >> > (
                    dout.packed_accessor32<scalar_t, 3>(),
                    z.packed_accessor32<scalar_t, 3>(),
                    rev_h.packed_accessor32<scalar_t, 3>(),
                    rev_c.packed_accessor32<scalar_t, 3>(),
                    ind_weights.packed_accessor32<scalar_t, 2>(),
                    hidden_weights.packed_accessor32<scalar_t, 2>(),
                    cell_weights.packed_accessor32<scalar_t, 2>(),
                    dz.packed_accessor32<scalar_t, 3>(),
                    dh_t.packed_accessor32<scalar_t, 2>(),
                    dc_t.packed_accessor32<scalar_t, 2>(),
                    dind_weights.packed_accessor32<scalar_t, 3>(),
                    dhidden_weights.packed_accessor32<scalar_t, 3>(),
                    dcell_weights.packed_accessor32<scalar_t, 3>(),
                    seq_len,
                    res_len);
                }));
            }));
        }));
    auto dind_weights_sum = torch::sum(dind_weights, 0);
    auto dhidden_weights_sum = torch::sum(dhidden_weights, 0);
    auto dcell_weights_sum = torch::sum(dcell_weights, 0);
    return { dz, dh_t, dc_t, dind_weights_sum, dhidden_weights_sum, dcell_weights_sum };
}