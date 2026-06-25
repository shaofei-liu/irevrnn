#include <torch/extension.h>
#include <vector>
#include <stdio.h>

std::vector<torch::Tensor> irevrnn_cuda_forward(
    torch::Tensor z,
    torch::Tensor h_0,
    torch::Tensor c_0,
    torch::Tensor ind_weights,
    torch::Tensor hidden_weights,
    torch::Tensor cell_weights,
    int res_len,
    int ind_act_typ,
    int res_act_typ);

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
    int res_act_typ);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// -- irevrnn on tensor

std::vector<torch::Tensor> irevrnn_forward(
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
    CHECK_INPUT(z);
    CHECK_INPUT(h_0);
    CHECK_INPUT(c_0);
    CHECK_INPUT(ind_weights);
    CHECK_INPUT(hidden_weights);
    CHECK_INPUT(cell_weights);

    return irevrnn_cuda_forward(z, h_0, c_0, ind_weights, hidden_weights,
        cell_weights, res_len, ind_act_typ, res_act_typ);
}


std::vector<torch::Tensor> irevrnn_backward(
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
    CHECK_INPUT(dout);
    CHECK_INPUT(dh_last);
    CHECK_INPUT(dc_last);
    CHECK_INPUT(z);
    CHECK_INPUT(rev_h);
    CHECK_INPUT(rev_c);
    CHECK_INPUT(ind_weights);
    CHECK_INPUT(hidden_weights);
    CHECK_INPUT(cell_weights); 
    return irevrnn_cuda_backward(dout, dh_last, dc_last, z, rev_h, rev_c,
        ind_weights, hidden_weights, cell_weights, res_len, ind_act_typ, res_act_typ);
}

// pybind magic

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fwd", &irevrnn_forward, "irevrnn forward");
    m.def("bwd", &irevrnn_backward, "irevrnn backward");
}
