#include <torch/extension.h>
#include <vector>
#include <utility>
#include <map>
#include <functional>

namespace {

    torch::Tensor d_relu(torch::Tensor t)
    {
        auto tt = t > 0;
        return tt.type_as(t);
    }

    torch::Tensor d_tanh(torch::Tensor t)
    {
        auto tt = torch::tanh(t);
        return 1 - tt * tt;
    }

    torch::Tensor d_sigmoid(torch::Tensor t)
    {
        auto tt = torch::sigmoid(t);
        return tt * (1 - tt);
    }

using PtWiseFunc = std::function<torch::Tensor(torch::Tensor)>;
const std::vector<std::pair<PtWiseFunc, PtWiseFunc>> act_func = {
    std::make_pair(torch::relu, d_relu),
    std::make_pair(torch::tanh, d_tanh),
    std::make_pair(torch::sigmoid, d_sigmoid)
};

} // namespace


// irevrnn forward function

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
    auto rev_h = torch::zeros_like(z);
    auto rev_c = torch::zeros_like(z);
    auto h_t = h_0;
    auto c_t = c_0;
    auto out = torch::zeros_like(z);
    int seq_len = z.size(0);
    for (int t = 0; t < seq_len; ++t)
    {
        h_t = ind_weights * h_t;
        for (int i = 0; i < res_len; ++i)
        {       
            c_t = c_t + hidden_weights[i] * act_func[res_act_typ].first(h_t);
            h_t = h_t + cell_weights[i] * act_func[res_act_typ].first(c_t);
        }
        rev_h[t] = h_t;
        rev_c[t] = c_t;
        h_t = act_func[ind_act_typ].first(z[t] + h_t);
        out[t] = h_t;
    }
    return { out, rev_h, rev_c };
}


// irevrnn backward function

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
    auto dz = torch::zeros_like(z);
    auto dh_t = torch::zeros_like(rev_h[0]);
    auto dc_t = torch::zeros_like(rev_c[0]);
    dh_t += dh_last;
    dc_t += dc_last;
    auto dind_weights = torch::zeros_like(ind_weights);
    auto dhidden_weights = torch::zeros_like(hidden_weights);
    auto dcell_weights = torch::zeros_like(cell_weights);
    int seq_len = z.size(0);
    for (int t = seq_len - 1; t > -1; --t)
    {
        dh_t = dh_t + dout[t];
        auto h_t = rev_h[t];
        auto c_t = rev_c[t];
        dz[t] = dh_t * act_func[ind_act_typ].second(z[t] + h_t);
        dh_t = dz[t];
        for (int i = res_len - 1; i > -1; --i)
        {
            dcell_weights[i] = dcell_weights[i] + torch::sum(dh_t * act_func[res_act_typ].first(c_t), 0);
            dc_t = dc_t + dh_t * cell_weights[i] * act_func[res_act_typ].second(c_t);
            h_t = h_t - cell_weights[i] * act_func[res_act_typ].first(c_t);
            dhidden_weights[i] = dhidden_weights[i] + torch::sum(dc_t * act_func[res_act_typ].first(h_t), 0);
            dh_t = dh_t + dc_t * hidden_weights[i] * act_func[res_act_typ].second(h_t);
            c_t = c_t - hidden_weights[i] * act_func[res_act_typ].first(h_t);
        }
        h_t = h_t / ind_weights;
        dind_weights = dind_weights + torch::sum(dh_t * h_t, 0);
        dh_t = dh_t * ind_weights;
    }
    return { dz, dh_t, dc_t, dind_weights, dhidden_weights, dcell_weights };
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fwd", &irevrnn_forward, "irevrnn forward");
    m.def("bwd", &irevrnn_backward, "irevrnn backward");
}
