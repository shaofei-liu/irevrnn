from typing import Optional, Union, Sequence, Tuple
from enum import Enum, unique
import torch
from torch import nn, Tensor, LongTensor
from torch.nn import Parameter
from torch.autograd import Function


# supported activation functions
@unique
class Activation(int, Enum):
    relu = 0
    tanh = 1
    sigmoid = 2


def _d_relu(t):
    return (t >= 0).type_as(t)


def _d_tanh(t):
    tt = torch.tanh(t)
    return 1 - tt * tt


def _d_sigmoid(t):
    tt = torch.sigmoid(t)
    return tt * (1 - tt)


act_func = [(torch.relu, _d_relu), (torch.tanh, _d_tanh), (torch.sigmoid, _d_sigmoid)]


# wrap time unaware module to process sequential data by merging the time dimension
# with the batch dimension, assuming they occupy the first 2 dimensions
class SeqWrap(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def _forward_tensor(self, x: Tensor, *args, **kwargs) -> Tensor:
        headdims = x.size()[:2]
        x = x.reshape(-1, *x.size()[2:])
        x = self.model(x, *args, **kwargs)
        return x.reshape(*headdims, *x.size()[1:])

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if isinstance(x, Tensor):
            return self._forward_tensor(x, *args, **kwargs)
        else:
            raise TypeError('x should be of type torch.Tensor, '
                            'found {}'.format(type(x)))


# simply take the data at the last timestep
class TakeLast(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _forward_tensor(t: Tensor) -> Tensor:
        return t[-1]

    def forward(self, t: Tensor) -> Tensor:
        if isinstance(t, Tensor):
            return self._forward_tensor(t)
        else:
            raise TypeError('t should be of type torch.Tensor, '
                            'found {}'.format(type(t)))


class _IndDropoutFuncPy(Function):
    @staticmethod
    def forward(ctx, input, training, p):
        noise = torch.ones_like(input[0])
        if training:
            noise.bernoulli_(1 - p).div_(1 - p)
            noise = noise.unsqueeze(0).expand_as(input)
            output = input.mul_(noise)
        else:
            output = input
        ctx.save_for_backward(noise)
        ctx.training = training
        return output

    @staticmethod
    def backward(ctx, grad_output):
        noise, = ctx.saved_tensors
        if ctx.training:
            return grad_output.mul(noise), None, None
        else:
            return grad_output, None, None


def inddropout(input, training=False, p=0.5):
    return _IndDropoutFuncPy.apply(input, training, p)


class _RecurrentBatchNorm(nn.Module):
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, hidden_size, seq_len, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_RecurrentBatchNorm, self).__init__()
        time_d = seq_len * 4 + 1
        self.max_time_step = seq_len * 2
        channel_d = hidden_size
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.axes = (1,)
        if self.affine:
            self.weight = Parameter(torch.Tensor(channel_d))  # time_d
            self.bias = Parameter(torch.Tensor(channel_d))
            self.register_parameter('weight', self.weight)
            self.register_parameter('bias', self.bias)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(time_d, channel_d))
            self.register_buffer('running_var', torch.ones(time_d, channel_d))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            # nn.init.uniform_(self.weight)
            # nn.init.zeros_(self.bias)
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input, bn_start_original):
        self._check_input_dim(input)
        bn_start = bn_start_original
        if bn_start_original > self.max_time_step:
            bn_start = self.max_time_step
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        input_t, _, _ = input.size()
        # input_t=len(input)
        # calculate running estimates
        if self.training:
            mean = input.mean(1)  # torch.cumsum(a, dim=0)  cumsummdr100 / np.arange(1,51)
            # use biased var in train
            var = input.var(1, unbiased=False)
            n = input.size()[1]  # input.numel() / input.size(1)#* n / (n - 1)

            # part_mean=self.running_mean[:input_t]
            with torch.no_grad():
                self.running_mean[bn_start:input_t + bn_start] = exponential_average_factor * mean \
                                                                 + (1 - exponential_average_factor) \
                                                                 * self.running_mean[bn_start:input_t + bn_start]
                # update running_var with unbiased var
                self.running_var[bn_start:input_t + bn_start] = exponential_average_factor * var * n / (n - 1) \
                                                                + (1 - exponential_average_factor) \
                                                                * self.running_var[bn_start:input_t + bn_start]
        else:
            mean = self.running_mean[bn_start:input_t + bn_start]
            var = self.running_var[bn_start:input_t + bn_start]

        output = (input - mean[:, None, :]) / (torch.sqrt(var[:, None, :] + self.eps))
        if self.affine:
            output = output * self.weight[None, None, :] + self.bias[None, None, :]  # [None, :, None, None]

        return output
