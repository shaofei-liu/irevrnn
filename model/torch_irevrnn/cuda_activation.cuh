#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


// macros to dispatch activation function "ind_act" and "res_act" according to flag
#define INDEPENDENTLY_ACTIVATION_TYPE(flag, type, ...)    \
  case flag: {                                            \
    using ind_act = type<scalar_t>;                         \
    return __VA_ARGS__();                                 \
  }

#define RESIDUAL_ACTIVATION_TYPE(flag, type, ...)    \
  case flag: {                                       \
    using res_act = type<scalar_t>;                    \
    return __VA_ARGS__();                            \
  }

#define DISPATCH_INDEPENDENTLY_ACTIVATION(IND_FLAG, ...)                             \
  [&] {                                                                              \
    switch (IND_FLAG) {                                                              \
      INDEPENDENTLY_ACTIVATION_TYPE(0, act_relu, __VA_ARGS__)                        \
      INDEPENDENTLY_ACTIVATION_TYPE(1, act_tanh, __VA_ARGS__)                        \
      INDEPENDENTLY_ACTIVATION_TYPE(2, act_sigmoid, __VA_ARGS__)                     \
      default:                                                                       \
        AT_ERROR("Unknown flag for independently activation function: ", #IND_FLAG); \
    }                                                                                \
  }()

#define DISPATCH_RESIDUAL_ACTIVATION(RES_FLAG, ...)                             \
  [&] {                                                                         \
    switch (RES_FLAG) {                                                         \
      RESIDUAL_ACTIVATION_TYPE(0, act_relu, __VA_ARGS__)                        \
      RESIDUAL_ACTIVATION_TYPE(1, act_tanh, __VA_ARGS__)                        \
      RESIDUAL_ACTIVATION_TYPE(2, act_sigmoid, __VA_ARGS__)                     \
      default:                                                                  \
        AT_ERROR("Unknown flag for residual activation function: ", #RES_FLAG); \
    }                                                                           \
  }()

// generic activation functions "ind_t" and "res_t", forward & backward

template <typename scalar_t>
struct act_relu
{
    __device__ static scalar_t forward(scalar_t x)
    {
        return (x > 0.0) ? x : 0.0;
    }

    __device__ static scalar_t backward(scalar_t x)
    {
        return (x > 0.0) ? 1.0 : 0.0;
    }
};

template <typename scalar_t>
struct act_tanh
{
    __device__ static scalar_t forward(scalar_t x)
    {
        return tanh(x);
    }

    __device__ static scalar_t backward(scalar_t x)
    {
        const auto t = tanh(x);
        return 1 - (t * t);
    }
};

template <typename scalar_t>
struct act_sigmoid
{
    __device__ static scalar_t forward(scalar_t x)
    {
        return 1 / (1 + exp(-x));
    }

    __device__ static scalar_t backward(scalar_t x)
    {
        const auto t = 1 / (1 + exp(-x));
        return t * (1 - t);
    }
};

template <typename scalar_t>
__device__ static scalar_t forward_tanh(scalar_t x)
{
    return tanh(x);
}

template <typename scalar_t>
__device__ static scalar_t backward_tanh(scalar_t x)
{
    const auto t = tanh(x);
    return 1 - (t * t);
}