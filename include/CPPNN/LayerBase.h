#pragma once

#include "TensorDevice.h"
#include "Optimizer.h"

namespace NN
{

    struct LayerBase
    {
        virtual void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) = 0;
        virtual void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) = 0;
        virtual void update_weights(float lr) = 0;
    };
}