#pragma once

namespace NN
{
    enum class Layer
    {
        Input,
        Dense,
        Conv2D,
        MaxPooling2D,
        Flatten,
        Output
    };

    enum class BufferID : uint32_t
    {
        Input = 0,
        Output = 1,
        Weights = 2,
        Biases = 3,
        Delta = 4,
        GradW = 5,
        GradB = 6,
        GradInput = 7
    };

    enum class Activation
    {
        Sigmoid,
        ReLU,
        Linear,
        Softmax,
    };

    enum class Loss
    {
        MSE,
        CrossEntropy,
        SparseCategoricalCrossEntropy
    };

    enum class Optimizer
    {
        SGD,
        SGDMomentum,
        Adam
    };

    enum class ComputeMode
    {
        SINGLE,
        MULTIPLE
    };

    class GPU
    {
    };
    class CPU
    {
    };

};