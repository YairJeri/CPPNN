#pragma once
#include <cmath>
#include "LayerBase.h"
#include "Optimizer.h"

namespace NN
{

    float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
    float sigmoid_derivative(float x) { return x * (1.0 - x); }

    struct ActivationBase : LayerBase
    {
        virtual void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) = 0;
        virtual void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) = 0;
        virtual void update_weights(float lr) {};
    };

    struct ActivationReLU : ActivationBase
    {
        Tensor<NN::CPU> output_cache;
        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            const int rows = input.getSize();

            for (int i = 0; i < rows; ++i)
                output(i) = input(i) > 0.0f ? input(i) : 0.0f;
            output_cache = output;
        }
        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            const int rows = delta.getSize();

            for (int i = 0; i < rows; ++i)
                delta(i) *= output_cache(i) > 0.0f ? 1.0f : 0.0f;
        }
    };

    struct ActivationSigmoid : ActivationBase
    {
        Tensor<NN::CPU> output_cache;
        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            const int rows = input.getSize();

            for (int i = 0; i < rows; ++i)
                output(i) = 1.0f / (1.0f + std::exp(-input(i)));
            output_cache = output;
        }
        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            const int rows = delta.getSize();

            for (int i = 0; i < rows; ++i)
                delta(i) *= output_cache(i) * (1.0f - output_cache(i));
        }
    };

    struct ActivationTanh : ActivationBase
    {
        Tensor<NN::CPU> output_cache;
        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            const int rows = input.getSize();

            for (int i = 0; i < rows; ++i)
                output(i) = std::tanh(input(i));
            output_cache = output;
        }
        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            const int rows = delta.getSize();

            for (int i = 0; i < rows; ++i)
                delta(i) *= (1.0f - output_cache(i) * output_cache(i));
        }
    };

    struct ActivationSoftmax : ActivationBase
    {
        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            const int batch_size = input.getDim(0);
            const int num_classes = input.getDim(1);

            Tensor<NN::CPU> temp(batch_size, num_classes);

            for (int i = 0; i < batch_size; ++i)
            {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < num_classes; ++j)
                {
                    float val = input(i, j);
                    max_val = std::max(max_val, val);
                }

                float sum = 0.0f;
                for (int j = 0; j < num_classes; ++j)
                {
                    float e = std::exp(input(i, j) - max_val);
                    temp(i, j) = e;
                    sum += e;
                }

                for (int j = 0; j < num_classes; ++j)
                    temp(i, j) /= sum;
            }
            output = temp;
        }
        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
        }
    };

}