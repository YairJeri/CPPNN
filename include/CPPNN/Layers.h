#pragma once

#include "LayerBase.h"
#include "TensorDevice.h"

namespace NN
{
    struct InputLayer : LayerBase
    {
        InputLayer()
        {
        }

        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            output = input;
        }
        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            return;
        }
        void update_weights(float lr) override
        {
            return;
        }
    };

    struct DenseLayerCPU : LayerBase
    {
        int output_size;
        int input_size;
        Tensor<NN::CPU> weights;
        Tensor<NN::CPU> biases;
        Tensor<NN::CPU> input_cache;
        ComputeMode *compute_mode;

        DenseLayerCPU(int input_size, int output_size, ComputeMode &compute_mode) : output_size(output_size), input_size(input_size), compute_mode(&compute_mode)
        {
            weights = Tensor<NN::CPU>(input_size, output_size);
            biases = Tensor<NN::CPU>(1, output_size);
            biases.fill(0.0f);
            weights.generateWeights();
        }

        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            input_cache = input;
            switch (*compute_mode)
            {
            case ComputeMode::SINGLE:
                output = input.matmul(weights);
                output.broadcastAdd(biases);
                break;
            case ComputeMode::MULTIPLE:
                output = Tensor<NN::CPU>::forward(input, weights, biases);
                break;
            }
            // output = input.matmul(weights);
        }

        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            int batch_size = input_cache.getDim(0);

            switch (*compute_mode)
            {
            case ComputeMode::SINGLE:
            {
                Tensor<NN::CPU> grad_w = input_cache.transposeView(0, 1).matmul(delta);
                grad_w *= 1.0f / batch_size;

                Tensor<NN::CPU> grad_b = delta.sumAxis0();
                grad_b *= 1.0f / batch_size;

                delta = delta.matmul(weights.transposeView(0, 1));

                optimizer->update(weights, grad_w, lr);
                optimizer->update(biases, grad_b, lr);
                break;
            }
            case ComputeMode::MULTIPLE:
            {
                Tensor<NN::CPU> grad_w = input_cache.transpose().matmulThreaded(delta);
                grad_w *= 1.0f / batch_size;

                Tensor<NN::CPU> grad_b = delta.sumAxis0();
                grad_b *= 1.0f / batch_size;

                delta = delta.matmulThreaded(weights.transpose());

                optimizer->update(weights, grad_w, lr);
                optimizer->update(biases, grad_b, lr);
                break;
            }
            }
            /*
            Tensor<NN::CPU> grad_w = input_cache.transposeView(0, 1).matmul(delta);
            grad_w *= 1.0f / batch_size;

            Tensor<NN::CPU> grad_b = delta.sumAxis0();
            grad_b *= 1.0f / batch_size;

            delta = delta.matmul(weights.transposeView(0, 1));

            optimizer->update(weights, grad_w, lr);
            optimizer->update(biases, grad_b, lr);

            */
        }

        void update_weights(float lr) override {}
    };

    struct DenseLayerCPUMulti : LayerBase
    {
        int output_size;
        int input_size;
        Tensor<NN::CPU> weights;
        Tensor<NN::CPU> biases;
        Tensor<NN::CPU> input_cache;

        DenseLayerCPUMulti(int input_size, int output_size) : output_size(output_size), input_size(input_size)
        {
            weights = Tensor<NN::CPU>(input_size, output_size);
            biases = Tensor<NN::CPU>(1, output_size);
            biases.fill(0.0f);
            weights.generateWeights();
        }

        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            input_cache = input;
            output = input.matmulThreaded(weights);
            output.broadcastAdd(biases);
        }
        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            int batch_size = input_cache.getDim(0);

            Tensor<NN::CPU> grad_w = input_cache.transpose().matmulThreaded(delta);
            grad_w *= 1.0f / batch_size;

            Tensor<NN::CPU> grad_b = delta.sumAxis0();
            grad_b *= 1.0f / batch_size;

            delta = delta.matmulThreaded(weights.transpose());

            optimizer->update(weights, grad_w, lr);
            optimizer->update(biases, grad_b, lr);
        }

        void update_weights(float lr) override {}
    };

    struct DenseLayerGPU : LayerBase
    {
        int output_size;
        int input_size;
        LayerInstance layer;

        void init(VulkanGPUCompute *gpu)
        {
            this->input_size = input_size;
            this->output_size = output_size;

            layer.buffers[0] = Tensor<NN::GPU>(input_size);
            layer.buffers[1] = Tensor<NN::GPU>(output_size);
            layer.buffers[2] = Tensor<NN::GPU>(input_size, output_size);
            layer.buffers[3] = Tensor<NN::GPU>(output_size);

            layer.buffers[2].generateWeights();
            layer.buffers[3].fill(0.0f);

            layer.buffers[0].gpu_buffer = gpu->createBuffer(input_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            layer.buffers[1].gpu_buffer = gpu->createBuffer(output_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            layer.buffers[2].gpu_buffer = gpu->createBuffer(input_size * output_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            layer.buffers[3].gpu_buffer = gpu->createBuffer(output_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        }
    };
}