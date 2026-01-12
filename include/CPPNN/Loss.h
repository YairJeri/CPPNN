#pragma once
#include "TensorDevice.h"

namespace NN
{
    struct LossFunction
    {
        virtual float computeLoss(const Tensor<NN::CPU> &output, const Tensor<NN::CPU> &target, Tensor<NN::CPU> &delta) = 0;
        virtual ~LossFunction() {}
    };
    struct MSELoss : LossFunction
    {
        float computeLoss(const Tensor<NN::CPU> &output, const Tensor<NN::CPU> &target, Tensor<NN::CPU> &delta) override
        {
            float sum = 0.0;
            const int totalElements = output.getSize();
            const int batch_size = output.getDim(0);
            for (int i = 0; i < totalElements; ++i)
            {
                sum += std::pow(output(i) - target(i), 2); // acceso plano
                delta(i) = 2.0 * (output(i) - target(i)) / batch_size;
            }

            return sum;
        }
    };
    struct CrossEntropyLoss : LossFunction
    {
        float computeLoss(const Tensor<NN::CPU> &output, const Tensor<NN::CPU> &target, Tensor<NN::CPU> &delta) override
        {
            float loss = 0.0;
            const int size = output.getSize();
            const int batch_size = output.getDim(0);
            for (int i = 0; i < size; ++i)
            {
                loss -= target(i) * std::log(output(i) + 1e-12f);
                delta(i) = (output(i) - target(i)) / batch_size;
            }
            return loss;
        }
    };

    struct SparseCategoricalCrossEntropyLoss : LossFunction
    {
        float computeLoss(const Tensor<NN::CPU> &output, const Tensor<NN::CPU> &target, Tensor<NN::CPU> &delta) override
        {
            Tensor<NN::CPU> targetOneHot(target.getDim(0), output.getDim(1));
            targetOneHot.fill(0.0f);
            for (int i = 0; i < target.getDim(0); ++i)
                targetOneHot(i, target(i)) = 1.0f;

            return CrossEntropyLoss().computeLoss(output, targetOneHot, delta);
        }
    };
}
