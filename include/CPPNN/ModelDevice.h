#pragma once

#include "TensorDevice.h"
#include "ActivationLayer.h"
#include "Layers.h"
#include "Loss.h"
#include "Optimizer.h"
#include <vector>
#include <iostream>
#include <cmath>

namespace NN
{
    struct LayerStruct
    {
        NN::Layer type;
        int output_size = 0;
        NN::Activation activation = NN::Activation::Linear;
    };

    class ModelBase
    {
    protected:
        std::vector<LayerBase *> layers;

    public:
        virtual void addLayer(LayerStruct layer, int &input_size) = 0;
        virtual void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) = 0;
        virtual float train(const Tensor<NN::CPU> &input, const Tensor<NN::CPU> &target, float lr) = 0;
        virtual void fit(const Tensor<NN::CPU> &input, const Tensor<NN::CPU> &target, float lr, int epochs, size_t batch_size) = 0;
        virtual void predict(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) = 0;
    };

    template <typename Device>
    class Model
    {
    };

    template <>
    class Model<NN::CPU> : public ModelBase
    {
        LossFunction *lossFunction = nullptr;
        OptimizerBase *optimizer = nullptr;
        ComputeMode compute_mode = ComputeMode::SINGLE;

    public:
        Model(const std::vector<LayerStruct> &layersBase, Loss loss, Optimizer opt = Optimizer::SGD)
        {
            layers.reserve(layersBase.size());
            int input_size = 0;
            for (auto &layerB : layersBase)
            {
                addLayer(layerB, input_size);
            }

            switch (loss)
            {
            case Loss::MSE:
                lossFunction = new MSELoss();
                break;
            case Loss::CrossEntropy:
                lossFunction = new CrossEntropyLoss();
                break;
            case Loss::SparseCategoricalCrossEntropy:
                lossFunction = new SparseCategoricalCrossEntropyLoss();
                break;
            }

            switch (opt)
            {
            case Optimizer::SGD:
                optimizer = new SGD();
                break;
            case Optimizer::SGDMomentum:
                optimizer = new SGDMomentum(0.9f);
                break;
            case Optimizer::Adam:
                optimizer = new Adam();
                break;
            }
        }
        ~Model()
        {
            for (auto &layer : layers)
            {
                delete layer;
            }
        }
        void addLayer(LayerStruct layer, int &input_size) override
        {
            switch (layer.type)
            {
            case Layer::Input:
                layers.push_back(new InputLayer());
                break;
            case Layer::Dense:
                layers.push_back(new DenseLayerCPU(input_size, layer.output_size, compute_mode));
                break;
            case Layer::Conv2D:
                // layers.push_back(new Conv2DLayer(layer.output_size, layer.kernel_size, layer.stride, layer.padding));
                break;
            case Layer::MaxPooling2D:
                // layers.push_back(new MaxPooling2D(layer.pool_size));
                break;
            case Layer::Flatten:
                // layers.push_back(new Flatten());
                break;
            }
            switch (layer.activation)
            {
            case Activation::Sigmoid:
                layers.push_back(new ActivationSigmoid());
                break;
            case Activation::ReLU:
                layers.push_back(new ActivationReLU());
                break;
            case Activation::Softmax:
                layers.push_back(new ActivationSoftmax());
                break;
            case Activation::Linear:
                // layers.back()->activation = ActivationType::Linear;
                break;
            }
            input_size = layer.output_size;
        }

        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            layers[1]->forward(input, output);
            for (int i = 2; i < layers.size(); ++i)
            {
                layers[i]->forward(output, output);
            }
        }

        float train(const Tensor<NN::CPU> &input, const Tensor<NN::CPU> &target, float lr) override
        {
            Tensor<NN::CPU> output;
            forward(input, output);
            Tensor<NN::CPU> delta(output.getDim(0), output.getDim(1));
            for (int i = 0; i < output.getSize(); ++i)
            {
                if (std::isnan(delta(i)))
                {
                    delta.print();
                    std::cout << "NAN" << std::endl;
                    return -20000;
                }
            }

            float loss = lossFunction->computeLoss(output, target, delta);
            for (int i = layers.size() - 1; i >= 0; i--)
            {
                layers[i]->backward(delta, lr, optimizer);
            }

            return loss;
        }

        void fit(const Tensor<NN::CPU> &input, const Tensor<NN::CPU> &target, float lr, int epochs, size_t batch_size) override
        {
            for (int e = 0; e < epochs; ++e)
            {
                double loss = 0.0;
                size_t s = input.getDim(0);
                for (size_t i = 0; i < s; i += batch_size)
                {
                    int cuantity = std::min(batch_size, s - i);
                    Tensor<NN::CPU> act_input = input.createView(i, batch_size);
                    Tensor<NN::CPU> act_target = target.createView(i, batch_size);

                    double l = train(act_input, act_target, lr);
                    if (l == -20000)
                        break;

                    loss += l;
                }
                loss /= s;
                if (e % 10 == 0)
                {
                    std::cout << "Epoch " << e << ", Loss: " << loss << std::endl;
                }
            }
        }
        void predict(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            forward(input, output);
        }

        void enableMultiThreading()
        {
            compute_mode = ComputeMode::MULTIPLE;
        }
        void disableMultiThreading()
        {
            compute_mode = ComputeMode::SINGLE;
        }
    };

    class Model2
    {
        std::vector<LayerBase *> layers;
        LossFunction *lossFunction = nullptr;
        OptimizerBase *optimizer = nullptr;
        ComputeMode compute_mode = ComputeMode::SINGLE;

    public:
        Model2(const std::vector<LayerStruct> &layersBase, Loss loss, Optimizer opt = Optimizer::SGD)
        {
            layers.reserve(layersBase.size());
            int input_size = 0;
            for (auto &layerB : layersBase)
            {
                addLayer(layerB, input_size);
            }

            switch (loss)
            {
            case Loss::MSE:
                lossFunction = new MSELoss();
                break;
            case Loss::CrossEntropy:
                lossFunction = new CrossEntropyLoss();
                break;
            case Loss::SparseCategoricalCrossEntropy:
                lossFunction = new SparseCategoricalCrossEntropyLoss();
                break;
            }

            switch (opt)
            {
            case Optimizer::SGD:
                optimizer = new SGD();
                break;
            case Optimizer::SGDMomentum:
                optimizer = new SGDMomentum(0.9f);
                break;
            case Optimizer::Adam:
                optimizer = new Adam();
                break;
            }
        }
        ~Model2()
        {
            for (auto &layer : layers)
            {
                delete layer;
            }
        }

        void addLayer(LayerStruct layer, int &input_size)
        {
            switch (layer.type)
            {
            case Layer::Input:
                layers.push_back(new InputLayer());
                break;
            case Layer::Dense:
                layers.push_back(new DenseLayerCPU(input_size, layer.output_size, compute_mode));
                break;
            case Layer::Conv2D:
                // layers.push_back(new Conv2DLayer(layer.output_size, layer.kernel_size, layer.stride, layer.padding));
                break;
            case Layer::MaxPooling2D:
                // layers.push_back(new MaxPooling2D(layer.pool_size));
                break;
            case Layer::Flatten:
                // layers.push_back(new Flatten());
                break;
            }
            switch (layer.activation)
            {
            case Activation::Sigmoid:
                layers.push_back(new ActivationSigmoid());
                break;
            case Activation::ReLU:
                layers.push_back(new ActivationReLU());
                break;
            case Activation::Softmax:
                layers.push_back(new ActivationSoftmax());
                break;
            case Activation::Linear:
                // layers.back()->activation = ActivationType::Linear;
                break;
            }
            input_size = layer.output_size;
        }
        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output)
        {
            layers[1]->forward(input, output);
            for (int i = 2; i < layers.size(); ++i)
            {
                layers[i]->forward(output, output);
            }
        }

        double train(const Tensor<NN::CPU> &input, const Tensor<NN::CPU> &target, float lr)
        {
            Tensor<NN::CPU> output;
            forward(input, output);
            Tensor<NN::CPU> delta(output.getDim(0), output.getDim(1));
            for (int i = 0; i < output.getSize(); ++i)
            {
                if (std::isnan(delta(i)))
                {
                    delta.print();
                    std::cout << "NAN" << std::endl;
                    return -20000;
                }
            }

            float loss = lossFunction->computeLoss(output, target, delta);
            for (int i = layers.size() - 1; i >= 0; i--)
            {
                layers[i]->backward(delta, lr, optimizer);
            }

            return loss;
        }

        void fit(const Tensor<NN::CPU> &input, const Tensor<NN::CPU> &target, float lr, int epochs, size_t batch_size)
        {
            for (int e = 0; e < epochs; ++e)
            {
                double loss = 0.0;
                size_t s = input.getDim(0);
                for (size_t i = 0; i < s; i += batch_size)
                {
                    int cuantity = std::min(batch_size, s - i);
                    Tensor<NN::CPU> act_input = input.createView(i, batch_size);
                    Tensor<NN::CPU> act_target = target.createView(i, batch_size);

                    double l = train(act_input, act_target, lr);
                    if (l == -20000)
                        break;

                    loss += l;
                }
                loss /= s;
                if (e % 10 == 0)
                {
                    std::cout << "Epoch " << e << ", Loss: " << loss << std::endl;
                }
            }
        }

        void predict(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output)
        {
            forward(input, output);
        }
    };
}
