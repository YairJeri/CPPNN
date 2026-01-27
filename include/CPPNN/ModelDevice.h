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
        float dropout_rate = 0.2f;
        int kernel_size = 3;
        int stride = 1;
        int padding = 1;
        int in_channels = 1;
        int out_channels = 16;
        int in_h = 28;
        int in_w = 28;
        float scale = 1.0f / 255.0f;
        float offset = 0.0f;
        NN::WeightInit weight_init = NN::WeightInit::HeNormal;
    };

    class ModelBase
    {
    protected:
        std::vector<LayerBase *> layers;

    public:
        const std::vector<LayerBase *> &getLayers() const { return layers; }
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
                input_size = layer.output_size;
                break;
            case Layer::Rescaling:
                layers.push_back(new RescalingLayerCPU(layer.scale, layer.offset));
                break;
            case Layer::Dense:
                layers.push_back(new DenseLayerCPU(input_size, layer.output_size, compute_mode, layer.weight_init));
                input_size = layer.output_size;
                break;
            case Layer::Dropout:
                layers.push_back(new DropoutLayerCPU(layer.dropout_rate));
                break;
            case Layer::MaxPooling2D:
                layers.push_back(new MaxPooling2DCPU(layer.kernel_size, layer.stride));
                break;
            case Layer::Flatten:
                layers.push_back(new FlattenLayerCPU());
                input_size = layer.output_size;
                break;
            case Layer::Conv2D:
                layers.push_back(new Conv2DLayerCPU(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, layer.weight_init));
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
            if (layer.type == Layer::Dense)
            {
                input_size = layer.output_size;
            }
        }

        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            if (layers.size() <= 1)
            {
                output = input;
                return;
            }
            Tensor<NN::CPU> cur_in = input;
            Tensor<NN::CPU> cur_out;
            for (size_t i = 1; i < layers.size(); ++i)
            {
                layers[i]->forward(cur_in, cur_out);
                cur_in = cur_out;
            }
            output = cur_out;
        }

        std::vector<std::vector<float>> getLayerActivations(const Tensor<NN::CPU> &input)
        {
            std::vector<std::vector<float>> acts;
            if (layers.size() <= 1) return acts;

            // Capa 0: entrada
            std::vector<float> in_act(input.getSize());
            const float *inp = input.getData();
            for (int j = 0; j < input.getSize(); ++j) in_act[j] = inp[j];
            acts.push_back(in_act);

            Tensor<NN::CPU> cur_in = input;
            Tensor<NN::CPU> cur_out;
            for (size_t i = 1; i < layers.size(); ++i)
            {
                layers[i]->forward(cur_in, cur_out);
                std::vector<float> l_act(cur_out.getSize());
                const float *outp = cur_out.getData();
                for (int j = 0; j < cur_out.getSize(); ++j) l_act[j] = outp[j];
                acts.push_back(l_act);
                cur_in = cur_out;
            }
            return acts;
        }

        float train(const Tensor<NN::CPU> &input, const Tensor<NN::CPU> &target, float lr) override
        {
            Tensor<NN::CPU> output;
            forward(input, output);
            Tensor<NN::CPU> delta(output.getDim(0), output.getDim(1));

            float loss = lossFunction->computeLoss(output, target, delta);
            for (int i = 0; i < delta.getSize(); ++i)
            {
                if (std::isnan(delta(i)))
                {
                    std::cout << "NAN detected in delta!" << std::endl;
                    return -20000;
                }
            }

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
                    size_t count = std::min(batch_size, s - i);
                    Tensor<NN::CPU> act_input = input.createView(i, count);
                    Tensor<NN::CPU> act_target = target.createView(i, count);

                    double l = train(act_input, act_target, lr);
                    if (l == -20000)
                        break;

                    loss += l * count;
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
                input_size = layer.output_size;
                break;
            case Layer::Rescaling:
                layers.push_back(new RescalingLayerCPU(layer.scale, layer.offset));
                break;
            case Layer::Dense:
                layers.push_back(new DenseLayerCPU(input_size, layer.output_size, compute_mode, layer.weight_init));
                input_size = layer.output_size;
                break;
            case Layer::Dropout:
                layers.push_back(new DropoutLayerCPU(layer.dropout_rate));
                break;
            case Layer::MaxPooling2D:
                layers.push_back(new MaxPooling2DCPU(layer.kernel_size, layer.stride));
                break;
            case Layer::Flatten:
                layers.push_back(new FlattenLayerCPU());
                input_size = layer.output_size;
                break;
            case Layer::Conv2D:
                layers.push_back(new Conv2DLayerCPU(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, layer.weight_init));
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
            if (layer.type == Layer::Dense)
            {
                input_size = layer.output_size;
            }
        }
        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output)
        {
            if (layers.size() <= 1)
            {
                output = input;
                return;
            }
            Tensor<NN::CPU> cur_in = input;
            Tensor<NN::CPU> cur_out;
            for (size_t i = 1; i < layers.size(); ++i)
            {
                layers[i]->forward(cur_in, cur_out);
                cur_in = cur_out;
            }
            output = cur_out;
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
                    size_t count = std::min(batch_size, s - i);
                    Tensor<NN::CPU> act_input = input.createView(i, count);
                    Tensor<NN::CPU> act_target = target.createView(i, count);

                    double l = train(act_input, act_target, lr);
                    if (l == -20000)
                        break;

                    loss += l * count;
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
