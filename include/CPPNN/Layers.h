#pragma once

#include "LayerBase.h"
#include "TensorDevice.h"
#include "VulkanGPUCompute.h"

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

    struct DropoutLayerCPU : LayerBase
    {
        float rate;
        bool is_training = true;
        Tensor<NN::CPU> mask;

        DropoutLayerCPU(float rate = 0.2f) : rate(rate) {}

        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            output = input;
            if (!is_training || rate <= 0.0f)
                return;

            mask = Tensor<NN::CPU>(input.getDim(0), input.getDim(1));
            float scale = 1.0f / (1.0f - rate);
            int sz = input.getSize();
            float *out_data = output.getData();
            float *mask_data = mask.getData();

            for (int i = 0; i < sz; ++i)
            {
                float r = (float)rand() / RAND_MAX;
                if (r < rate)
                {
                    out_data[i] = 0.0f;
                    mask_data[i] = 0.0f;
                }
                else
                {
                    out_data[i] *= scale;
                    mask_data[i] = scale;
                }
            }
        }

        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            if (!is_training || rate <= 0.0f)
                return;

            int sz = delta.getSize();
            float *d_data = delta.getData();
            float *m_data = mask.getData();
            for (int i = 0; i < sz; ++i)
            {
                d_data[i] *= m_data[i];
            }
        }

        void update_weights(float lr) override {}
    };

    struct RescalingLayerCPU : LayerBase
    {
        float scale = 1.0f / 255.0f;
        float offset = 0.0f;

        RescalingLayerCPU(float scale = 1.0f / 255.0f, float offset = 0.0f)
            : scale(scale), offset(offset) {}

        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            Tensor<NN::CPU> res(input);
            int sz = res.getSize();
            float *out_ptr = res.getData();
            for (int i = 0; i < sz; ++i)
            {
                out_ptr[i] = out_ptr[i] * scale + offset;
            }
            output = std::move(res);
        }

        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            int sz = delta.getSize();
            float *d_ptr = delta.getData();
            for (int i = 0; i < sz; ++i)
            {
                d_ptr[i] = d_ptr[i] * scale;
            }
        }

        void update_weights(float lr) override {}
    };

    struct FlattenLayerCPU : LayerBase
    {
        int original_dims[4] = {1, 1, 1, 1};

        FlattenLayerCPU() {}

        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            int batch = input.getDim(0);
            int total = input.getSize();
            int feat = (batch > 0) ? (total / batch) : total;

            for (int i = 0; i < 4; ++i)
                original_dims[i] = input.getDim(i);

            Tensor<NN::CPU> res(batch, feat);
            std::memcpy(res.getData(), input.getData(), total * sizeof(float));
            output = std::move(res);
        }

        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            int total = delta.getSize();
            Tensor<NN::CPU> reshaped(original_dims[0], original_dims[1], original_dims[2], original_dims[3]);
            std::memcpy(reshaped.getData(), delta.getData(), total * sizeof(float));
            delta = std::move(reshaped);
        }

        void update_weights(float lr) override {}
    };

    struct Conv2DLayerCPU : LayerBase
    {
        int out_channels;
        int in_channels;
        int kernel_size;
        int stride;
        int padding;

        Tensor<NN::CPU> weights; // [out_c, in_c, k, k]
        Tensor<NN::CPU> biases;  // [1, out_c]
        Tensor<NN::CPU> input_cache;

        int in_b = 1, in_c = 1, in_h = 28, in_w = 28;
        int out_h = 28, out_w = 28;

        Conv2DLayerCPU(int in_c, int out_c, int k_size = 3, int stride = 1, int pad = 1, WeightInit init = WeightInit::HeNormal)
            : in_channels(in_c), out_channels(out_c), kernel_size(k_size), stride(stride), padding(pad)
        {
            weights = Tensor<NN::CPU>(out_c, in_c, k_size, k_size);
            biases = Tensor<NN::CPU>(1, out_c);
            biases.fill(0.0f);
            weights.generateWeights(init);
        }

        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            in_b = input.getDim(0);
            if (in_b <= 0) in_b = 1;

            if (input.getNdim() >= 4) {
                in_c = input.getDim(1);
                in_h = input.getDim(2);
                in_w = input.getDim(3);
            } else {
                in_c = in_channels;
                int total_per_batch = (in_b > 0) ? (input.getSize() / in_b) : input.getSize();
                int spatial = (in_c > 0) ? (total_per_batch / in_c) : total_per_batch;
                in_h = (int)std::round(std::sqrt((double)spatial));
                in_w = (in_h > 0) ? (spatial / in_h) : 1;
            }

            out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
            out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
            if (out_h < 1) out_h = 1;
            if (out_w < 1) out_w = 1;

            input_cache = input;
            Tensor<NN::CPU> res(in_b, out_channels, out_h, out_w);

            const float *in_ptr = input_cache.getData();
            const float *w_ptr = weights.getData();
            const float *b_ptr = biases.getData();
            float *out_ptr = res.getData();

            int in_sz = input_cache.getSize();
            int out_sz = res.getSize();
            int w_sz = weights.getSize();

            for (int b = 0; b < in_b; ++b) {
                for (int oc = 0; oc < out_channels; ++oc) {
                    float bias_val = b_ptr[oc];
                    for (int oh = 0; oh < out_h; ++oh) {
                        int in_y_org = oh * stride - padding;
                        for (int ow = 0; ow < out_w; ++ow) {
                            int in_x_org = ow * stride - padding;
                            float sum = bias_val;

                            for (int ic = 0; ic < in_c; ++ic) {
                                for (int kh = 0; kh < kernel_size; ++kh) {
                                    int iy = in_y_org + kh;
                                    if (iy < 0 || iy >= in_h) continue;
                                    for (int kw = 0; kw < kernel_size; ++kw) {
                                        int ix = in_x_org + kw;
                                        if (ix < 0 || ix >= in_w) continue;

                                        int in_idx = ((b * in_c + ic) * in_h + iy) * in_w + ix;
                                        int w_idx  = ((oc * in_c + ic) * kernel_size + kh) * kernel_size + kw;
                                        if (in_idx >= 0 && in_idx < in_sz && w_idx >= 0 && w_idx < w_sz) {
                                            sum += in_ptr[in_idx] * w_ptr[w_idx];
                                        }
                                    }
                                }
                            }
                            int out_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
                            if (out_idx >= 0 && out_idx < out_sz) {
                                out_ptr[out_idx] = sum;
                            }
                        }
                    }
                }
            }
            output = std::move(res);
        }

        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            Tensor<NN::CPU> grad_w(out_channels, in_channels, kernel_size, kernel_size);
            grad_w.fill(0.0f);
            Tensor<NN::CPU> grad_b(1, out_channels);
            grad_b.fill(0.0f);
            Tensor<NN::CPU> grad_in(in_b, in_channels, in_h, in_w);
            grad_in.fill(0.0f);

            const float *in_ptr = input_cache.getData();
            const float *d_ptr = delta.getData();
            const float *w_ptr = weights.getData();
            float *gw_ptr = grad_w.getData();
            float *gb_ptr = grad_b.getData();
            float *gin_ptr = grad_in.getData();

            int in_sz = input_cache.getSize();
            int d_sz = delta.getSize();
            int w_sz = weights.getSize();
            int gin_sz = grad_in.getSize();
            int gw_sz = grad_w.getSize();

            for (int b = 0; b < in_b; ++b) {
                for (int oc = 0; oc < out_channels; ++oc) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        int in_y_org = oh * stride - padding;
                        for (int ow = 0; ow < out_w; ++ow) {
                            int in_x_org = ow * stride - padding;
                            int d_idx = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
                            if (d_idx < 0 || d_idx >= d_sz) continue;
                            float d_val = d_ptr[d_idx];
                            gb_ptr[oc] += d_val;

                            for (int ic = 0; ic < in_c; ++ic) {
                                for (int kh = 0; kh < kernel_size; ++kh) {
                                    int iy = in_y_org + kh;
                                    if (iy < 0 || iy >= in_h) continue;
                                    for (int kw = 0; kw < kernel_size; ++kw) {
                                        int ix = in_x_org + kw;
                                        if (ix < 0 || ix >= in_w) continue;

                                        int in_idx = ((b * in_c + ic) * in_h + iy) * in_w + ix;
                                        int w_idx  = ((oc * in_c + ic) * kernel_size + kh) * kernel_size + kw;

                                        if (in_idx >= 0 && in_idx < in_sz && w_idx >= 0 && w_idx < w_sz) {
                                            gw_ptr[w_idx] += in_ptr[in_idx] * d_val;
                                            if (in_idx < gin_sz) {
                                                gin_ptr[in_idx] += d_val * w_ptr[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            optimizer->update(weights, grad_w, lr);
            optimizer->update(biases, grad_b, lr);

            delta = grad_in;
        }

        void update_weights(float lr) override {}
    };

    struct MaxPooling2DCPU : LayerBase
    {
        int pool_size = 2;
        int stride = 2;
        std::vector<int> max_indices;
        int in_b = 1, in_c = 1, in_h = 28, in_w = 28;
        int out_h = 14, out_w = 14;

        MaxPooling2DCPU(int pool_size = 2, int stride = 2) : pool_size(pool_size), stride(stride) {}

        void forward(const Tensor<NN::CPU> &input, Tensor<NN::CPU> &output) override
        {
            in_b = input.getDim(0);
            if (input.getNdim() >= 4) {
                in_c = input.getDim(1);
                in_h = input.getDim(2);
                in_w = input.getDim(3);
            } else {
                in_c = 1;
                in_h = (int)std::sqrt(input.getDim(1));
                in_w = (in_h > 0) ? (input.getDim(1) / in_h) : 1;
            }

            out_h = (in_h >= pool_size) ? ((in_h - pool_size) / stride + 1) : 1;
            out_w = (in_w >= pool_size) ? ((in_w - pool_size) / stride + 1) : 1;

            Tensor<NN::CPU> res(in_b, in_c, out_h, out_w);
            int total_out = in_b * in_c * out_h * out_w;
            max_indices.assign(total_out, 0);

            const float *in_ptr = input.getData();
            float *out_ptr = res.getData();
            int max_in_size = input.getSize();

            for (int b = 0; b < in_b; ++b) {
                for (int c = 0; c < in_c; ++c) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            float max_v = -1e38f;
                            int best_i = 0;
                            for (int ph = 0; ph < pool_size; ++ph) {
                                int ih = oh * stride + ph;
                                if (ih >= in_h) continue;
                                for (int pw = 0; pw < pool_size; ++pw) {
                                    int iw = ow * stride + pw;
                                    if (iw >= in_w) continue;
                                    int in_idx = ((b * in_c + c) * in_h + ih) * in_w + iw;
                                    if (in_idx < max_in_size && in_ptr[in_idx] > max_v) {
                                        max_v = in_ptr[in_idx];
                                        best_i = in_idx;
                                    }
                                }
                            }
                            int out_idx = ((b * in_c + c) * out_h + oh) * out_w + ow;
                            out_ptr[out_idx] = max_v;
                            max_indices[out_idx] = best_i;
                        }
                    }
                }
            }
            output = std::move(res);
        }

        void backward(Tensor<NN::CPU> &delta, float lr, OptimizerBase *optimizer) override
        {
            Tensor<NN::CPU> grad_in(in_b, in_c, in_h, in_w);
            grad_in.fill(0.0f);
            float *gin_ptr = grad_in.getData();
            const float *d_ptr = delta.getData();

            int total_out = in_b * in_c * out_h * out_w;
            for (int i = 0; i < total_out; ++i) {
                int target = max_indices[i];
                if (target >= 0 && target < in_b * in_c * in_h * in_w)
                    gin_ptr[target] += d_ptr[i];
            }
            delta = grad_in;
        }

        void update_weights(float lr) override {}
    };

    struct DenseLayerCPU : LayerBase
    {
        int output_size;
        int input_size;
        Tensor<NN::CPU> weights;
        Tensor<NN::CPU> biases;
        Tensor<NN::CPU> input_cache;
        ComputeMode *compute_mode;

        DenseLayerCPU(int input_size, int output_size, ComputeMode &compute_mode, WeightInit init = WeightInit::HeNormal) : output_size(output_size), input_size(input_size), compute_mode(&compute_mode)
        {
            weights = Tensor<NN::CPU>(input_size, output_size);
            biases = Tensor<NN::CPU>(1, output_size);
            biases.fill(0.0f);
            weights.generateWeights(init);
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

            // delta ya viene normalizado por batch_size desde LossFunction
            switch (*compute_mode)
            {
            case ComputeMode::SINGLE:
            {
                Tensor<NN::CPU> grad_w = input_cache.transposeView(0, 1).matmul(delta);
                Tensor<NN::CPU> grad_b = delta.sumAxis0();

                delta = delta.matmul(weights.transposeView(0, 1));

                optimizer->update(weights, grad_w, lr);
                optimizer->update(biases, grad_b, lr);
                break;
            }
            case ComputeMode::MULTIPLE:
            {
                Tensor<NN::CPU> grad_w = input_cache.transpose().matmulThreaded(delta);
                Tensor<NN::CPU> grad_b = delta.sumAxis0();

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

            layer.buffers[NN::BufferID::Input] = Tensor<NN::GPU>(input_size);
            layer.buffers[NN::BufferID::Output] = Tensor<NN::GPU>(output_size);
            layer.buffers[NN::BufferID::Weights] = Tensor<NN::GPU>(input_size, output_size);
            layer.buffers[NN::BufferID::Biases] = Tensor<NN::GPU>(output_size);

            layer.buffers[NN::BufferID::Weights].generateWeights();
            layer.buffers[NN::BufferID::Biases].fill(0.0f);

            layer.buffers[NN::BufferID::Input].gpu_buffer = gpu->createBuffer(input_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            layer.buffers[NN::BufferID::Output].gpu_buffer = gpu->createBuffer(output_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            layer.buffers[NN::BufferID::Weights].gpu_buffer = gpu->createBuffer(input_size * output_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
            layer.buffers[NN::BufferID::Biases].gpu_buffer = gpu->createBuffer(output_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
        }
    };
}
// Updated gradient calculation notes

// Updated gradient calculation notes
