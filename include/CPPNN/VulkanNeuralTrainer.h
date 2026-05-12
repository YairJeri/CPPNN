#pragma once

#include "VulkanGPUCompute.h"
#include "TensorDevice.h"
#include "enums.h"
#include "ModelDevice.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <random>

namespace NN
{
    struct PushConstantsDense {
        uint32_t input_size;
        uint32_t output_size;
        uint32_t batch_size;
    };

    struct PushConstantsOptimizer {
        uint32_t count;
        float learning_rate;
    };

    struct PushConstantsAdam {
        uint32_t count;
        float lr;
        float beta1;
        float beta2;
        float eps;
        float beta1_pow;
        float beta2_pow;
    };

    struct PushConstantsMomentum {
        uint32_t count;
        float lr;
        float momentum;
    };

    struct PushConstantsLoss {
        uint32_t total_elements;
        uint32_t batch_size;
    };

    struct PushConstantsDropout {
        uint32_t count;
        float rate;
        uint32_t seed;
    };

    struct PushConstantsConv2D {
        uint32_t batch_size;
        uint32_t in_c;
        uint32_t in_h;
        uint32_t in_w;
        uint32_t out_c;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t k_size;
        uint32_t stride;
        uint32_t pad;
    };

    struct PushConstantsMaxPool2D {
        uint32_t batch_size;
        uint32_t c;
        uint32_t in_h;
        uint32_t in_w;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t pool_size;
        uint32_t stride;
    };

    struct PushConstantsMaxPool2DBack {
        uint32_t batch_size;
        uint32_t c;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t total_in_elements;
    };

    struct PushConstantsRescaling {
        uint32_t count;
        float scale;
        float offset;
    };

    enum class GPULayerKind {
        Rescaling,
        Conv2D,
        MaxPooling2D,
        Dropout,
        Flatten,
        Dense
    };

    struct GPULayer {
        GPULayerKind kind;
        Activation activation = Activation::Linear;

        // Formas
        int in_c = 1, in_h = 1, in_w = 1, in_size = 0;
        int out_c = 1, out_h = 1, out_w = 1, out_size = 0;

        // Hiperparámetros
        int kernel_size = 3;
        int stride = 1;
        int padding = 0;
        float dropout_rate = 0.0f;
        float scale = 1.0f;
        float offset = 0.0f;

        // Buffers para parámetros
        GPUBuffer buf_w;
        GPUBuffer buf_b;
        GPUBuffer buf_grad_w;
        GPUBuffer buf_grad_b;
        GPUBuffer buf_m_w, buf_v_w;
        GPUBuffer buf_m_b, buf_v_b;

        // Buffer auxiliar (ej. mask en dropout o indices en maxpool)
        GPUBuffer buf_aux;

        // Descriptor sets
        VkDescriptorSet desc_set_fwd = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_bwd_param = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_bwd_in = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_opt_w = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_opt_b = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_opt_adam_w = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_opt_adam_b = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_opt_mom_w = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_opt_mom_b = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_relu_fwd = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_relu_bwd = VK_NULL_HANDLE;
    };

    class VulkanNeuralTrainer {
    private:
        VulkanGPUCompute gpu;

        // Pipelines y Layouts
        VkDescriptorSetLayout desc_layout_dense_fwd = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_dense_fwd = VK_NULL_HANDLE;
        VkPipeline pipe_dense_fwd = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_dense_pg = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_dense_pg = VK_NULL_HANDLE;
        VkPipeline pipe_dense_pg = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_dense_ig = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_dense_ig = VK_NULL_HANDLE;
        VkPipeline pipe_dense_ig = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_relu = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_relu = VK_NULL_HANDLE;
        VkPipeline pipe_relu_fwd = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_relu_bwd = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_relu_bwd = VK_NULL_HANDLE;
        VkPipeline pipe_relu_bwd = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_opt = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_opt = VK_NULL_HANDLE;
        VkPipeline pipe_opt = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_opt_adam = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_opt_adam = VK_NULL_HANDLE;
        VkPipeline pipe_opt_adam = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_opt_mom = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_opt_mom = VK_NULL_HANDLE;
        VkPipeline pipe_opt_mom = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_loss = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_loss = VK_NULL_HANDLE;
        VkPipeline pipe_loss = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_loss = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_softmax = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_softmax = VK_NULL_HANDLE;
        VkPipeline pipe_softmax = VK_NULL_HANDLE;
        VkDescriptorSet desc_set_softmax = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_conv_fwd = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_conv_fwd = VK_NULL_HANDLE;
        VkPipeline pipe_conv_fwd = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_conv_pg = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_conv_pg = VK_NULL_HANDLE;
        VkPipeline pipe_conv_pg = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_conv_ig = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_conv_ig = VK_NULL_HANDLE;
        VkPipeline pipe_conv_ig = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_maxpool_fwd = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_maxpool_fwd = VK_NULL_HANDLE;
        VkPipeline pipe_maxpool_fwd = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_maxpool_bwd = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_maxpool_bwd = VK_NULL_HANDLE;
        VkPipeline pipe_maxpool_bwd = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_rescale_fwd = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_rescale_fwd = VK_NULL_HANDLE;
        VkPipeline pipe_rescale_fwd = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_rescale_bwd = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_rescale_bwd = VK_NULL_HANDLE;
        VkPipeline pipe_rescale_bwd = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_dropout_fwd = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_dropout_fwd = VK_NULL_HANDLE;
        VkPipeline pipe_dropout_fwd = VK_NULL_HANDLE;

        VkDescriptorSetLayout desc_layout_dropout_bwd = VK_NULL_HANDLE;
        VkPipelineLayout pipe_layout_dropout_bwd = VK_NULL_HANDLE;
        VkPipeline pipe_dropout_bwd = VK_NULL_HANDLE;

        std::vector<GPULayer> layers;
        std::vector<GPUBuffer> activations; // [0] entrada, [i+1] salida de la capa i
        std::vector<GPUBuffer> deltas;      // [0] gradiente de entrada, [i+1] delta de salida de la capa i
        GPUBuffer buf_target_onehot;

        VkCommandBuffer cmd_train = VK_NULL_HANDLE;
        int batch_size = 64;
        bool initialized = false;
        Optimizer current_optimizer = Optimizer::Adam;
        uint32_t step_t = 0;

    public:
        VulkanNeuralTrainer() {}
        ~VulkanNeuralTrainer() { cleanup(); }

        void cleanup() {
            if (!initialized) return;
            vkQueueWaitIdle(gpu.getQueue());

            for (auto &l : layers) {
                if (l.buf_w.buffer) gpu.destroyBuffer(l.buf_w);
                if (l.buf_b.buffer) gpu.destroyBuffer(l.buf_b);
                if (l.buf_grad_w.buffer) gpu.destroyBuffer(l.buf_grad_w);
                if (l.buf_grad_b.buffer) gpu.destroyBuffer(l.buf_grad_b);
                if (l.buf_m_w.buffer) gpu.destroyBuffer(l.buf_m_w);
                if (l.buf_v_w.buffer) gpu.destroyBuffer(l.buf_v_w);
                if (l.buf_m_b.buffer) gpu.destroyBuffer(l.buf_m_b);
                if (l.buf_v_b.buffer) gpu.destroyBuffer(l.buf_v_b);
                if (l.buf_aux.buffer) gpu.destroyBuffer(l.buf_aux);
            }
            for (auto &b : activations) gpu.destroyBuffer(b);
            for (auto &b : deltas) gpu.destroyBuffer(b);
            gpu.destroyBuffer(buf_target_onehot);

            layers.clear();
            activations.clear();
            deltas.clear();

            auto dev = gpu.getDevice();
            auto destroyPipe = [&](VkPipeline &p, VkPipelineLayout &pl, VkDescriptorSetLayout &dl) {
                if (p) { vkDestroyPipeline(dev, p, nullptr); p = VK_NULL_HANDLE; }
                if (pl) { vkDestroyPipelineLayout(dev, pl, nullptr); pl = VK_NULL_HANDLE; }
                if (dl) { vkDestroyDescriptorSetLayout(dev, dl, nullptr); dl = VK_NULL_HANDLE; }
            };

            destroyPipe(pipe_dense_fwd, pipe_layout_dense_fwd, desc_layout_dense_fwd);
            destroyPipe(pipe_dense_pg, pipe_layout_dense_pg, desc_layout_dense_pg);
            destroyPipe(pipe_dense_ig, pipe_layout_dense_ig, desc_layout_dense_ig);
            destroyPipe(pipe_relu_fwd, pipe_layout_relu, desc_layout_relu);
            if (pipe_relu_bwd) vkDestroyPipeline(dev, pipe_relu_bwd, nullptr);
            if (pipe_layout_relu_bwd) vkDestroyPipelineLayout(dev, pipe_layout_relu_bwd, nullptr);
            if (desc_layout_relu_bwd) vkDestroyDescriptorSetLayout(dev, desc_layout_relu_bwd, nullptr);
            destroyPipe(pipe_opt, pipe_layout_opt, desc_layout_opt);
            destroyPipe(pipe_opt_adam, pipe_layout_opt_adam, desc_layout_opt_adam);
            destroyPipe(pipe_opt_mom, pipe_layout_opt_mom, desc_layout_opt_mom);
            destroyPipe(pipe_loss, pipe_layout_loss, desc_layout_loss);
            destroyPipe(pipe_softmax, pipe_layout_softmax, desc_layout_softmax);
            destroyPipe(pipe_conv_fwd, pipe_layout_conv_fwd, desc_layout_conv_fwd);
            destroyPipe(pipe_conv_pg, pipe_layout_conv_pg, desc_layout_conv_pg);
            destroyPipe(pipe_conv_ig, pipe_layout_conv_ig, desc_layout_conv_ig);
            destroyPipe(pipe_maxpool_fwd, pipe_layout_maxpool_fwd, desc_layout_maxpool_fwd);
            destroyPipe(pipe_maxpool_bwd, pipe_layout_maxpool_bwd, desc_layout_maxpool_bwd);
            destroyPipe(pipe_rescale_fwd, pipe_layout_rescale_fwd, desc_layout_rescale_fwd);
            destroyPipe(pipe_rescale_bwd, pipe_layout_rescale_bwd, desc_layout_rescale_bwd);
            destroyPipe(pipe_dropout_fwd, pipe_layout_dropout_fwd, desc_layout_dropout_fwd);
            destroyPipe(pipe_dropout_bwd, pipe_layout_dropout_bwd, desc_layout_dropout_bwd);

            initialized = false;
        }

        void build(const std::vector<LayerStruct> &defs, int batch_sz, int in_c, int in_h, int in_w,
                   WeightInit weight_init = WeightInit::HeNormal, Optimizer opt = Optimizer::Adam) {
            cleanup();
            this->batch_size = batch_sz;
            this->current_optimizer = opt;
            this->step_t = 0;

            createPipelines();

            int cur_c = in_c > 0 ? in_c : 1;
            int cur_h = in_h > 0 ? in_h : 28;
            int cur_w = in_w > 0 ? in_w : 28;
            int cur_size = cur_c * cur_h * cur_w;

            // Buffer de entrada (activations[0] y deltas[0])
            activations.push_back(gpu.createBuffer(batch_size * cur_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0));
            deltas.push_back(gpu.createBuffer(batch_size * cur_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0));

            auto rand_normal = []() -> float {
                float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
                float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
                return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265358979323846f * u2);
            };

            auto rand_uniform = [](float limit) -> float {
                return (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * limit;
            };

            for (size_t d = 0; d < defs.size(); ++d) {
                const auto &def = defs[d];
                if (def.type == Layer::Input) {
                    cur_size = def.output_size;
                    continue;
                }

                GPULayer gly;
                gly.activation = def.activation;
                gly.in_c = cur_c;
                gly.in_h = cur_h;
                gly.in_w = cur_w;
                gly.in_size = cur_size;

                if (def.type == Layer::Rescaling) {
                    gly.kind = GPULayerKind::Rescaling;
                    gly.scale = def.scale;
                    gly.offset = def.offset;
                    gly.out_c = cur_c;
                    gly.out_h = cur_h;
                    gly.out_w = cur_w;
                    gly.out_size = cur_size;
                }
                else if (def.type == Layer::Conv2D) {
                    gly.kind = GPULayerKind::Conv2D;
                    gly.kernel_size = def.kernel_size;
                    gly.stride = def.stride;
                    gly.padding = def.padding;
                    gly.out_c = def.out_channels;
                    gly.out_h = (cur_h + 2 * def.padding - def.kernel_size) / def.stride + 1;
                    gly.out_w = (cur_w + 2 * def.padding - def.kernel_size) / def.stride + 1;
                    gly.out_size = gly.out_c * gly.out_h * gly.out_w;

                    size_t w_sz = gly.out_c * gly.in_c * gly.kernel_size * gly.kernel_size;
                    size_t b_sz = gly.out_c;

                    gly.buf_w = gpu.createBuffer(w_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_b = gpu.createBuffer(b_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_grad_w = gpu.createBuffer(w_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_grad_b = gpu.createBuffer(b_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_m_w = gpu.createBuffer(w_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_v_w = gpu.createBuffer(w_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_m_b = gpu.createBuffer(b_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_v_b = gpu.createBuffer(b_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);

                    std::vector<float> zeros_w(w_sz, 0.0f), zeros_b(b_sz, 0.0f);
                    gpu.uploadToBuffer(gly.buf_m_w, zeros_w.data(), w_sz * sizeof(float));
                    gpu.uploadToBuffer(gly.buf_v_w, zeros_w.data(), w_sz * sizeof(float));
                    gpu.uploadToBuffer(gly.buf_m_b, zeros_b.data(), b_sz * sizeof(float));
                    gpu.uploadToBuffer(gly.buf_v_b, zeros_b.data(), b_sz * sizeof(float));

                    // He Normal para Conv
                    float fan_in = gly.in_c * gly.kernel_size * gly.kernel_size;
                    float stddev = std::sqrt(2.0f / fan_in);
                    std::vector<float> init_w(w_sz);
                    for (float &val : init_w) val = rand_normal() * stddev;
                    gpu.uploadToBuffer(gly.buf_w, init_w.data(), w_sz * sizeof(float));
                    gpu.uploadToBuffer(gly.buf_b, zeros_b.data(), b_sz * sizeof(float));

                    cur_c = gly.out_c;
                    cur_h = gly.out_h;
                    cur_w = gly.out_w;
                    cur_size = gly.out_size;
                }
                else if (def.type == Layer::MaxPooling2D) {
                    gly.kind = GPULayerKind::MaxPooling2D;
                    gly.kernel_size = def.kernel_size;
                    gly.stride = def.stride;
                    gly.out_c = cur_c;
                    gly.out_h = (cur_h - def.kernel_size) / def.stride + 1;
                    gly.out_w = (cur_w - def.kernel_size) / def.stride + 1;
                    gly.out_size = gly.out_c * gly.out_h * gly.out_w;

                    gly.buf_aux = gpu.createBuffer(batch_size * gly.out_size * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);

                    cur_h = gly.out_h;
                    cur_w = gly.out_w;
                    cur_size = gly.out_size;
                }
                else if (def.type == Layer::Flatten) {
                    gly.kind = GPULayerKind::Flatten;
                    gly.out_c = 1;
                    gly.out_h = 1;
                    gly.out_w = cur_size;
                    gly.out_size = cur_size;

                    cur_c = 1;
                    cur_h = 1;
                    cur_w = cur_size;
                }
                else if (def.type == Layer::Dropout) {
                    gly.kind = GPULayerKind::Dropout;
                    gly.dropout_rate = def.dropout_rate;
                    gly.out_c = cur_c;
                    gly.out_h = cur_h;
                    gly.out_w = cur_w;
                    gly.out_size = cur_size;

                    gly.buf_aux = gpu.createBuffer(batch_size * cur_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                }
                else if (def.type == Layer::Dense) {
                    gly.kind = GPULayerKind::Dense;
                    gly.out_c = 1;
                    gly.out_h = 1;
                    gly.out_w = def.output_size;
                    gly.out_size = def.output_size;

                    int in_d = cur_size;
                    int out_d = def.output_size;
                    size_t w_sz = in_d * out_d;
                    size_t b_sz = out_d;

                    gly.buf_w = gpu.createBuffer(w_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_b = gpu.createBuffer(b_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_grad_w = gpu.createBuffer(w_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_grad_b = gpu.createBuffer(b_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_m_w = gpu.createBuffer(w_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_v_w = gpu.createBuffer(w_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_m_b = gpu.createBuffer(b_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);
                    gly.buf_v_b = gpu.createBuffer(b_sz * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);

                    std::vector<float> zeros_w(w_sz, 0.0f), zeros_b(b_sz, 0.0f);
                    gpu.uploadToBuffer(gly.buf_m_w, zeros_w.data(), w_sz * sizeof(float));
                    gpu.uploadToBuffer(gly.buf_v_w, zeros_w.data(), w_sz * sizeof(float));
                    gpu.uploadToBuffer(gly.buf_m_b, zeros_b.data(), b_sz * sizeof(float));
                    gpu.uploadToBuffer(gly.buf_v_b, zeros_b.data(), b_sz * sizeof(float));

                    std::vector<float> init_w(w_sz);
                    switch (weight_init) {
                    case WeightInit::HeNormal: {
                        float stddev = std::sqrt(2.0f / (float)in_d);
                        for (float &val : init_w) val = rand_normal() * stddev;
                        break;
                    }
                    case WeightInit::HeUniform: {
                        float limit = std::sqrt(6.0f / (float)in_d);
                        for (float &val : init_w) val = rand_uniform(limit);
                        break;
                    }
                    case WeightInit::GlorotUniform: {
                        float limit = std::sqrt(6.0f / (float)(in_d + out_d));
                        for (float &val : init_w) val = rand_uniform(limit);
                        break;
                    }
                    case WeightInit::GlorotNormal: {
                        float stddev = std::sqrt(2.0f / (float)(in_d + out_d));
                        for (float &val : init_w) val = rand_normal() * stddev;
                        break;
                    }
                    default: {
                        for (float &val : init_w) val = rand_uniform(0.05f);
                        break;
                    }
                    }
                    gpu.uploadToBuffer(gly.buf_w, init_w.data(), w_sz * sizeof(float));
                    gpu.uploadToBuffer(gly.buf_b, zeros_b.data(), b_sz * sizeof(float));

                    cur_c = 1;
                    cur_h = 1;
                    cur_w = out_d;
                    cur_size = out_d;
                }

                // Buffer de salida de esta capa
                activations.push_back(gpu.createBuffer(batch_size * gly.out_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0));
                deltas.push_back(gpu.createBuffer(batch_size * gly.out_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0));

                layers.push_back(gly);
                bindDescriptors((int)layers.size() - 1);
            }

            int final_out_size = layers.back().out_size;
            buf_target_onehot = gpu.createBuffer(batch_size * final_out_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0);

            // Bind Loss Descriptor Set (Output, Target, Delta_out)
            auto dev = gpu.getDevice();
            auto pool = gpu.getDescriptorPool();
            VkDescriptorSetAllocateInfo ai_loss{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1, &desc_layout_loss};
            vkAllocateDescriptorSets(dev, &ai_loss, &desc_set_loss);

            int last = (int)layers.size();
            VkDescriptorBufferInfo loss_infos[3] = {
                {activations[last].buffer, 0, VK_WHOLE_SIZE},
                {buf_target_onehot.buffer, 0, VK_WHOLE_SIZE},
                {deltas[last].buffer, 0, VK_WHOLE_SIZE}
            };
            VkWriteDescriptorSet loss_writes[3]{};
            for (int i = 0; i < 3; ++i) {
                loss_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                loss_writes[i].dstSet = desc_set_loss;
                loss_writes[i].dstBinding = i;
                loss_writes[i].descriptorCount = 1;
                loss_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                loss_writes[i].pBufferInfo = &loss_infos[i];
            }
            vkUpdateDescriptorSets(dev, 3, loss_writes, 0, nullptr);

            // Bind Softmax Descriptor Set
            VkDescriptorSetAllocateInfo ai_sm{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1, &desc_layout_softmax};
            vkAllocateDescriptorSets(dev, &ai_sm, &desc_set_softmax);
            VkDescriptorBufferInfo sm_info{activations[last].buffer, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet sm_write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, desc_set_softmax, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sm_info, nullptr};
            vkUpdateDescriptorSets(dev, 1, &sm_write, 0, nullptr);

            gpu.createCommandBuffer(cmd_train);
            initialized = true;
        }

    private:
        void createPipelines() {
            auto dev = gpu.getDevice();

            // 1. Dense Forward
            {
                std::vector<VkDescriptorSetLayoutBinding> b = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_dense_fwd = gpu.createDescriptorSetLayout(b);
                VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsDense)};
                VkPipelineLayoutCreateInfo linfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_dense_fwd, 1, &pr};
                vkCreatePipelineLayout(dev, &linfo, nullptr, &pipe_layout_dense_fwd);
                auto spv = gpu.loadShader("dense_forward.comp.spv");
                pipe_dense_fwd = createComputePipe(spv, pipe_layout_dense_fwd);
            }

            // 2. Dense Param Grads
            {
                std::vector<VkDescriptorSetLayoutBinding> b = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_dense_pg = gpu.createDescriptorSetLayout(b);
                VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsDense)};
                VkPipelineLayoutCreateInfo linfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_dense_pg, 1, &pr};
                vkCreatePipelineLayout(dev, &linfo, nullptr, &pipe_layout_dense_pg);
                auto spv = gpu.loadShader("dense_backward_param_grads.comp.spv");
                pipe_dense_pg = createComputePipe(spv, pipe_layout_dense_pg);
            }

            // 3. Dense Input Grad
            {
                std::vector<VkDescriptorSetLayoutBinding> b = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_dense_ig = gpu.createDescriptorSetLayout(b);
                VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsDense)};
                VkPipelineLayoutCreateInfo linfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_dense_ig, 1, &pr};
                vkCreatePipelineLayout(dev, &linfo, nullptr, &pipe_layout_dense_ig);
                auto spv = gpu.loadShader("dense_backward_input_grad.comp.spv");
                pipe_dense_ig = createComputePipe(spv, pipe_layout_dense_ig);
            }

            // 4. ReLU Fwd & Bwd
            {
                std::vector<VkDescriptorSetLayoutBinding> b_fwd = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_relu = gpu.createDescriptorSetLayout(b_fwd);
                VkPipelineLayoutCreateInfo linfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_relu, 0, nullptr};
                vkCreatePipelineLayout(dev, &linfo, nullptr, &pipe_layout_relu);
                auto spv_fwd = gpu.loadShader("activation_relu_forward.comp.spv");
                pipe_relu_fwd = createComputePipe(spv_fwd, pipe_layout_relu);

                std::vector<VkDescriptorSetLayoutBinding> b_bwd = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_relu_bwd = gpu.createDescriptorSetLayout(b_bwd);
                VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
                VkPipelineLayoutCreateInfo linfo_bwd{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_relu_bwd, 1, &pr};
                vkCreatePipelineLayout(dev, &linfo_bwd, nullptr, &pipe_layout_relu_bwd);
                auto spv_bwd = gpu.loadShader("activation_relu_backward.comp.spv");
                pipe_relu_bwd = createComputePipe(spv_bwd, pipe_layout_relu_bwd);
            }

            // 5. Optimizer Pipelines (SGD, Adam, SGDMomentum)
            {
                std::vector<VkDescriptorSetLayoutBinding> b = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_opt = gpu.createDescriptorSetLayout(b);
                VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsOptimizer)};
                VkPipelineLayoutCreateInfo linfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_opt, 1, &pr};
                vkCreatePipelineLayout(dev, &linfo, nullptr, &pipe_layout_opt);
                auto spv = gpu.loadShader("optimizer_sgd.comp.spv");
                pipe_opt = createComputePipe(spv, pipe_layout_opt);

                std::vector<VkDescriptorSetLayoutBinding> b_adam = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_opt_adam = gpu.createDescriptorSetLayout(b_adam);
                VkPushConstantRange pr_adam{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsAdam)};
                VkPipelineLayoutCreateInfo linfo_adam{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_opt_adam, 1, &pr_adam};
                vkCreatePipelineLayout(dev, &linfo_adam, nullptr, &pipe_layout_opt_adam);
                auto spv_adam = gpu.loadShader("optimizer_adam.comp.spv");
                pipe_opt_adam = createComputePipe(spv_adam, pipe_layout_opt_adam);

                std::vector<VkDescriptorSetLayoutBinding> b_mom = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_opt_mom = gpu.createDescriptorSetLayout(b_mom);
                VkPushConstantRange pr_mom{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsMomentum)};
                VkPipelineLayoutCreateInfo linfo_mom{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_opt_mom, 1, &pr_mom};
                vkCreatePipelineLayout(dev, &linfo_mom, nullptr, &pipe_layout_opt_mom);
                auto spv_mom = gpu.loadShader("optimizer_momentum.comp.spv");
                pipe_opt_mom = createComputePipe(spv_mom, pipe_layout_opt_mom);
            }

            // 6. Loss Cross Entropy
            {
                std::vector<VkDescriptorSetLayoutBinding> b = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_loss = gpu.createDescriptorSetLayout(b);
                VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsLoss)};
                VkPipelineLayoutCreateInfo linfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_loss, 1, &pr};
                vkCreatePipelineLayout(dev, &linfo, nullptr, &pipe_layout_loss);
                auto spv = gpu.loadShader("loss_cross_entropy.comp.spv");
                pipe_loss = createComputePipe(spv, pipe_layout_loss);
            }

            // 7. Softmax
            {
                std::vector<VkDescriptorSetLayoutBinding> b = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_softmax = gpu.createDescriptorSetLayout(b);
                VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t)};
                VkPipelineLayoutCreateInfo linfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_softmax, 1, &pr};
                vkCreatePipelineLayout(dev, &linfo, nullptr, &pipe_layout_softmax);
                auto spv = gpu.loadShader("activation_softmax.comp.spv");
                pipe_softmax = createComputePipe(spv, pipe_layout_softmax);
            }

            // 8. Dropout
            {
                std::vector<VkDescriptorSetLayoutBinding> b_fwd = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_dropout_fwd = gpu.createDescriptorSetLayout(b_fwd);
                VkPushConstantRange pr_fwd{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsDropout)};
                VkPipelineLayoutCreateInfo linfo_fwd{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_dropout_fwd, 1, &pr_fwd};
                vkCreatePipelineLayout(dev, &linfo_fwd, nullptr, &pipe_layout_dropout_fwd);
                auto spv_fwd = gpu.loadShader("dropout_forward.comp.spv");
                pipe_dropout_fwd = createComputePipe(spv_fwd, pipe_layout_dropout_fwd);

                desc_layout_dropout_bwd = gpu.createDescriptorSetLayout(b_fwd);
                VkPushConstantRange pr_bwd{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
                VkPipelineLayoutCreateInfo linfo_bwd{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_dropout_bwd, 1, &pr_bwd};
                vkCreatePipelineLayout(dev, &linfo_bwd, nullptr, &pipe_layout_dropout_bwd);
                auto spv_bwd = gpu.loadShader("dropout_backward.comp.spv");
                pipe_dropout_bwd = createComputePipe(spv_bwd, pipe_layout_dropout_bwd);
            }

            // 9. Conv2D
            {
                std::vector<VkDescriptorSetLayoutBinding> b_fwd = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_conv_fwd = gpu.createDescriptorSetLayout(b_fwd);
                VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsConv2D)};
                VkPipelineLayoutCreateInfo linfo_fwd{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_conv_fwd, 1, &pr};
                vkCreatePipelineLayout(dev, &linfo_fwd, nullptr, &pipe_layout_conv_fwd);
                auto spv_fwd = gpu.loadShader("conv2d_forward.comp.spv");
                pipe_conv_fwd = createComputePipe(spv_fwd, pipe_layout_conv_fwd);

                desc_layout_conv_pg = gpu.createDescriptorSetLayout(b_fwd);
                VkPipelineLayoutCreateInfo linfo_pg{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_conv_pg, 1, &pr};
                vkCreatePipelineLayout(dev, &linfo_pg, nullptr, &pipe_layout_conv_pg);
                auto spv_pg = gpu.loadShader("conv2d_backward_param_grads.comp.spv");
                pipe_conv_pg = createComputePipe(spv_pg, pipe_layout_conv_pg);

                std::vector<VkDescriptorSetLayoutBinding> b_ig = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_conv_ig = gpu.createDescriptorSetLayout(b_ig);
                VkPipelineLayoutCreateInfo linfo_ig{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_conv_ig, 1, &pr};
                vkCreatePipelineLayout(dev, &linfo_ig, nullptr, &pipe_layout_conv_ig);
                auto spv_ig = gpu.loadShader("conv2d_backward_input_grad.comp.spv");
                pipe_conv_ig = createComputePipe(spv_ig, pipe_layout_conv_ig);
            }

            // 10. MaxPool2D
            {
                std::vector<VkDescriptorSetLayoutBinding> b_fwd = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_maxpool_fwd = gpu.createDescriptorSetLayout(b_fwd);
                VkPushConstantRange pr_fwd{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsMaxPool2D)};
                VkPipelineLayoutCreateInfo linfo_fwd{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_maxpool_fwd, 1, &pr_fwd};
                vkCreatePipelineLayout(dev, &linfo_fwd, nullptr, &pipe_layout_maxpool_fwd);
                auto spv_fwd = gpu.loadShader("maxpool2d_forward.comp.spv");
                pipe_maxpool_fwd = createComputePipe(spv_fwd, pipe_layout_maxpool_fwd);

                std::vector<VkDescriptorSetLayoutBinding> b_bwd = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_maxpool_bwd = gpu.createDescriptorSetLayout(b_bwd);
                VkPushConstantRange pr_bwd{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsMaxPool2DBack)};
                VkPipelineLayoutCreateInfo linfo_bwd{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_maxpool_bwd, 1, &pr_bwd};
                vkCreatePipelineLayout(dev, &linfo_bwd, nullptr, &pipe_layout_maxpool_bwd);
                auto spv_bwd = gpu.loadShader("maxpool2d_backward.comp.spv");
                pipe_maxpool_bwd = createComputePipe(spv_bwd, pipe_layout_maxpool_bwd);
            }

            // 11. Rescaling
            {
                std::vector<VkDescriptorSetLayoutBinding> b_fwd = {
                    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                };
                desc_layout_rescale_fwd = gpu.createDescriptorSetLayout(b_fwd);
                VkPushConstantRange pr_fwd{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRescaling)};
                VkPipelineLayoutCreateInfo linfo_fwd{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_rescale_fwd, 1, &pr_fwd};
                vkCreatePipelineLayout(dev, &linfo_fwd, nullptr, &pipe_layout_rescale_fwd);
                auto spv_fwd = gpu.loadShader("rescaling_forward.comp.spv");
                pipe_rescale_fwd = createComputePipe(spv_fwd, pipe_layout_rescale_fwd);

                desc_layout_rescale_bwd = gpu.createDescriptorSetLayout(b_fwd);
                VkPushConstantRange pr_bwd{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) + sizeof(float)};
                VkPipelineLayoutCreateInfo linfo_bwd{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &desc_layout_rescale_bwd, 1, &pr_bwd};
                vkCreatePipelineLayout(dev, &linfo_bwd, nullptr, &pipe_layout_rescale_bwd);
                auto spv_bwd = gpu.loadShader("rescaling_backward.comp.spv");
                pipe_rescale_bwd = createComputePipe(spv_bwd, pipe_layout_rescale_bwd);
            }
        }

        VkPipeline createComputePipe(const std::vector<uint32_t> &spv, VkPipelineLayout layout) {
            auto dev = gpu.getDevice();
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv.size() * sizeof(uint32_t), spv.data()};
            VkShaderModule module;
            vkCreateShaderModule(dev, &smi, nullptr, &module);

            VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pci.stage.module = module;
            pci.stage.pName = "main";
            pci.layout = layout;

            VkPipeline pipe;
            vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe);
            vkDestroyShaderModule(dev, module, nullptr);
            return pipe;
        }

        void bindDescriptors(int idx) {
            auto &l = layers[idx];
            auto dev = gpu.getDevice();
            auto pool = gpu.getDescriptorPool();

            auto alloc_set = [&](VkDescriptorSetLayout layout, VkDescriptorSet &s) {
                VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1, &layout};
                vkAllocateDescriptorSets(dev, &ai, &s);
            };

            auto update_buf = [&](VkDescriptorSet set, uint32_t binding, VkBuffer buf) {
                VkDescriptorBufferInfo info{buf, 0, VK_WHOLE_SIZE};
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info, nullptr};
                vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
            };

            if (l.kind == GPULayerKind::Rescaling) {
                alloc_set(desc_layout_rescale_fwd, l.desc_set_fwd);
                update_buf(l.desc_set_fwd, 0, activations[idx].buffer);
                update_buf(l.desc_set_fwd, 1, activations[idx + 1].buffer);

                alloc_set(desc_layout_rescale_bwd, l.desc_set_bwd_in);
                update_buf(l.desc_set_bwd_in, 0, deltas[idx + 1].buffer);
                update_buf(l.desc_set_bwd_in, 1, deltas[idx].buffer);
            }
            else if (l.kind == GPULayerKind::Conv2D) {
                // Forward: 0: in, 1: w, 2: b, 3: out
                alloc_set(desc_layout_conv_fwd, l.desc_set_fwd);
                update_buf(l.desc_set_fwd, 0, activations[idx].buffer);
                update_buf(l.desc_set_fwd, 1, l.buf_w.buffer);
                update_buf(l.desc_set_fwd, 2, l.buf_b.buffer);
                update_buf(l.desc_set_fwd, 3, activations[idx + 1].buffer);

                // Param grads: 0: in_cache, 1: delta_out, 2: grad_w, 3: grad_b
                alloc_set(desc_layout_conv_pg, l.desc_set_bwd_param);
                update_buf(l.desc_set_bwd_param, 0, activations[idx].buffer);
                update_buf(l.desc_set_bwd_param, 1, deltas[idx + 1].buffer);
                update_buf(l.desc_set_bwd_param, 2, l.buf_grad_w.buffer);
                update_buf(l.desc_set_bwd_param, 3, l.buf_grad_b.buffer);

                // Input grad: 0: delta_out, 1: w, 2: grad_in
                alloc_set(desc_layout_conv_ig, l.desc_set_bwd_in);
                update_buf(l.desc_set_bwd_in, 0, deltas[idx + 1].buffer);
                update_buf(l.desc_set_bwd_in, 1, l.buf_w.buffer);
                update_buf(l.desc_set_bwd_in, 2, deltas[idx].buffer);

                bindOptimizers(l);
            }
            else if (l.kind == GPULayerKind::MaxPooling2D) {
                // Forward: 0: in, 1: out, 2: mask_idx
                alloc_set(desc_layout_maxpool_fwd, l.desc_set_fwd);
                update_buf(l.desc_set_fwd, 0, activations[idx].buffer);
                update_buf(l.desc_set_fwd, 1, activations[idx + 1].buffer);
                update_buf(l.desc_set_fwd, 2, l.buf_aux.buffer);

                // Backward: 0: delta_out, 1: mask_idx, 2: grad_in
                alloc_set(desc_layout_maxpool_bwd, l.desc_set_bwd_in);
                update_buf(l.desc_set_bwd_in, 0, deltas[idx + 1].buffer);
                update_buf(l.desc_set_bwd_in, 1, l.buf_aux.buffer);
                update_buf(l.desc_set_bwd_in, 2, deltas[idx].buffer);
            }
            else if (l.kind == GPULayerKind::Dropout) {
                alloc_set(desc_layout_dropout_fwd, l.desc_set_fwd);
                update_buf(l.desc_set_fwd, 0, activations[idx].buffer);
                update_buf(l.desc_set_fwd, 1, activations[idx + 1].buffer);
                update_buf(l.desc_set_fwd, 2, l.buf_aux.buffer);

                alloc_set(desc_layout_dropout_bwd, l.desc_set_bwd_in);
                update_buf(l.desc_set_bwd_in, 0, deltas[idx + 1].buffer);
                update_buf(l.desc_set_bwd_in, 1, deltas[idx].buffer);
                update_buf(l.desc_set_bwd_in, 2, l.buf_aux.buffer);
            }
            else if (l.kind == GPULayerKind::Dense) {
                // Forward: 0: in, 1: out, 2: w, 3: b
                alloc_set(desc_layout_dense_fwd, l.desc_set_fwd);
                update_buf(l.desc_set_fwd, 0, activations[idx].buffer);
                update_buf(l.desc_set_fwd, 1, activations[idx + 1].buffer);
                update_buf(l.desc_set_fwd, 2, l.buf_w.buffer);
                update_buf(l.desc_set_fwd, 3, l.buf_b.buffer);

                // Param grads: 0: in_cache, 1: delta_out, 2: grad_w, 3: grad_b
                alloc_set(desc_layout_dense_pg, l.desc_set_bwd_param);
                update_buf(l.desc_set_bwd_param, 0, activations[idx].buffer);
                update_buf(l.desc_set_bwd_param, 1, deltas[idx + 1].buffer);
                update_buf(l.desc_set_bwd_param, 2, l.buf_grad_w.buffer);
                update_buf(l.desc_set_bwd_param, 3, l.buf_grad_b.buffer);

                // Input grad: 0: delta_out, 1: w, 2: grad_in
                alloc_set(desc_layout_dense_ig, l.desc_set_bwd_in);
                update_buf(l.desc_set_bwd_in, 0, deltas[idx + 1].buffer);
                update_buf(l.desc_set_bwd_in, 1, l.buf_w.buffer);
                update_buf(l.desc_set_bwd_in, 2, deltas[idx].buffer);

                bindOptimizers(l);
            }
            else if (l.kind == GPULayerKind::Flatten) {
                // Flatten es passthrough en memoria física
                // activations[idx+1] se copia de activations[idx]
            }

            // Activación ReLU si aplica
            if (l.activation == Activation::ReLU) {
                alloc_set(desc_layout_relu, l.desc_set_relu_fwd);
                update_buf(l.desc_set_relu_fwd, 0, activations[idx + 1].buffer);

                alloc_set(desc_layout_relu_bwd, l.desc_set_relu_bwd);
                update_buf(l.desc_set_relu_bwd, 0, deltas[idx + 1].buffer);
                update_buf(l.desc_set_relu_bwd, 1, activations[idx + 1].buffer);
            }
        }

        void bindOptimizers(GPULayer &l) {
            auto dev = gpu.getDevice();
            auto pool = gpu.getDescriptorPool();

            auto alloc_set = [&](VkDescriptorSetLayout layout, VkDescriptorSet &s) {
                VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1, &layout};
                vkAllocateDescriptorSets(dev, &ai, &s);
            };
            auto update_buf = [&](VkDescriptorSet set, uint32_t binding, VkBuffer buf) {
                VkDescriptorBufferInfo info{buf, 0, VK_WHOLE_SIZE};
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info, nullptr};
                vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
            };

            // SGD
            alloc_set(desc_layout_opt, l.desc_set_opt_w);
            update_buf(l.desc_set_opt_w, 0, l.buf_w.buffer);
            update_buf(l.desc_set_opt_w, 1, l.buf_grad_w.buffer);

            alloc_set(desc_layout_opt, l.desc_set_opt_b);
            update_buf(l.desc_set_opt_b, 0, l.buf_b.buffer);
            update_buf(l.desc_set_opt_b, 1, l.buf_grad_b.buffer);

            // Adam
            alloc_set(desc_layout_opt_adam, l.desc_set_opt_adam_w);
            update_buf(l.desc_set_opt_adam_w, 0, l.buf_w.buffer);
            update_buf(l.desc_set_opt_adam_w, 1, l.buf_grad_w.buffer);
            update_buf(l.desc_set_opt_adam_w, 2, l.buf_m_w.buffer);
            update_buf(l.desc_set_opt_adam_w, 3, l.buf_v_w.buffer);

            alloc_set(desc_layout_opt_adam, l.desc_set_opt_adam_b);
            update_buf(l.desc_set_opt_adam_b, 0, l.buf_b.buffer);
            update_buf(l.desc_set_opt_adam_b, 1, l.buf_grad_b.buffer);
            update_buf(l.desc_set_opt_adam_b, 2, l.buf_m_b.buffer);
            update_buf(l.desc_set_opt_adam_b, 3, l.buf_v_b.buffer);

            // SGDMomentum
            alloc_set(desc_layout_opt_mom, l.desc_set_opt_mom_w);
            update_buf(l.desc_set_opt_mom_w, 0, l.buf_w.buffer);
            update_buf(l.desc_set_opt_mom_w, 1, l.buf_grad_w.buffer);
            update_buf(l.desc_set_opt_mom_w, 2, l.buf_m_w.buffer);

            alloc_set(desc_layout_opt_mom, l.desc_set_opt_mom_b);
            update_buf(l.desc_set_opt_mom_b, 0, l.buf_b.buffer);
            update_buf(l.desc_set_opt_mom_b, 1, l.buf_grad_b.buffer);
            update_buf(l.desc_set_opt_mom_b, 2, l.buf_m_b.buffer);
        }

    public:
        void trainBatchGPU(const float *host_x, const float *host_y, float lr) {
            if (!initialized) return;

            // 1. Upload Batch Input y Target a la GPU
            gpu.uploadToBuffer(activations[0], (void*)host_x, batch_size * layers[0].in_size * sizeof(float));
            int last = (int)layers.size();
            gpu.uploadToBuffer(buf_target_onehot, (void*)host_y, batch_size * layers.back().out_size * sizeof(float));

            // 2. Grabar Command Buffer
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            vkBeginCommandBuffer(cmd_train, &bi);

            auto barrier = [&]() {
                VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
                vkCmdPipelineBarrier(cmd_train, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            };

            // ==========================================
            // FORWARD PASS
            // ==========================================
            for (size_t i = 0; i < layers.size(); ++i) {
                auto &l = layers[i];
                if (l.kind == GPULayerKind::Rescaling) {
                    vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_rescale_fwd);
                    vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_rescale_fwd, 0, 1, &l.desc_set_fwd, 0, nullptr);
                    PushConstantsRescaling pcr{ (uint32_t)(batch_size * l.out_size), l.scale, l.offset };
                    vkCmdPushConstants(cmd_train, pipe_layout_rescale_fwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRescaling), &pcr);
                    vkCmdDispatch(cmd_train, (batch_size * l.out_size + 63) / 64, 1, 1);
                }
                else if (l.kind == GPULayerKind::Conv2D) {
                    vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_conv_fwd);
                    vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_conv_fwd, 0, 1, &l.desc_set_fwd, 0, nullptr);
                    PushConstantsConv2D pc{ (uint32_t)batch_size, (uint32_t)l.in_c, (uint32_t)l.in_h, (uint32_t)l.in_w,
                                           (uint32_t)l.out_c, (uint32_t)l.out_h, (uint32_t)l.out_w,
                                           (uint32_t)l.kernel_size, (uint32_t)l.stride, (uint32_t)l.padding };
                    vkCmdPushConstants(cmd_train, pipe_layout_conv_fwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsConv2D), &pc);
                    vkCmdDispatch(cmd_train, (l.out_w + 15) / 16, (l.out_h + 15) / 16, batch_size * l.out_c);
                }
                else if (l.kind == GPULayerKind::MaxPooling2D) {
                    vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_maxpool_fwd);
                    vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_maxpool_fwd, 0, 1, &l.desc_set_fwd, 0, nullptr);
                    PushConstantsMaxPool2D pc{ (uint32_t)batch_size, (uint32_t)l.in_c, (uint32_t)l.in_h, (uint32_t)l.in_w,
                                              (uint32_t)l.out_h, (uint32_t)l.out_w, (uint32_t)l.kernel_size, (uint32_t)l.stride };
                    vkCmdPushConstants(cmd_train, pipe_layout_maxpool_fwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsMaxPool2D), &pc);
                    vkCmdDispatch(cmd_train, (l.out_w + 15) / 16, (l.out_h + 15) / 16, batch_size * l.in_c);
                }
                else if (l.kind == GPULayerKind::Flatten) {
                    // Copiar memoria directa de activations[i] a activations[i+1]
                    VkBufferCopy copyRegion{0, 0, (VkDeviceSize)(batch_size * l.out_size * sizeof(float))};
                    vkCmdCopyBuffer(cmd_train, activations[i].buffer, activations[i + 1].buffer, 1, &copyRegion);
                }
                else if (l.kind == GPULayerKind::Dropout) {
                    vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dropout_fwd);
                    vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_dropout_fwd, 0, 1, &l.desc_set_fwd, 0, nullptr);
                    PushConstantsDropout pcd{ (uint32_t)(batch_size * l.out_size), l.dropout_rate, (uint32_t)rand() };
                    vkCmdPushConstants(cmd_train, pipe_layout_dropout_fwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsDropout), &pcd);
                    vkCmdDispatch(cmd_train, (batch_size * l.out_size + 63) / 64, 1, 1);
                }
                else if (l.kind == GPULayerKind::Dense) {
                    vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dense_fwd);
                    vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_dense_fwd, 0, 1, &l.desc_set_fwd, 0, nullptr);
                    PushConstantsDense pc{ (uint32_t)l.in_size, (uint32_t)l.out_size, (uint32_t)batch_size };
                    vkCmdPushConstants(cmd_train, pipe_layout_dense_fwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsDense), &pc);
                    vkCmdDispatch(cmd_train, batch_size * l.out_size, 1, 1);
                }

                barrier();

                // Activación si aplica
                if (l.activation == Activation::ReLU) {
                    vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_relu_fwd);
                    vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_relu, 0, 1, &l.desc_set_relu_fwd, 0, nullptr);
                    vkCmdDispatch(cmd_train, batch_size * l.out_size, 1, 1);
                    barrier();
                }
            }

            // Softmax en la última salida
            if (layers.back().activation == Activation::Softmax) {
                uint32_t sm_pc[2] = { (uint32_t)batch_size, (uint32_t)layers.back().out_size };
                vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_softmax);
                vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_softmax, 0, 1, &desc_set_softmax, 0, nullptr);
                vkCmdPushConstants(cmd_train, pipe_layout_softmax, VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t), sm_pc);
                vkCmdDispatch(cmd_train, (batch_size + 63) / 64, 1, 1);
                barrier();
            }

            // Loss: Delta = (output - target) / batch_size
            uint32_t total_out = batch_size * layers.back().out_size;
            vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_loss);
            vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_loss, 0, 1, &desc_set_loss, 0, nullptr);
            PushConstantsLoss pcl{ total_out, (uint32_t)batch_size };
            vkCmdPushConstants(cmd_train, pipe_layout_loss, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsLoss), &pcl);
            vkCmdDispatch(cmd_train, (total_out + 63) / 64, 1, 1);
            barrier();

            // ==========================================
            // BACKWARD PASS
            // ==========================================
            step_t++;
            float b1_pow = std::pow(0.9f, (float)step_t);
            float b2_pow = std::pow(0.999f, (float)step_t);

            for (int i = (int)layers.size() - 1; i >= 0; --i) {
                auto &l = layers[i];

                if (l.activation == Activation::ReLU) {
                    uint32_t cnt = batch_size * l.out_size;
                    vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_relu_bwd);
                    vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_relu_bwd, 0, 1, &l.desc_set_relu_bwd, 0, nullptr);
                    vkCmdPushConstants(cmd_train, pipe_layout_relu_bwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &cnt);
                    vkCmdDispatch(cmd_train, (cnt + 63) / 64, 1, 1);
                    barrier();
                }

                if (l.kind == GPULayerKind::Dense) {
                    // 1. Param Grads
                    vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dense_pg);
                    vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_dense_pg, 0, 1, &l.desc_set_bwd_param, 0, nullptr);
                    PushConstantsDense pc{ (uint32_t)l.in_size, (uint32_t)l.out_size, (uint32_t)batch_size };
                    vkCmdPushConstants(cmd_train, pipe_layout_dense_pg, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsDense), &pc);
                    vkCmdDispatch(cmd_train, (l.in_size + 15) / 16, (l.out_size + 15) / 16, 1);
                    barrier();

                    // 2. Input Grad
                    if (i > 0) {
                        vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dense_ig);
                        vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_dense_ig, 0, 1, &l.desc_set_bwd_in, 0, nullptr);
                        vkCmdPushConstants(cmd_train, pipe_layout_dense_ig, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsDense), &pc);
                        vkCmdDispatch(cmd_train, (batch_size + 15) / 16, (l.in_size + 15) / 16, 1);
                        barrier();
                    }

                    // 3. Optimizer Step
                    applyOptimizer(l, (uint32_t)(l.in_size * l.out_size), (uint32_t)l.out_size, lr, b1_pow, b2_pow);
                }
                else if (l.kind == GPULayerKind::Conv2D) {
                    // 1. Param Grads
                    PushConstantsConv2D pc{ (uint32_t)batch_size, (uint32_t)l.in_c, (uint32_t)l.in_h, (uint32_t)l.in_w,
                                           (uint32_t)l.out_c, (uint32_t)l.out_h, (uint32_t)l.out_w,
                                           (uint32_t)l.kernel_size, (uint32_t)l.stride, (uint32_t)l.padding };
                    vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_conv_pg);
                    vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_conv_pg, 0, 1, &l.desc_set_bwd_param, 0, nullptr);
                    vkCmdPushConstants(cmd_train, pipe_layout_conv_pg, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsConv2D), &pc);
                    vkCmdDispatch(cmd_train, (l.kernel_size * l.kernel_size + 15) / 16, (l.in_c * l.out_c + 15) / 16, 1);
                    barrier();

                    // 2. Input Grad
                    if (i > 0) {
                        vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_conv_ig);
                        vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_conv_ig, 0, 1, &l.desc_set_bwd_in, 0, nullptr);
                        vkCmdPushConstants(cmd_train, pipe_layout_conv_ig, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsConv2D), &pc);
                        vkCmdDispatch(cmd_train, (l.in_w + 15) / 16, (l.in_h + 15) / 16, batch_size * l.in_c);
                        barrier();
                    }

                    // 3. Optimizer Step
                    applyOptimizer(l, (uint32_t)(l.out_c * l.in_c * l.kernel_size * l.kernel_size), (uint32_t)l.out_c, lr, b1_pow, b2_pow);
                }
                else if (l.kind == GPULayerKind::MaxPooling2D) {
                    if (i > 0) {
                        vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_maxpool_bwd);
                        vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_maxpool_bwd, 0, 1, &l.desc_set_bwd_in, 0, nullptr);
                        PushConstantsMaxPool2DBack pc{ (uint32_t)batch_size, (uint32_t)l.in_c, (uint32_t)l.out_h, (uint32_t)l.out_w, (uint32_t)(batch_size * l.in_size) };
                        vkCmdPushConstants(cmd_train, pipe_layout_maxpool_bwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsMaxPool2DBack), &pc);
                        vkCmdDispatch(cmd_train, (l.out_w + 15) / 16, (l.out_h + 15) / 16, batch_size * l.in_c);
                        barrier();
                    }
                }
                else if (l.kind == GPULayerKind::Dropout) {
                    if (i > 0) {
                        vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dropout_bwd);
                        vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_dropout_bwd, 0, 1, &l.desc_set_bwd_in, 0, nullptr);
                        uint32_t cnt = batch_size * l.out_size;
                        vkCmdPushConstants(cmd_train, pipe_layout_dropout_bwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &cnt);
                        vkCmdDispatch(cmd_train, (cnt + 63) / 64, 1, 1);
                        barrier();
                    }
                }
                else if (l.kind == GPULayerKind::Flatten) {
                    if (i > 0) {
                        VkBufferCopy copyRegion{0, 0, (VkDeviceSize)(batch_size * l.out_size * sizeof(float))};
                        vkCmdCopyBuffer(cmd_train, deltas[i + 1].buffer, deltas[i].buffer, 1, &copyRegion);
                        barrier();
                    }
                }
                else if (l.kind == GPULayerKind::Rescaling) {
                    if (i > 0) {
                        vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_rescale_bwd);
                        vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_rescale_bwd, 0, 1, &l.desc_set_bwd_in, 0, nullptr);
                        uint32_t cnt = batch_size * l.out_size;
                        struct { uint32_t c; float s; } pc_res{ cnt, l.scale };
                        vkCmdPushConstants(cmd_train, pipe_layout_rescale_bwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_res), &pc_res);
                        vkCmdDispatch(cmd_train, (cnt + 63) / 64, 1, 1);
                        barrier();
                    }
                }
            }

            vkEndCommandBuffer(cmd_train);

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmd_train, 0, nullptr};
            vkQueueSubmit(gpu.getQueue(), 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(gpu.getQueue());
        }

    private:
        void applyOptimizer(GPULayer &l, uint32_t w_count, uint32_t b_count, float lr, float b1_pow, float b2_pow) {
            auto barrier = [&]() {
                VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
                vkCmdPipelineBarrier(cmd_train, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            };

            if (current_optimizer == Optimizer::Adam) {
                vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_opt_adam);
                // Update W
                vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_opt_adam, 0, 1, &l.desc_set_opt_adam_w, 0, nullptr);
                PushConstantsAdam pca_w{ w_count, lr, 0.9f, 0.999f, 1e-8f, b1_pow, b2_pow };
                vkCmdPushConstants(cmd_train, pipe_layout_opt_adam, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsAdam), &pca_w);
                vkCmdDispatch(cmd_train, (w_count + 63) / 64, 1, 1);

                // Update B
                vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_opt_adam, 0, 1, &l.desc_set_opt_adam_b, 0, nullptr);
                PushConstantsAdam pca_b{ b_count, lr, 0.9f, 0.999f, 1e-8f, b1_pow, b2_pow };
                vkCmdPushConstants(cmd_train, pipe_layout_opt_adam, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsAdam), &pca_b);
                vkCmdDispatch(cmd_train, (b_count + 63) / 64, 1, 1);
            }
            else if (current_optimizer == Optimizer::SGDMomentum) {
                vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_opt_mom);
                vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_opt_mom, 0, 1, &l.desc_set_opt_mom_w, 0, nullptr);
                PushConstantsMomentum pcm_w{ w_count, lr, 0.9f };
                vkCmdPushConstants(cmd_train, pipe_layout_opt_mom, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsMomentum), &pcm_w);
                vkCmdDispatch(cmd_train, (w_count + 63) / 64, 1, 1);

                vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_opt_mom, 0, 1, &l.desc_set_opt_mom_b, 0, nullptr);
                PushConstantsMomentum pcm_b{ b_count, lr, 0.9f };
                vkCmdPushConstants(cmd_train, pipe_layout_opt_mom, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsMomentum), &pcm_b);
                vkCmdDispatch(cmd_train, (b_count + 63) / 64, 1, 1);
            }
            else {
                vkCmdBindPipeline(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_opt);
                vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_opt, 0, 1, &l.desc_set_opt_w, 0, nullptr);
                PushConstantsOptimizer pc_w{ w_count, lr };
                vkCmdPushConstants(cmd_train, pipe_layout_opt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsOptimizer), &pc_w);
                vkCmdDispatch(cmd_train, (w_count + 63) / 64, 1, 1);

                vkCmdBindDescriptorSets(cmd_train, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_opt, 0, 1, &l.desc_set_opt_b, 0, nullptr);
                PushConstantsOptimizer pc_b{ b_count, lr };
                vkCmdPushConstants(cmd_train, pipe_layout_opt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsOptimizer), &pc_b);
                vkCmdDispatch(cmd_train, (b_count + 63) / 64, 1, 1);
            }
            barrier();
        }

    public:
        void predictGPU(const float *host_x, float *host_out, int total_samples) {
            if (!initialized || total_samples <= 0) return;

            int in_features = layers[0].in_size;
            int out_features = layers.back().out_size;

            VkCommandBuffer cmd;
            gpu.createCommandBuffer(cmd);

            for (int offset = 0; offset < total_samples; offset += batch_size) {
                int cur_batch = std::min(batch_size, total_samples - offset);

                gpu.uploadToBuffer(activations[0], (void*)(host_x + (size_t)offset * in_features), cur_batch * in_features * sizeof(float));

                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
                vkBeginCommandBuffer(cmd, &bi);

                auto barrier = [&]() {
                    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
                };

                for (size_t i = 0; i < layers.size(); ++i) {
                    auto &l = layers[i];
                    if (l.kind == GPULayerKind::Rescaling) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_rescale_fwd);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_rescale_fwd, 0, 1, &l.desc_set_fwd, 0, nullptr);
                        PushConstantsRescaling pcr{ (uint32_t)(cur_batch * l.out_size), l.scale, l.offset };
                        vkCmdPushConstants(cmd, pipe_layout_rescale_fwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRescaling), &pcr);
                        vkCmdDispatch(cmd, (cur_batch * l.out_size + 63) / 64, 1, 1);
                    }
                    else if (l.kind == GPULayerKind::Conv2D) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_conv_fwd);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_conv_fwd, 0, 1, &l.desc_set_fwd, 0, nullptr);
                        PushConstantsConv2D pc{ (uint32_t)cur_batch, (uint32_t)l.in_c, (uint32_t)l.in_h, (uint32_t)l.in_w,
                                               (uint32_t)l.out_c, (uint32_t)l.out_h, (uint32_t)l.out_w,
                                               (uint32_t)l.kernel_size, (uint32_t)l.stride, (uint32_t)l.padding };
                        vkCmdPushConstants(cmd, pipe_layout_conv_fwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsConv2D), &pc);
                        vkCmdDispatch(cmd, (l.out_w + 15) / 16, (l.out_h + 15) / 16, cur_batch * l.out_c);
                    }
                    else if (l.kind == GPULayerKind::MaxPooling2D) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_maxpool_fwd);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_maxpool_fwd, 0, 1, &l.desc_set_fwd, 0, nullptr);
                        PushConstantsMaxPool2D pc{ (uint32_t)cur_batch, (uint32_t)l.in_c, (uint32_t)l.in_h, (uint32_t)l.in_w,
                                                  (uint32_t)l.out_h, (uint32_t)l.out_w, (uint32_t)l.kernel_size, (uint32_t)l.stride };
                        vkCmdPushConstants(cmd, pipe_layout_maxpool_fwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsMaxPool2D), &pc);
                        vkCmdDispatch(cmd, (l.out_w + 15) / 16, (l.out_h + 15) / 16, cur_batch * l.in_c);
                    }
                    else if (l.kind == GPULayerKind::Flatten) {
                        VkBufferCopy copyRegion{0, 0, (VkDeviceSize)(cur_batch * l.out_size * sizeof(float))};
                        vkCmdCopyBuffer(cmd, activations[i].buffer, activations[i + 1].buffer, 1, &copyRegion);
                    }
                    else if (l.kind == GPULayerKind::Dense) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dense_fwd);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_dense_fwd, 0, 1, &l.desc_set_fwd, 0, nullptr);
                        PushConstantsDense pc{ (uint32_t)l.in_size, (uint32_t)l.out_size, (uint32_t)cur_batch };
                        vkCmdPushConstants(cmd, pipe_layout_dense_fwd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsDense), &pc);
                        vkCmdDispatch(cmd, cur_batch * l.out_size, 1, 1);
                    }

                    barrier();

                    if (l.activation == Activation::ReLU) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_relu_fwd);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_relu, 0, 1, &l.desc_set_relu_fwd, 0, nullptr);
                        vkCmdDispatch(cmd, cur_batch * l.out_size, 1, 1);
                        barrier();
                    }
                }

                // Softmax
                if (layers.back().activation == Activation::Softmax) {
                    uint32_t sm_pc[2] = { (uint32_t)cur_batch, (uint32_t)layers.back().out_size };
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_softmax);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_softmax, 0, 1, &desc_set_softmax, 0, nullptr);
                    vkCmdPushConstants(cmd, pipe_layout_softmax, VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t), sm_pc);
                    vkCmdDispatch(cmd, (cur_batch + 63) / 64, 1, 1);
                }

                VkMemoryBarrier mb_end{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb_end, 0, nullptr, 0, nullptr);

                vkEndCommandBuffer(cmd);
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr};
                vkQueueSubmit(gpu.getQueue(), 1, &si, VK_NULL_HANDLE);
                vkQueueWaitIdle(gpu.getQueue());

                gpu.downloadFromBuffer(activations.back(), host_out + (size_t)offset * out_features, cur_batch * out_features * sizeof(float));
            }
        }

        std::vector<std::vector<float>> getLayerActivations(const float *single_input) {
            std::vector<std::vector<float>> result;
            if (!initialized) return result;

            std::vector<float> dummy_out(layers.back().out_size);
            predictGPU(single_input, dummy_out.data(), 1);

            result.resize(layers.size() + 1);
            result[0].resize(layers[0].in_size);
            gpu.downloadFromBuffer(activations[0], result[0].data(), layers[0].in_size * sizeof(float));

            for (size_t i = 0; i < layers.size(); ++i) {
                int out_dim = layers[i].out_size;
                result[i + 1].resize(out_dim);
                gpu.downloadFromBuffer(activations[i + 1], result[i + 1].data(), out_dim * sizeof(float));
            }
            return result;
        }
    };
}
