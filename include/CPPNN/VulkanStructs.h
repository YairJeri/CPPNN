#pragma once

#include "VMA/VmaUsage.h"
#include "enums.h"

#include <unordered_map>

struct GPUBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

struct PipelineContext
{
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;
};

struct LayerPipelineContext
{
    PipelineContext forward;              // Cálculo de output = f(x)
    PipelineContext backward_param_grads; // Cálculo de grad_w y grad_b
    PipelineContext backward_input_grad;  // Cálculo de grad_input
};

struct ActivationPipelineContext
{
    PipelineContext forward;             // Cálculo de f(x)
    PipelineContext backward_input_grad; // Cálculo de grad_input (dL/dx = dL/dy * f'(x))
};

struct QueueFamily
{
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t compute_family = UINT32_MAX;
    uint32_t transfer_family = UINT32_MAX;
};
