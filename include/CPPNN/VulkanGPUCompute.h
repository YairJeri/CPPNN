#pragma once

#include "VulkanStructs.h"
#include "TensorDevice.h"

#include <vector>
#include <stdexcept>
#include <map>
#include <cstdlib>
#include <fstream>

struct LayerInstance
{
    NN::Layer layer;
    NN::Activation activation;

    uint32_t input_size;
    uint32_t output_size;

    std::unordered_map<NN::BufferID, NN::Tensor<NN::GPU>> buffers;

    uint32_t dispatch_x;
    uint32_t dispatch_y;
    uint32_t dispatch_z;

    VkCommandBuffer command_buffer_forward;
    VkCommandBuffer command_buffer_backward;
};

class VulkanGPUCompute
{
    VkInstance instance;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t max_buffer_size;
    VkDevice device;
    VmaAllocator allocator;
    VkQueue compute_queue;
    uint32_t queue_compute_fam_index;
    VkCommandPool command_pool;
    VkDescriptorPool descriptor_pool;

    std::unordered_map<NN::Layer, LayerPipelineContext> layers_pipeline_contexts;
    std::unordered_map<NN::Activation, ActivationPipelineContext> activation_pipeline_contexts;
    std::vector<LayerInstance> layers_base;

public:
    void initVulkan();
    void cleanup();
    void createInstance();
    void pickPhysicalDevice();
    int rateDeviceSuitability(QueueFamily &device);
    void createLogicalDevice();
    void createAllocator();

    std::vector<uint32_t> loadShader(const std::string &path);
    VkDescriptorSetLayout createDescriptorSetLayout(const std::vector<VkDescriptorSetLayoutBinding> &bindings);
    void createLayerPipeline(PipelineContext &layer_context, std::vector<uint32_t> &shader_code);
    void createLayerPipelineContext(NN::Layer layer);
    void createLayerPipelineContext(NN::Activation activation);
    GPUBuffer createBuffer(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    void uploadToBuffer(GPUBuffer &buffer, void *data, size_t size);
    void downloadFromBuffer(GPUBuffer &buffer, void *data, size_t size);

    void createDescriptorSetForward(LayerInstance &layer);
    void createDescriptorSetBackward(LayerInstance &layer);

    void createLayer(LayerInstance &layer);

    void createCommandPool();
    void createCommandBuffer(VkCommandBuffer &command_buffer);
    void recordCommandBufferForward(LayerInstance &layer);
    void recordCommandBufferBackward(LayerInstance &layer);
    void createDescriptorPool();
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

public:
    VulkanGPUCompute();
    ~VulkanGPUCompute();
};

VulkanGPUCompute::VulkanGPUCompute()
{
    initVulkan();
}

VulkanGPUCompute::~VulkanGPUCompute()
{
    cleanup();
}

void VulkanGPUCompute::initVulkan()
{
    createInstance();
    pickPhysicalDevice();
    createLogicalDevice();
    createAllocator();
    createDescriptorPool();
    createCommandPool();
}

void VulkanGPUCompute::cleanup()
{
    for (auto &[layer, ctx] : layers_pipeline_contexts)
    {
        vkDestroyPipeline(device, ctx.forward.pipeline, nullptr);
        vkDestroyPipelineLayout(device, ctx.forward.pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, ctx.forward.descriptor_set_layout, nullptr);
        vkDestroyPipeline(device, ctx.backward_param_grads.pipeline, nullptr);
        vkDestroyPipelineLayout(device, ctx.backward_param_grads.pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, ctx.backward_param_grads.descriptor_set_layout, nullptr);
        vkDestroyPipeline(device, ctx.backward_input_grad.pipeline, nullptr);
        vkDestroyPipelineLayout(device, ctx.backward_input_grad.pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, ctx.backward_input_grad.descriptor_set_layout, nullptr);
    }
    for (auto &[layer, ctx] : activation_pipeline_contexts)
    {
        vkDestroyPipeline(device, ctx.forward.pipeline, nullptr);
        vkDestroyPipelineLayout(device, ctx.forward.pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, ctx.forward.descriptor_set_layout, nullptr);
        vkDestroyPipeline(device, ctx.backward_input_grad.pipeline, nullptr);
        vkDestroyPipelineLayout(device, ctx.backward_input_grad.pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, ctx.backward_input_grad.descriptor_set_layout, nullptr);
    }

    layers_pipeline_contexts.clear();
    activation_pipeline_contexts.clear();

    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vmaDestroyAllocator(allocator);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}

void VulkanGPUCompute::createInstance()
{
    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "VulkanGPUCompute";
    app_info.applicationVersion = 1;
    app_info.pEngineName = "CPPNN";
    app_info.engineVersion = 1;
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app_info;

    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("Failed to create instance!");
}

void VulkanGPUCompute::pickPhysicalDevice()
{
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

    if (device_count == 0)
        throw std::runtime_error("No physical devices found!");

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());

    std::multimap<int, QueueFamily> scored_devices;

    for (const auto &device : physical_devices)
    {
        QueueFamily queue_family;
        queue_family.physical_device = device;
        int score = rateDeviceSuitability(queue_family);
        scored_devices.insert(std::make_pair(score, queue_family));
    }

    if (scored_devices.rbegin()->first > 0)
    {
        physical_device = scored_devices.rbegin()->second.physical_device;
        queue_compute_fam_index = scored_devices.rbegin()->second.compute_family;
    }
    else
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    VkPhysicalDeviceProperties device_properties;
    vkGetPhysicalDeviceProperties(physical_device, &device_properties);

    max_buffer_size = device_properties.limits.maxStorageBufferRange;
}

int VulkanGPUCompute::rateDeviceSuitability(QueueFamily &fam)
{
    VkPhysicalDeviceProperties device_properties;
    vkGetPhysicalDeviceProperties(fam.physical_device, &device_properties);

    int score = 0;

    score += device_properties.limits.maxStorageBufferRange / 1024 / 1024;
    if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    {
        score += 1000;
    }

    std::cout << "Device name: " << device_properties.deviceName << std::endl;
    std::cout << "Score: " << score << " points" << std::endl;

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(fam.physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(fam.physical_device, &queue_family_count, queue_families.data());

    uint32_t act_family = 0;
    for (const auto &queue_family : queue_families)
    {
        if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT)
        {
            fam.compute_family = act_family;
        }
        // Verifica las colas de transferencia
        if (queue_family.queueFlags & VK_QUEUE_TRANSFER_BIT)
        {
            fam.transfer_family = act_family;
        }

        // Si ya hemos encontrado las tres familias de colas necesarias, salimos del ciclo
        if (fam.compute_family != UINT32_MAX && fam.transfer_family != UINT32_MAX)
        {
            return score;
        }

        act_family++;
    }

    return 0;
}

void VulkanGPUCompute::createLogicalDevice()
{
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_compute_fam_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures device_features{};

    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &device_features;

    if (vkCreateDevice(physical_device, &device_info, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("Failed to create device!");

    vkGetDeviceQueue(device, queue_compute_fam_index, 0, &compute_queue);
}

void VulkanGPUCompute::createAllocator()
{
    VmaAllocatorCreateInfo alloc_info = {};
    alloc_info.physicalDevice = physical_device;
    alloc_info.device = device;
    alloc_info.instance = instance;
    if (vmaCreateAllocator(&alloc_info, &allocator) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create allocator!");
    }
}

std::vector<uint32_t> VulkanGPUCompute::loadShader(const std::string &path)
{
    std::ifstream file("shaders/" + path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Failed to open shader file!");
    size_t size = file.tellg();
    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(spirv.data()), size);
    file.close();
    return spirv;
}

VkDescriptorSetLayout VulkanGPUCompute::createDescriptorSetLayout(const std::vector<VkDescriptorSetLayoutBinding> &bindings)
{
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    VkDescriptorSetLayout layout;
    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor set layout!");
    return layout;
}

void VulkanGPUCompute::createLayerPipeline(PipelineContext &layer_context, std::vector<uint32_t> &shader_code)
{
    VkShaderModuleCreateInfo shader_module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_module_info.codeSize = shader_code.size() * sizeof(uint32_t);
    shader_module_info.pCode = reinterpret_cast<const uint32_t *>(shader_code.data());

    VkShaderModule compute_shader_module;
    if (vkCreateShaderModule(device, &shader_module_info, nullptr, &compute_shader_module) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module!");

    VkPipelineShaderStageCreateInfo shader_stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_info.module = compute_shader_module;
    shader_stage_info.pName = "main";

    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &layer_context.descriptor_set_layout;

    if (vkCreatePipelineLayout(device, &layout_info, nullptr, &layer_context.pipeline_layout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create pipeline layout!");

    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = shader_stage_info;
    pipeline_info.layout = layer_context.pipeline_layout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &layer_context.pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create compute pipeline!");

    vkDestroyShaderModule(device, compute_shader_module, nullptr);
}

void VulkanGPUCompute::createLayerPipelineContext(NN::Layer layer)
{
    if (layers_pipeline_contexts.find(layer) != layers_pipeline_contexts.end())
        return;

    LayerPipelineContext layer_context;

    std::vector<uint32_t> shader_code_forward;
    std::vector<uint32_t> shader_code_backward_param_grads;
    std::vector<uint32_t> shader_code_backward_input_grad;

    switch (layer)
    {
    case NN::Layer::Dense:
    {
        shader_code_forward = loadShader("dense_forward.comp.spv");
        shader_code_backward_param_grads = loadShader("dense_backward_param_grads.comp.spv");
        shader_code_backward_input_grad = loadShader("dense_backward_input_grad.comp.spv");
        std::vector<VkDescriptorSetLayoutBinding> bindings_forward = {
            // Input
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            // Output
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT}, // Input de la siguiente capa
            // Weights
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            // Biases
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        };
        std::vector<VkDescriptorSetLayoutBinding> bindings_backward_param_grads = {
            // Input_cache
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT}, // binding_forward 0
            // Delta
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            // GradW
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            // GradB
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        };
        std::vector<VkDescriptorSetLayoutBinding> bindings_backward_input_grad = {
            // Delta
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT}, // binding_bacward_param 1
            // Weights
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT}, // binding_forward 2
            // GradInput
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT}, // Delta de los siguientes
        };

        layer_context.forward.descriptor_set_layout = createDescriptorSetLayout(bindings_forward);
        layer_context.backward_param_grads.descriptor_set_layout = createDescriptorSetLayout(bindings_backward_param_grads);
        layer_context.backward_input_grad.descriptor_set_layout = createDescriptorSetLayout(bindings_backward_input_grad);

        createLayerPipeline(layer_context.forward, shader_code_forward);
        createLayerPipeline(layer_context.backward_param_grads, shader_code_backward_param_grads);
        createLayerPipeline(layer_context.backward_input_grad, shader_code_backward_input_grad);

        {
            VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            alloc_info.descriptorPool = descriptor_pool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &layer_context.forward.descriptor_set_layout;

            if (vkAllocateDescriptorSets(device, &alloc_info, &layer_context.forward.descriptor_set) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate descriptor set for forward pipeline");
        }

        {
            VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            alloc_info.descriptorPool = descriptor_pool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &layer_context.backward_param_grads.descriptor_set_layout;

            if (vkAllocateDescriptorSets(device, &alloc_info, &layer_context.backward_param_grads.descriptor_set) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate descriptor set for backward_param_grads pipeline");
        }

        {
            VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            alloc_info.descriptorPool = descriptor_pool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &layer_context.backward_input_grad.descriptor_set_layout;

            if (vkAllocateDescriptorSets(device, &alloc_info, &layer_context.backward_input_grad.descriptor_set) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate descriptor set for backward_input_grad pipeline");
        }
        layers_pipeline_contexts[layer] = layer_context;
        break;
    }
    case NN::Layer::Conv2D:
        break;
    case NN::Layer::MaxPooling2D:
        break;
    case NN::Layer::Flatten:
        break;
    case NN::Layer::Output:
        break;
    }
}

void VulkanGPUCompute::createLayerPipelineContext(NN::Activation activation)
{
    if (activation_pipeline_contexts.find(activation) != activation_pipeline_contexts.end())
        return;

    ActivationPipelineContext layer_context;

    std::vector<uint32_t> shader_code_forward;
    std::vector<uint32_t> shader_code_backward;

    switch (activation)
    {
    case NN::Activation::Sigmoid:
        shader_code_forward = loadShader("shaders/activation_sigmoid_forward.comp.spv");
        shader_code_backward = loadShader("shaders/activation_sigmoid_backward.comp.spv");
        break;
    case NN::Activation::ReLU:
        shader_code_forward = loadShader("shaders/activation_relu_forward.comp.spv");
        shader_code_backward = loadShader("shaders/activation_relu_backward.comp.spv");
        break;
    case NN::Activation::Linear:
        break;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings_forward = {
        // Input
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
    };
    std::vector<VkDescriptorSetLayoutBinding> bindings_backward = {
        // Input
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        // Cache
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
    };

    layer_context.forward.descriptor_set_layout = createDescriptorSetLayout(bindings_forward);
    layer_context.backward_input_grad.descriptor_set_layout = createDescriptorSetLayout(bindings_backward);

    createLayerPipeline(layer_context.forward, shader_code_forward);
    createLayerPipeline(layer_context.backward_input_grad, shader_code_backward);
    activation_pipeline_contexts[activation] = layer_context;
}

GPUBuffer VulkanGPUCompute::createBuffer(size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
{
    GPUBuffer buffer;

    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer.buffer, &buffer.allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Failed to create buffer!");

    return buffer;
}

void VulkanGPUCompute::uploadToBuffer(GPUBuffer &buffer, void *data, size_t size)
{
    void *mappedData = nullptr;
    vmaMapMemory(allocator, buffer.allocation, &mappedData);
    memcpy(mappedData, data, size);
    vmaUnmapMemory(allocator, buffer.allocation);
}

void VulkanGPUCompute::downloadFromBuffer(GPUBuffer &buffer, void *data, size_t size)
{
    void *mappedData = nullptr;
    vmaMapMemory(allocator, buffer.allocation, &mappedData);
    memcpy(data, mappedData, size);
    vmaUnmapMemory(allocator, buffer.allocation);
}

void VulkanGPUCompute::createDescriptorSetForward(LayerInstance &layer)
{
}

inline void VulkanGPUCompute::createDescriptorSetBackward(LayerInstance &layer)
{
}

void VulkanGPUCompute::createLayer(LayerInstance &layer)
{
    createLayerPipelineContext(layer.layer);
    createLayerPipelineContext(layer.activation);
    createDescriptorSetForward(layer);
    createCommandBuffer(layer.command_buffer_forward);
    recordCommandBufferForward(layer);
    createDescriptorSetBackward(layer);
    createCommandBuffer(layer.command_buffer_backward);
    // recordCommandBufferBackward(layer);
}

void VulkanGPUCompute::createCommandPool()
{
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_compute_fam_index;
    if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool!");
}

void VulkanGPUCompute::createCommandBuffer(VkCommandBuffer &command_buffer)
{
    VkCommandBufferAllocateInfo alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc_info.commandPool = command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffer!");
}

void VulkanGPUCompute::recordCommandBufferForward(LayerInstance &layer)
{
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    begin_info.pInheritanceInfo = nullptr;

    if (vkBeginCommandBuffer(layer.command_buffer_forward, &begin_info) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin recording command buffer!");

    const auto &ctx = layers_pipeline_contexts.at(layer.layer);

    vkCmdBindPipeline(layer.command_buffer_forward, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.forward.pipeline);
    vkCmdBindDescriptorSets(layer.command_buffer_forward, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.forward.pipeline_layout, 0, 1, &ctx.forward.descriptor_set, 0, nullptr);
    vkCmdDispatch(layer.command_buffer_forward, layer.dispatch_x, layer.dispatch_y, layer.dispatch_z);

    if (layer.activation != NN::Activation::Linear)
    {
        const auto &ctx_activation = activation_pipeline_contexts.at(layer.activation);
        vkCmdBindPipeline(layer.command_buffer_forward, VK_PIPELINE_BIND_POINT_COMPUTE, ctx_activation.forward.pipeline);
        vkCmdBindDescriptorSets(layer.command_buffer_forward, VK_PIPELINE_BIND_POINT_COMPUTE, ctx_activation.forward.pipeline_layout, 0, 1, &ctx.forward.descriptor_set, 0, nullptr);
        vkCmdDispatch(layer.command_buffer_forward, layer.dispatch_x, layer.dispatch_y, layer.dispatch_z);
    }

    if (vkEndCommandBuffer(layer.command_buffer_forward) != VK_SUCCESS)
        throw std::runtime_error("Failed to end recording command buffer!");
}

void VulkanGPUCompute::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> pool_sizes(2);
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[0].descriptorCount = 1000;

    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = 100;

    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = 500;

    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor pool!");
}

uint32_t VulkanGPUCompute::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}
