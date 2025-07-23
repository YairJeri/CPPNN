#pragma once

#include <vulkan/vulkan.h>
#include <fstream>
#include <iostream>
#include <vector>
/*

class VulkanCompute
{
    VkDevice device;
    VkInstance instance;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkQueue queue;
    VkPipeline pipeline;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkPipelineLayout pipeline_layout;
    VkShaderModule compute_shader_module;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    uint32_t max_buffer_size;
    VkPhysicalDeviceMemoryProperties device_memory_properties;

    std::vector<VkBuffer> weight_buffers;
    std::vector<VkDeviceMemory> weight_buffers_mem;
    std::vector<VkBuffer> bias_buffers;
    std::vector<VkDeviceMemory> bias_buffers_mem;

    std::vector<VkBuffer> input_buffers;
    std::vector<VkDeviceMemory> input_buffers_mem;
    std::vector<VkBuffer> output_buffers;
    std::vector<VkDeviceMemory> output_buffers_mem;

public:
    VulkanCompute();
    ~VulkanCompute();

    void initVulkan();

    std::vector<char> readFile(const std::string &filename);
    void createInstance();
    void createLogicalDevice();
    void createDescriptorPool();
    void createDescriptorSetLayout();
    void createBuffers(int input_size, int output_size, int weight_size, int bias_size);
    void createComputePipeline();

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceMemory &mem);
    VkDeviceMemory createBufferMemory(VkDeviceSize size, VkMemoryPropertyFlags properties);
    void createBuffers();
    void cleanup();

    void run_compute(int size, const void *data, void *result);
};
*/