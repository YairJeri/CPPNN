#include "CPPNN/VulkanCompute.h"
/*
VulkanCompute::VulkanCompute()
{
    initVulkan();
}

VulkanCompute::~VulkanCompute()
{
    cleanup();
}

std::vector<char> VulkanCompute::readFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + filename);

    size_t file_size = (size_t)file.tellg();
    std::vector<char> buffer(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();
    return buffer;
}

void VulkanCompute::createInstance()
{
    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "VulkanCompute";
    app_info.apiVersion = VK_API_VERSION_1_1;
    app_info.applicationVersion = 1;
    app_info.engineVersion = 0;

    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app_info;

    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("Failed to create instance!");
}

void VulkanCompute::createLogicalDevice()
{
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0)
        throw std::runtime_error("No physical devices found!");
    std::vector<VkPhysicalDevice> physical_devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());

    uint32_t comp_fam = 0;
    bool found = false;
    for (const auto &dev : physical_devices)
    {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_family_count, queue_families.data());

        comp_fam = 0;
        for (; comp_fam < queue_family_count; comp_fam++)
            if (queue_families[comp_fam].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                physical_device = dev;
                found = true;
                break;
            }
        if (found)
            break;
    }

    if (!found)
        throw std::runtime_error("Failed to find a suitable device!");

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = comp_fam;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;

    VkPhysicalDeviceProperties device_properties;
    vkGetPhysicalDeviceProperties(physical_device, &device_properties);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &device_memory_properties);

    max_buffer_size = device_properties.limits.maxStorageBufferRange;
    std::cout << "Device name: " << device_properties.deviceName << std::endl;
    std::cout << "Maximum storage buffer size: " << max_buffer_size / 1024 / 1024 << " MB" << std::endl;

    if (vkCreateDevice(physical_device, &device_info, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("Failed to create device!");
    vkGetDeviceQueue(device, comp_fam, 0, &queue);

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = comp_fam;
    if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool!");

    VkCommandBufferAllocateInfo alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc_info.commandPool = command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &alloc_info, &command_buffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffer!");
}

void VulkanCompute::createComputePipeline()
{
    auto shader_code = readFile("shader.comp.spv");

    VkShaderModuleCreateInfo shader_module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_module_info.codeSize = shader_code.size();
    shader_module_info.pCode = reinterpret_cast<const uint32_t *>(shader_code.data());

    if (vkCreateShaderModule(device, &shader_module_info, nullptr, &compute_shader_module) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module!");

    VkPipelineShaderStageCreateInfo shader_stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_info.module = compute_shader_module;
    shader_stage_info.pName = "main";

    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};

    if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create pipeline layout!");

    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = shader_stage_info;
    pipeline_info.layout = pipeline_layout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create compute pipeline!");
}

uint32_t VulkanCompute::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    // Buscar un tipo de memoria que cumpla con los requisitos
    for (uint32_t i = 0; i < device_memory_properties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) &&
            (device_memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i; // Retorna el índice del tipo de memoria adecuado
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

void VulkanCompute::createDescriptorPool()
{
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;

    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor pool!");
}

void VulkanCompute::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding layout_binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 1;
    layout_info.pBindings = &layout_binding;

    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor set layout!");
}

// En VulkanCompute.cpp
VkBuffer VulkanCompute::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceMemory &buffer_memory)
{
    VkBuffer buffer;
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to create buffer!");

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = findMemoryType(mem_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate buffer memory!");

    vkBindBufferMemory(device, buffer, buffer_memory, 0);
    return buffer;
}

VkDeviceMemory VulkanCompute::createBufferMemory(VkDeviceSize size, VkMemoryPropertyFlags properties)
{
    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = size;

    // Obtener los tipos de memoria disponibles
    VkMemoryRequirements memory_properties;
    vkGetBufferMemoryRequirements(physical_device, &memory_properties);

    // Filtrar los tipos de memoria que cumplen con las propiedades requeridas
    uint32_t memory_type_index = findMemoryType(memory_properties.memoryTypeBits, properties);

    alloc_info.memoryTypeIndex = memory_type_index;

    VkDeviceMemory buffer_memory;
    if (vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate buffer memory!");
    }

    return buffer_memory;
}

void VulkanCompute::createBuffers(int input_size, int output_size, int weight_size, int bias_size)
{

    // Tamaño máximo recomendado de buffer (por ejemplo, 256MB a 1GB)
    const size_t MAX_BUFFER_SIZE = std::min(size_t(4 * 1024 * 1024 * 1024), (size_t)max_buffer_size); // Max 4GB o lo que permita la GPU

    // Crear buffers de entrada (divididos en fragmentos)
    input_buffers.clear();
    input_buffers_mem.clear();
    size_t remaining_input_size = input_size * sizeof(float);
    while (remaining_input_size > 0)
    {
        size_t buffer_size = std::min(MAX_BUFFER_SIZE, remaining_input_size);

        // Crear el buffer de entrada
        VkDeviceMemory buffer_mem = createBufferMemory(buffer_size, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkBuffer buffer = createBuffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer_mem);

        input_buffers.push_back(buffer);
        input_buffers_mem.push_back(buffer_mem);

        remaining_input_size -= buffer_size;
    }

    // Crear buffers de salida (divididos en fragmentos)
    output_buffers.clear();
    output_buffers_mem.clear();
    size_t remaining_output_size = output_size * sizeof(float);
    while (remaining_output_size > 0)
    {
        size_t buffer_size = std::min(MAX_BUFFER_SIZE, remaining_output_size);

        // Crear el buffer de salida
        VkDeviceMemory buffer_mem = createBufferMemory(buffer_size, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkBuffer buffer = createBuffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer_mem);

        output_buffers.push_back(buffer);
        output_buffers_mem.push_back(buffer_mem);

        remaining_output_size -= buffer_size;
    }

    // Crear buffers de pesos (divididos en fragmentos)
    weight_buffers.clear();
    weight_buffers_mem.clear();
    size_t remaining_weight_size = weight_size * sizeof(float);
    while (remaining_weight_size > 0)
    {
        size_t buffer_size = std::min(MAX_BUFFER_SIZE, remaining_weight_size);

        // Crear el buffer de pesos
        VkDeviceMemory buffer_mem = createBufferMemory(buffer_size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkBuffer buffer = createBuffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer_mem);

        weight_buffers.push_back(buffer);
        weight_buffers_mem.push_back(buffer_mem);

        remaining_weight_size -= buffer_size;
    }

    // Crear buffers de sesgos (divididos en fragmentos)
    bias_buffers.clear();
    bias_buffers_mem.clear();
    size_t remaining_bias_size = bias_size * sizeof(float);
    while (remaining_bias_size > 0)
    {
        size_t buffer_size = std::min(MAX_BUFFER_SIZE, remaining_bias_size);

        // Crear el buffer de sesgos
        VkDeviceMemory buffer_mem = createBufferMemory(buffer_size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkBuffer buffer = createBuffer(buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer_mem);

        bias_buffers.push_back(buffer);
        bias_buffers_mem.push_back(buffer_mem);

        remaining_bias_size -= buffer_size;
    }
}

void VulkanCompute::initVulkan()
{
    createInstance();
    createLogicalDevice();
    createDescriptorPool();
    createDescriptorSetLayout();
    // createComputePipeline();
}

void VulkanCompute::cleanup()
{
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, compute_shader_module, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}
*/