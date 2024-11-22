
#define NOMINMAX
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#if defined(__APPLE__)
#include <vulkan/vulkan_beta.h>
#endif

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <cstring>
#include <cerrno>
#include <filesystem>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2; // Number of frames to keep in flight at a time (a "frame in flight" is a frame that has been submitted but not yet completed)

// Validation layers are hooks into the Vulkan API that allow us to get various debug information
// Extremely useful for debugging, as Vulkan has no error checking by default for faster performance
// They can easily be fully disabled in release builds, but enabled in debug builds
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#ifdef __APPLE__
    VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#endif
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

std::vector<char> readFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        std::string errorMsg = "Failed to open file: " + filename + " - " + std::strerror(errno);
        throw std::runtime_error(errorMsg);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerEXT *pDebugMessenger)
{
    // Since this function is from a vulkan extension, we need to load it manually
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks *pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices
{
    // Store the index of a queue family that supports graphics operations and one that supports presentation operations
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow *window; // Handle to the window

    VkInstance instance;                     // Connection between application and Vulkan library
    VkDebugUtilsMessengerEXT debugMessenger; // Debug messenger to receive debug messages from validation layers
    VkSurfaceKHR surface;                    // Connection between window and Vulkan library

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // GPU
    VkDevice device;                                  // Logical device to interact with physical device

    VkQueue graphicsQueue; // Queue to submit command buffers to
    VkQueue presentQueue;  // Queue to present images to the window

    VkSwapchainKHR swapChain;                     // Queue of images to be presented to the screen (double buffering, triple buffering)
    std::vector<VkImage> swapChainImages;         // Images in the swap chain
    VkFormat swapChainImageFormat;                // Format of the images in the swap chain
    VkExtent2D swapChainExtent;                   // Resolution of the swap chain images
    std::vector<VkImageView> swapChainImageViews; // Image views to interpret the images in the swap chain

    VkRenderPass renderPass;         // Describes the attachments used during rendering
    VkPipelineLayout pipelineLayout; // Uniform values for shaders

    VkPipeline graphicsPipeline; // Configures the programmable and fixed-function stages of the pipeline

    std::vector<VkFramebuffer> swapChainFramebuffers; // Framebuffers to render images to

    VkCommandPool commandPool;                   // We need one command pool for each queue family that we want to send commands to (e.g. graphics, presentation)
    std::vector<VkCommandBuffer> commandBuffers; // Command buffers to record commands into

    std::vector<VkSemaphore> imageAvailableSemaphores; // Semaphores to synchronize rendering
    std::vector<VkSemaphore> renderFinishedSemaphores; // Semaphores to synchronize rendering
    std::vector<VkFence> inFlightFences;               // Fences to synchronize rendering
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
    {
        auto app = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan()
    {
        createInstance();         // Connection between application and Vulkan library
        setupDebugMessenger();    // Debug messenger to receive debug messages from validation layers
        createSurface();          // Connection between window and Vulkan library (because Vulkan is platform agnostic)
        pickPhysicalDevice();     // GPU
        createLogicalDevice();    // Logical device to interact with physical device (requires setting up queues)
        createSwapChain();        // Queue of images to be presented to the screen (double buffering, triple buffering) - (requires surface, sets up images, format, extent)
        createImageViews();       // Image views to interpret the images in the swap chain (requires swap chain)
        createRenderPass();       // Describes the attachments used during rendering (requires swap chain image format)
        createGraphicsPipeline(); // Configure the programmable and fixed-function stages of the pipeline
        createFramebuffers();     // Framebuffers to render images to (requires swap chain image views)
        createCommandPool();      // Command pool to allocate command buffers from
        createCommandBuffers();   // Command buffer to record commands into
        createSyncObjects();      // Semaphores and fences to synchronize rendering
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            {

                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    // Write the commands we want to execute to the command buffer
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{}; // Begin a new render pass
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        // Begin recording commands to the command buffer
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        // We configured viewport and scissor to be dynamic in the pipeline, so we need to set them here
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // Primary: Can be submitted to a queue for execution, but cannot be called from other command buffers. Secondary: Cannot be submitted directly, but can be called from primary command buffers.
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // An attachment is represented by a VkImageView.
    // The attachments specified during render pass creation are bound by wrapping them in a VkFramebuffer.
    void createFramebuffers()
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++)
        {
            VkImageView attachments[] = {
                swapChainImageViews[i]};

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass; // Soecify with which render pass the framebuffer needs to be compatible (must have the same number and type of attachments)
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createRenderPass()
    {
        // A render pass is a collection of attachments, subpasses, and dependencies between the subpasses.
        // It describes the layout of the attachments, the number of subpasses, and how the subpasses are connected.
        // An attachment is a reference to an image resource that can be used as a framebuffer.
        // A subpass is a rendering operation that reads from and writes to the attachments.
        // One subpass can reference the contents of the previous subpass, allowing for composition of multiple rendering operations.
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;   // Format should match the format of the swap chain images
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // Number of samples to write for multisampling

        // LoadOp: What to do with the data in the attachment before rendering (e.g. clear it or preserve it)
        // StoreOp: What to do with the data in the attachment after rendering (e.g. store it or discard it)
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // StencilLoadOp and StencilStoreOp are similar to LoadOp and StoreOp, but apply to the stencil data
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        // InitialLayout: Layout of the image before the render pass
        // FinalLayout: Layout of the image after the render pass
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;     // Layout to automatically transition from when the render pass begins
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // Layout to automatically transition to when the render pass finishes (in this case, the image will be presented to the screen)

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;                                    // Index of the attachment in the attachment descriptions array
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Layout of the attachment during the subpass (color attachment)

        // Subpasses are subsequent rendering operations that depend on the contents of the previous subpass
        // A subpass description references the index of the color and depth/stencil attachments
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // Pipeline type (graphics, compute, ray tracing)
        subpass.colorAttachmentCount = 1;                            // Number of color attachments. The index of the attachment in this array is directly referenced from the fragment shader with the layout(location = 0) out vec4 outColor directive
        subpass.pColorAttachments = &colorAttachmentRef;             // Reference to the color attachment

        // A VkSubpassDependency describes a dependency between two subpasses
        // We only have one subpass, but nevertheless, there are two implicit subpasses: 1) External subpass before the render pass, 2) External subpass after the render pass
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;                             // Dependency
        dependency.dstSubpass = 0;                                               // Dependent
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // Pipeline stage that must be finished before the dependency can be executed
        dependency.srcAccessMask = 0;                                            // Access mask that must be satisfied before the dependency can be executed
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // Pipeline stage that the dependent subpass waits on
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;         // Access mask that the dependent subpass waits on

        // The following other types of attachments can be referenced in a subpass:
        // pInputAttachments: Attachments that are read from a shader
        // pResolveAttachments: Attachments used for multisampling color attachments
        // pDepthStencilAttachment: Attachment for depth and stencil data
        // pPreserveAttachments: Attachments that are not used by this subpass, but for which the data must be preserved

        // Finally, we create the render pass
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;             // Number of attachments
        renderPassInfo.pAttachments = &colorAttachment; // Attachments
        renderPassInfo.subpassCount = 1;                // Number of subpasses
        renderPassInfo.pSubpasses = &subpass;           // Subpasses
        renderPassInfo.dependencyCount = 1;             // Number of dependencies
        renderPassInfo.pDependencies = &dependency;     // Dependencies

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    // Shader modules are used to load SPIR-V bytecode into Vulkan
    VkShaderModule createShaderModule(const std::vector<char> &code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    void createGraphicsPipeline()
    {
        // We need to configure the stages of the graphics pipeline
        // These stages can include:
        // 1) Vertex input (format of the vertex data - e.g. the position, color, texture coordinates, etc.)
        // 2) Input assembly (how vertices are assembled into primitives - e.g. points, lines, triangles)
        // 3) Vertex shader (operations on each vertex)
        // 4) Tesselation shaders (optional, used for tessellation)
        // 5) Geometry shader  (optional, operations on geometry)
        // 6) Rasterization (creating fragments from vertices - e.g. wireframe, point, fill)
        // 7) Fragment shader (operations on each fragment)
        // 8) Color blending (combining the color of a fragment with the color already in the framebuffer)
        // 9) Viewport (transformation from image space to screen space)
        // 10) Scissor test (discarding fragments outside a rectangle)

        // -- Start by configuring the programmable stages of the pipeline --
        std::filesystem::path shaderPath = std::filesystem::current_path() / "src" / "shaders";
        auto vertShaderCode = readFile((shaderPath / "vert.spv").string());
        auto fragShaderCode = readFile((shaderPath / "frag.spv").string());

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // Shader type
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main"; // Entry point

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        // -- Next, configure the fixed-function stages of the pipeline --

        // Describes the format of the vertex data that will be passed to the vertex shader
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        // Describes the type of geometry (e.g. triangle, line, point) and if primitive restart should be enabled
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // The viewport describes the region of the framebuffer that the output will be rendered to
        // Typically (0, 0) to (width, height)
        // The scissor rectangle defines in which regions pixels will actually be stored
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        // The rasterizer takes the geometry that is shaped by the vertices from the vertex shader and turns it into fragments to be colored by the fragment shader
        // It also performs depth testing, face culling and the scissor test
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;         // If true, fragments beyond the near and far planes are clamped to them as opposed to discarding them. Useful for shadow maps.
        rasterizer.rasterizerDiscardEnable = VK_FALSE;  // If true, geometry never passes through the rasterizer stage. Effectively disables any output to the framebuffer.
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;  // Fill the area of the polygon with fragments
        rasterizer.lineWidth = 1.0f;                    // Line width in fragments
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;    // Cull the back faces
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // Clockwise winding order
        rasterizer.depthBiasEnable = VK_FALSE;          // Depth bias adjusts the depth of a fragment to reduce visual artifacts

        // The multisampling state is used for anti-aliasing
        // We will use 1 sample per pixel, so we will disable it
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;               // Enable multisampling
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; // Number of samples to use per fragment

        // After a fragment shader has returned a color, it needs to be combined with the color that is already in the framebuffer
        // This is known as color blending
        // It could be done by: 1) Mix the old and new color, 2) Combine the old and new color using a bitwise operation
        // VkPipelineColorBlendAttachmentState allows us to configure the first way (mixing), but in our case we will disable it (just overwrite the color)
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; // RGBA color channels
        colorBlendAttachment.blendEnable = VK_FALSE;

        // VkPipelineColorBlendStateCreateInfo describes the global color blending settings
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE; // If true, bitwise color blending is enabled (will overwrite blending, even if blendEnable is true)
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        // Usually, a Vulkan pipeline is immutable and cannot be changed after creation
        // However, some state can be configured as dynamic, such as the viewport and scissor rectangle
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // Describes the layout of the uniform values in the shaders
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // WE NOW HAVE:
        // 1) Shader modules (vertex and fragment, programmable stages)
        // 2) Fixed-function stages (vertex input, input assembly, viewport, rasterizer, multisampling, color blending)
        // 3) Pipeline layout (uniform values for shaders that can be changed at drawing time)
        // 4) Render pass (attachments, subpasses, dependencies)
        // 5) Dynamic state (dynamic viewport and scissor rectangle)
        // We can now create the graphics pipeline (all of the above combined)

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;         // Number of shader stages
        pipelineInfo.pStages = shaderStages; // Shader stages
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr; // Optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        // Vulkan allows you to create a new graphics pipeline by deriving from an existing pipeline
        // It is less expensive than creating a new pipeline from scratch, if we only need to change a few things
        // Switching between pipelines from the same parent can be done more efficiently
        // However, we don't need this, so we set the basePipelineHandle and basePipelineIndex to VK_NULL_HANDLE and -1 respectively
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex = -1;              // Optional

        // The second parameter is a cache object that allows you to reuse the result of pipeline creation
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    // An image view is a way to interpret an image, analogous to std::string_view
    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++)
        {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            // The components field allows you to swizzle the color channels around
            // For example, you can map all of the channels to the red channel for a monochrome texture
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            // The subresourceRange field describes what the image's purpose is and which part of the image should be accessed
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    // A swap chain is essentially a queue of images that are waiting to be presented to the screen e.g. double buffering, triple bufferings
    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;                             // Always 1 unless stereoscopic 3D application (e.g. VR)
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // This means that we will render directly to the images of the swap chain (a color attachment is an image that will be presented on the screen)

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        // If the graphics and present queues are different, we need to specify that the images can be used across multiple queue families
        if (indices.graphicsFamily != indices.presentFamily)
        {
            // Images can be used across multiple queue families without explicit ownership transfers
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            // An image is owned by one queue family at a time and ownership must be explicitly transferred before using it in another queue family
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;     // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        // We can specify that a certain transform should be applied to the images in the swap chain if it is supported, such as a 90 degree rotation or horizontal flip
        // In this case, we don't want any transformation, so we use the currentTransform value
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

        // We can specify if the alpha channel should be used for blending with other windows in the system
        // We don't need this, so we use the opaque value
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        createInfo.presentMode = presentMode;
        // We can specify if we want to clip parts of the window that are obscured by other windows, for better performance
        createInfo.clipped = VK_TRUE;

        // If the old swap chain is no longer compatible with the new one (e.g. window was resized), we need to recreate the swap chain from scratch
        // Here we specify the old swap chain, so that we can continue to present images that are already acquired
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create swap chain!");
        }

        // The swap chain automatically creates images with the format and extent that match the window, for each image in the swap chain
        // We need to retrieve the handles to these images with vkGetSwapchainImagesKHR
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
    {
        // Vulkan will automatically set currentExtent to the window size
        // However, sometimes it will set it to the maximum value of uint32_t - indicating that the window manager supports choosing a custom size that differs from the total window size
        // In this case, we will choose the size of the window
        // https://www.glfw.org/docs/latest/intro_guide.html#coordinate_systems
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)};

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes)
    {
        for (const auto &availablePresentMode : availablePresentModes)
        {
            // Triple buffering if supported
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return availablePresentMode;
            }
        }

        // Otherwise, use double buffering as it is guaranteed to be supported
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats)
    {
        for (const auto &availableFormat : availableFormats)
        {
            // We prefer 32-bit BGRA with sRGB if supported
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    // Logical devices are the interface to interact with physical devices
    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        // We need a VkQueue that supports graphics, and one that supports presentation
        // These queues may be the same (from the same queue family), or they may be different
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // Queues are the interface to submit commands to the GPU
    // Different queues support different types of operations, such as graphics/compute/memory transfer
    // A single queue can support multiple types of operations
    // Queue families are simply groups of n queues that have the same capabilities
    // To render graphics to a window, we need a queue that supports graphics operations and one that supports presentation operations
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto &queueFamily : queueFamilies)
        {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport)
            {
                indices.presentFamily = i;
            }

            if (indices.isComplete())
            {
                break;
            }

            i++;
        }

        return indices;
    }

    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0)
        {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto &device : devices)
        {
            if (isDeviceSuitable(device))
            {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void
    populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    // Debug messenger is used to receive debug messages from the validation layers
    void setupDebugMessenger()
    {
        if (!enableValidationLayers)
            return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void *pUserData)
    {

        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }

    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char *layerName : validationLayers)
        {
            bool layerFound = false;

            for (const auto &layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
            {
                return false;
            }
        }

        return true;
    }

    std::vector<const char *> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

// if macos, add the extra extensions
#ifdef __APPLE__
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

        return extensions;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto &extension : availableExtensions)
        {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    // VkInstance is the connection between your application and the Vulkan library
    void createInstance()
    {
        if (enableValidationLayers && !checkValidationLayerSupport())
        {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();

#if __APPLE__
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        createInfo.enabledExtensionCount = (uint32_t)extensions.size();
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
        }
        else
        {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }

        // There may still be asynchronous operations in progress, so we need to wait for the device to finish before cleaning up
        vkDeviceWaitIdle(device);
    }

    void drawFrame()
    {
        // We need to:
        // 1) Wait for the previous frame to finish
        // 2) Acquire an image from the swap chain
        // 3) Record a command buffer which draws the scene onto that image
        // 4) Submit the command buffer to the graphics queue
        // 5) Return the image to the swap chain for presentation

        // SYNCHRONIZATION:
        // Many Vulkan API calls that work on the GPU are asynchronous (e.g. acquiring an image from the swap chain, executing commands, presenting to the screen)
        // Therefore, we must manually synchronize these operations, e.g. with semaphores and fences

        // SEMAPHORE:
        // A semaphore is a synchronization primitive that can be used to insert a dependency between operations
        // There are two types of semaphores in Vulkan: 1) Binary semaphores 2) Timeline semaphores (integer value)
        // We will only use binary semaphores here
        // Let's say we want to execute two async operations on the GPU in sequence: A and B
        // We create a semaphore that signals when A is done, and that B waits for
        // However, this only effects the synchronization on the GPU, not the CPU. For the CPU, we need to use FENCES

        // FENCE:
        // If the CPU needs to know when a GPU operation is finished, we use a fence
        // A fence can be signaled by the GPU, and waited on by the CPU
        // We will use a fence to wait for the frame to finish before starting the next one

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        // VkSubmitInfo is used to submit command buffers to queues
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // We need to specify that before we can execute the command buffer, we need to wait for the imageAvailableSemaphore to be signaled
        // Then after the command buffer has finished executing, we need to signal the renderFinishedSemaphore
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores; // Set the semaphore to wait on before starting the command buffer
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores; // Set the semaphore to signal when the command buffer finishes

        // Submit the command buffer to the graphics queue, and wait for the image available semaphore
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        // We need to specify that we need to wait for the renderFinishedSemaphore to be signaled before presenting the image
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        // Present the image to the screen
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
        {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void cleanup()
    {
        cleanupSwapChain();

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers)
        {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void cleanupSwapChain()
    {
        for (size_t i = 0; i < swapChainFramebuffers.size(); i++)
        {
            vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
        }

        for (size_t i = 0; i < swapChainImageViews.size(); i++)
        {
            vkDestroyImageView(device, swapChainImageViews[i], nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    // Recreate the swap chain if it is no longer adequate (e.g. if the window is resized)
    void recreateSwapChain()
    {
        // It is possible that the window has been minimised, resulting in a framebuffer size of 0.
        // In this case, check until the window is no longer minimised before recreating the swap chain
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFramebuffers();
    }
};

int main()
{
    HelloTriangleApplication app;

    try
    {
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}