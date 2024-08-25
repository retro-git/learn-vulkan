cmake_minimum_required(VERSION 3.30)

# Set the project name and version
project(VulkanApp VERSION 1.0)

include(cmake/CPM.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find GLFW
# find_package(GLFW3 REQUIRED)

# add glfw3 with cpm
CPMAddPackage(
  NAME glfw
  GITHUB_REPOSITORY glfw/glfw
  GIT_TAG 3.3.4
  OPTIONS
	"GLFW_BUILD_TESTS Off"
	"GLFW_BUILD_EXAMPLES Off"
	"GLFW_BUILD_DOCS Off"
    "GLFW_INSTALL Off"
)

# Find GLM
# find_package(glm REQUIRED)

# add glm with cpm
CPMAddPackage(
  NAME glm
  GITHUB_REPOSITORY g-truc/glm
  GIT_TAG 1.0.1
)

# Set include directories
include_directories(${GLFW3_INCLUDE_DIRS})
include_directories(${Vulkan_INCLUDE_DIR})
include_directories(${GLM_INCLUDE_DIRS})

# Add the source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h")

# Add the executable
add_executable(VulkanApp ${SOURCES})

# print where find_package is looking
message(STATUS "Vulkan_INCLUDE_DIR: ${Vulkan_INCLUDE_DIR}")

# Link Vulkan and GLFW
target_link_libraries(VulkanApp Vulkan::Vulkan glfw glm::glm)

# Platform-specific settings
if(APPLE)
    find_library(IOSURFACE_LIBRARY IOSurface REQUIRED)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(COCOA_LIBRARY Cocoa REQUIRED)
    find_library(IOKIT_LIBRARY IOKit REQUIRED)
    find_library(COREVIDEO_LIBRARY CoreVideo REQUIRED)
    find_library(QUARTZCORE_LIBRARY QuartzCore REQUIRED)
    find_library(MOLTENVK MoltenVK REQUIRED)
    
    target_link_libraries(VulkanApp ${IOSURFACE_LIBRARY} ${METAL_LIBRARY} ${COCOA_LIBRARY} ${IOKIT_LIBRARY} ${COREVIDEO_LIBRARY} ${QUARTZCORE_LIBRARY})
    
    # Add MoltenVK path (assuming MoltenVK is installed via Vulkan SDK)
    target_link_libraries(VulkanApp ${Vulkan_LIBRARY})
    
    # Set compile definitions for macOS
    target_compile_definitions(VulkanApp PRIVATE VK_USE_PLATFORM_MACOS_MVK)
else()
    # Windows specific settings
    target_compile_definitions(VulkanApp PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()

# Post build instructions
if(APPLE)
    # Copy MoltenVK dynamic library to the executable output directory on macOS
    add_custom_command(TARGET VulkanApp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${Vulkan_LIBRARY} $<TARGET_FILE_DIR:VulkanApp>)
endif()
