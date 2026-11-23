#ifndef MALI_PJRT_WORKER_BUFFERS_H_
#define MALI_PJRT_WORKER_BUFFERS_H_

#include <vulkan/vulkan.h>

int serve_buffers(int socket, VkDevice device, const VkPhysicalDeviceMemoryProperties *memory);

#endif
