#include <stdio.h>
#include <stdlib.h>

#include "protocol.h"
#include "worker_buffers.h"

struct device_buffer {
    struct device_buffer *next;
    uint64_t handle;
    uint64_t size;
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
};

static void destroy_buffer(VkDevice device, struct device_buffer *buffer) {
    if (buffer->mapped != NULL) {
        vkUnmapMemory(device, buffer->memory);
    }
    vkDestroyBuffer(device, buffer->buffer, NULL);
    if (buffer->memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer->memory, NULL);
    }
    free(buffer);
}

static uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties *memory,
                                 uint32_t allowed_types) {
    const VkMemoryPropertyFlags coherent =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t alternative = UINT32_MAX;
    for (uint32_t i = 0; i < memory->memoryTypeCount; ++i) {
        VkMemoryPropertyFlags flags = memory->memoryTypes[i].propertyFlags;
        if ((allowed_types & (1u << i)) == 0 || (flags & coherent) != coherent) {
            continue;
        }
        if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            return i;
        }
        alternative = i;
    }
    return alternative;
}

static enum mali_status allocate_buffer(VkDevice device,
                                        const VkPhysicalDeviceMemoryProperties *memory,
                                        uint64_t size, uint64_t handle,
                                        struct device_buffer **buffers) {
    struct device_buffer *buffer = calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
        return MALI_OUT_OF_MEMORY;
    }
    const VkBufferCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size != 0 ? size : 1,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult result = vkCreateBuffer(device, &create_info, NULL, &buffer->buffer);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "vkCreateBuffer failed: %d\n", (int)result);
        free(buffer);
        return MALI_OUT_OF_MEMORY;
    }
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffer->buffer, &requirements);
    uint32_t type = find_memory_type(memory, requirements.memoryTypeBits);
    if (type == UINT32_MAX) {
        destroy_buffer(device, buffer);
        return MALI_UNSUPPORTED_MEMORY;
    }
    const VkMemoryAllocateInfo allocation = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = type,
    };
    VkDeviceMemory memory_handle;
    result = vkAllocateMemory(device, &allocation, NULL, &memory_handle);
    if (result == VK_SUCCESS) {
        buffer->memory = memory_handle;
        result = vkBindBufferMemory(device, buffer->buffer, buffer->memory, 0);
    }
    void *mapped;
    if (result == VK_SUCCESS) {
        result = vkMapMemory(device, buffer->memory, 0, VK_WHOLE_SIZE, 0, &mapped);
    }
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Vulkan buffer allocation, bind, or map failed: %d\n", (int)result);
        destroy_buffer(device, buffer);
        return MALI_OUT_OF_MEMORY;
    }
    buffer->mapped = mapped;
    buffer->size = size;
    buffer->handle = handle;
    buffer->next = *buffers;
    *buffers = buffer;
    return MALI_OK;
}

int serve_buffers(int socket, VkDevice device, const VkPhysicalDeviceMemoryProperties *memory) {
    struct device_buffer *buffers = NULL;
    uint64_t next_handle = 1;
    unsigned char payload[MALI_TRANSFER_CHUNK];
    int result = 1;
    for (;;) {
        struct mali_request request;
        ssize_t count = receive_packet(socket, &request, sizeof(request), payload, sizeof(payload));
        if (count == 0) {
            result = 0;
            break;
        }
        if (count < (ssize_t)sizeof(request) ||
            count > (ssize_t)(sizeof(request) + sizeof(payload))) {
            fputs("Invalid Mali buffer packet length\n", stderr);
            break;
        }

        struct mali_reply reply = {.status = MALI_INVALID};
        size_t input_size = (size_t)count - sizeof(request);
        const void *output = NULL;
        size_t output_size = 0;
        if (request.operation == MALI_ALLOCATE) {
            if (input_size == 0 && next_handle != 0) {
                reply.status = allocate_buffer(device, memory, request.length, next_handle,
                                               &buffers);
                if (reply.status == MALI_OK) {
                    reply.handle = next_handle++;
                }
            }
        } else {
            struct device_buffer **link = &buffers;
            while (*link != NULL && (*link)->handle != request.handle) {
                link = &(*link)->next;
            }
            struct device_buffer *buffer = *link;
            if (buffer == NULL) {
                reply.status = MALI_NOT_FOUND;
            } else if (request.operation == MALI_RELEASE && input_size == 0) {
                *link = buffer->next;
                destroy_buffer(device, buffer);
                reply.status = MALI_OK;
            } else if (request.offset <= buffer->size &&
                       request.length <= buffer->size - request.offset &&
                       request.length <= sizeof(payload)) {
                unsigned char *bytes = (unsigned char *)buffer->mapped + request.offset;
                if (request.operation == MALI_WRITE && input_size == request.length) {
                    memcpy(bytes, payload, input_size);
                    reply.status = MALI_OK;
                } else if (request.operation == MALI_READ && input_size == 0) {
                    output = bytes;
                    output_size = (size_t)request.length;
                    reply.status = MALI_OK;
                }
            }
        }
        if (!send_packet(socket, &reply, sizeof(reply), output, output_size)) {
            fputs("Could not reply to Mali buffer request\n", stderr);
            break;
        }
    }
    while (buffers != NULL) {
        struct device_buffer *next = buffers->next;
        destroy_buffer(device, buffers);
        buffers = next;
    }
    return result;
}
