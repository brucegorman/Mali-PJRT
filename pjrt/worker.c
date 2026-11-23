#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "protocol.h"
#include "worker_buffers.h"

struct mali_vulkan {
    VkInstance instance;
    VkDevice device;
    VkQueue queue;
    VkPhysicalDeviceMemoryProperties memory;
    struct mali_device_info info;
};

static bool find_compute_queue(VkPhysicalDevice physical_device, uint32_t *family) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, NULL);
    VkQueueFamilyProperties *properties = calloc(count, sizeof(*properties));
    if (properties == NULL) {
        return false;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, properties);
    uint32_t index = 0;
    while (index < count &&
           (!properties[index].queueCount ||
            !(properties[index].queueFlags & VK_QUEUE_COMPUTE_BIT))) {
        ++index;
    }
    free(properties);
    *family = index;
    return index < count;
}

static bool open_vulkan(struct mali_vulkan *vulkan) {
    const VkApplicationInfo application = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Mali PJRT worker",
        .apiVersion = VK_API_VERSION_1_1,
    };
    const VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application,
    };
    VkResult result = vkCreateInstance(&instance_info, NULL, &vulkan->instance);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed: %d\n", (int)result);
        return false;
    }

    uint32_t count = 0;
    result = vkEnumeratePhysicalDevices(vulkan->instance, &count, NULL);
    if (result != VK_SUCCESS || count == 0) {
        fprintf(stderr, "No Vulkan physical devices: %d\n", (int)result);
        return false;
    }
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    if (devices == NULL) {
        return false;
    }
    result = vkEnumeratePhysicalDevices(vulkan->instance, &count, devices);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "vkEnumeratePhysicalDevices failed: %d\n", (int)result);
        free(devices);
        return false;
    }

    for (uint32_t index = 0; index < count; ++index) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(devices[index], &properties);
        uint32_t family;
        if (properties.vendorID != MALI_VULKAN_VENDOR_ID ||
            properties.apiVersion < VK_API_VERSION_1_1 ||
            !find_compute_queue(devices[index], &family)) {
            continue;
        }
        const float priority = 1.0f;
        const VkDeviceQueueCreateInfo queue_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
        const VkDeviceCreateInfo device_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_info,
        };
        result = vkCreateDevice(devices[index], &device_info, NULL, &vulkan->device);
        if (result != VK_SUCCESS) {
            fprintf(stderr, "vkCreateDevice failed for %s: %d\n", properties.deviceName,
                    (int)result);
            vulkan->device = VK_NULL_HANDLE;
            continue;
        }
        vkGetDeviceQueue(vulkan->device, family, 0, &vulkan->queue);
        vkGetPhysicalDeviceMemoryProperties(devices[index], &vulkan->memory);
        vulkan->info = (struct mali_device_info){
            .protocol_version = MALI_PJRT_PROTOCOL_VERSION,
            .vendor_id = properties.vendorID,
            .device_id = properties.deviceID,
            .api_version = properties.apiVersion,
            .queue_family = family,
        };
        snprintf(vulkan->info.name, sizeof(vulkan->info.name), "%s", properties.deviceName);
        free(devices);
        return true;
    }
    free(devices);
    fputs("Could not open an Arm Vulkan 1.1 device with a compute queue\n", stderr);
    return false;
}

static void close_vulkan(struct mali_vulkan *vulkan) {
    if (vulkan->device != VK_NULL_HANDLE) {
        vkDestroyDevice(vulkan->device, NULL);
    }
    if (vulkan->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(vulkan->instance, NULL);
    }
}

static int serve_client(int listener, const struct mali_vulkan *vulkan) {
    int client = accept(listener, NULL, NULL);
    if (client < 0) {
        perror("accept");
        return 1;
    }

    char request[sizeof(MALI_PJRT_HELLO)];
    ssize_t length = recv(client, request, sizeof(request), MSG_TRUNC);
    int result = 1;
    if (length == sizeof(MALI_PJRT_HELLO) - 1 &&
        memcmp(request, MALI_PJRT_HELLO, (size_t)length) == 0) {
        if (send_packet(client, &vulkan->info, sizeof(vulkan->info), NULL, 0)) {
            result = serve_buffers(client, vulkan->device, &vulkan->memory);
        } else {
            perror("send");
        }
    } else {
        fputs("Invalid PJRT worker handshake\n", stderr);
    }
    close(client);
    return result;
}

int main(int argc, char **argv) {
    struct sockaddr_un address;
    socklen_t address_length;
    if (argc != 2 || !worker_address(argv[1], &address, &address_length)) {
        fprintf(stderr, "Usage: %s ABSTRACT_SOCKET_NAME\n", argv[0]);
        return 2;
    }

    int listener = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (listener < 0) {
        perror("socket");
        return 1;
    }
    if (bind(listener, (const struct sockaddr *)&address, address_length) < 0) {
        perror("bind");
        close(listener);
        return 1;
    }

    int result = 1;
    if (listen(listener, 1) != 0) {
        perror("listen");
    } else {
        struct mali_vulkan vulkan = {0};
        if (open_vulkan(&vulkan)) {
            puts("READY");
            fflush(stdout);
            result = serve_client(listener, &vulkan);
        }
        close_vulkan(&vulkan);
    }
    close(listener);
    return result;
}
