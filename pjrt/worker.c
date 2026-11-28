#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "fence_probe.h"
#include "protocol.h"
#include "worker_buffers.h"

struct mali_vulkan {
    VkInstance instance;
    iree_runtime_instance_t *runtime;
    iree_hal_device_t *device;
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
        vulkan->info = (struct mali_device_info){
            .protocol_version = MALI_PJRT_PROTOCOL_VERSION,
            .vendor_id = properties.vendorID,
            .device_id = properties.deviceID,
            .api_version = properties.apiVersion,
            .queue_family = family,
            .driver_version = properties.driverVersion,
        };
        memcpy(vulkan->info.pipeline_cache_uuid, properties.pipelineCacheUUID,
               sizeof(vulkan->info.pipeline_cache_uuid));
        snprintf(vulkan->info.name, sizeof(vulkan->info.name), "%s", properties.deviceName);
        free(devices);
        iree_runtime_instance_options_t options;
        iree_runtime_instance_options_initialize(&options);
        iree_runtime_instance_options_use_all_available_drivers(&options);
        iree_status_t status = iree_runtime_instance_create(
            &options, iree_allocator_system(), &vulkan->runtime);
        if (iree_status_is_ok(status)) {
            iree_hal_driver_t *driver = NULL;
            iree_hal_driver_registry_t *registry =
                iree_runtime_instance_driver_registry(vulkan->runtime);
            status = iree_hal_driver_registry_try_create(
                registry, IREE_SV("vulkan"), iree_allocator_system(), &driver);
            if (iree_status_is_ok(status)) {
                iree_host_size_t device_count = 0;
                iree_hal_device_info_t *iree_devices = NULL;
                status = iree_hal_driver_query_available_devices(
                    driver, iree_allocator_system(), &device_count, &iree_devices);
                if (iree_status_is_ok(status)) {
                    bool found = false;
                    for (iree_host_size_t i = 0; i < device_count; ++i) {
                        if (iree_string_view_equal(iree_devices[i].name,
                                                   iree_make_cstring_view(vulkan->info.name))) {
                            status = iree_hal_driver_create_device_by_id(
                                driver, iree_devices[i].device_id, 0, NULL,
                                iree_allocator_system(), &vulkan->device);
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                                  "IREE did not find %s", vulkan->info.name);
                    }
                }
                iree_allocator_free(iree_allocator_system(), iree_devices);
                iree_hal_driver_release(driver);
            }
        }
        if (!iree_status_is_ok(status)) {
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }
        return true;
    }
    free(devices);
    fputs("Could not open an Arm Vulkan 1.1 device with a compute queue\n", stderr);
    return false;
}

static void close_vulkan(struct mali_vulkan *vulkan) {
    iree_hal_device_release(vulkan->device);
    iree_runtime_instance_release(vulkan->runtime);
    if (vulkan->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(vulkan->instance, NULL);
    }
}

static int accept_client(int listener) {
    const uid_t worker_uid = geteuid();
    for (;;) {
        int client = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
        if (client < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("accept4");
            return -1;
        }

        struct ucred peer;
        socklen_t peer_size = sizeof(peer);
        if (getsockopt(client, SOL_SOCKET, SO_PEERCRED, &peer, &peer_size) == 0 &&
            peer_size == sizeof(peer) && peer.uid == worker_uid) {
            return client;
        }

        fputs("Rejected Mali worker connection without matching peer UID\n", stderr);
        close(client);
    }
}

static int serve_client(int listener, const struct mali_vulkan *vulkan) {
    int client = accept_client(listener);
    if (client < 0) {
        return 1;
    }

    char request[sizeof(MALI_PJRT_HELLO)];
    ssize_t length = recv(client, request, sizeof(request), MSG_TRUNC);
    int result = 1;
    if (length == sizeof(MALI_PJRT_HELLO) - 1 &&
        memcmp(request, MALI_PJRT_HELLO, (size_t)length) == 0) {
        if (send_packet(client, &vulkan->info, sizeof(vulkan->info), NULL, 0)) {
            result = serve_buffers(client, vulkan->runtime, vulkan->device);
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
    if (argc == 3 && strcmp(argv[1], "--fence-probe") == 0) {
        struct mali_vulkan vulkan = {0};
        int result = 1;
        if (open_vulkan(&vulkan)) {
            printf("Fence probe device: %s\n", vulkan.info.name);
            result = run_fence_probe(vulkan.runtime, vulkan.device, argv[2]);
        }
        close_vulkan(&vulkan);
        return result;
    }
    struct sockaddr_un address;
    socklen_t address_length;
    if (argc != 2 || !worker_address(argv[1], &address, &address_length)) {
        fprintf(stderr, "Usage: %s ABSTRACT_SOCKET_NAME | --fence-probe VMFB\n", argv[0]);
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
