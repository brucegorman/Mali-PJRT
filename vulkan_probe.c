#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

static void print_version(const char *label, uint32_t version) {
    printf("%s=%u.%u.%u\n", label, VK_API_VERSION_MAJOR(version),
           VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version));
}

static bool inspect_device(VkPhysicalDevice device, uint32_t instance_version,
                           PFN_vkGetPhysicalDeviceFeatures2 get_features) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);
    printf("device=%s vendor_id=0x%04x\n", properties.deviceName, properties.vendorID);
    print_version("device_api", properties.apiVersion);

    uint32_t queue_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, NULL);
    VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
    if (queues == NULL && queue_count != 0) {
        fprintf(stderr, "Could not allocate queue properties\n");
        return false;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, queues);
    bool compute = false;
    for (uint32_t i = 0; i < queue_count; ++i) {
        if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            compute = true;
        }
    }
    free(queues);
    printf("compute_queue=%d\n", compute);

    if (instance_version < VK_API_VERSION_1_3 ||
        properties.apiVersion < VK_API_VERSION_1_3 || get_features == NULL) {
        puts("iree_vulkan_1_3_baseline=missing");
        return false;
    }

    VkPhysicalDeviceVulkan13Features v13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    };
    VkPhysicalDeviceVulkan12Features v12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &v13,
    };
    VkPhysicalDeviceFeatures2 features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &v12,
    };
    get_features(device, &features);

    printf("timelineSemaphore=%u scalarBlockLayout=%u synchronization2=%u\n",
           v12.timelineSemaphore, v12.scalarBlockLayout, v13.synchronization2);
    bool supported = compute && v12.timelineSemaphore && v12.scalarBlockLayout &&
                     v13.synchronization2;
    printf("iree_vulkan_1_3_baseline=%s\n", supported ? "present" : "missing");
    return supported && properties.vendorID == 0x13b5;
}

int main(void) {
    uint32_t loader_version = VK_API_VERSION_1_0;
    PFN_vkEnumerateInstanceVersion enumerate_version =
        (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion");
    if (enumerate_version && enumerate_version(&loader_version) != VK_SUCCESS) {
        fputs("Could not query the Vulkan loader version\n", stderr);
        return 1;
    }
    print_version("loader_api", loader_version);
    uint32_t instance_version =
        loader_version < VK_API_VERSION_1_3 ? loader_version : VK_API_VERSION_1_3;
    VkApplicationInfo application = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "JAX Mali Vulkan probe",
        .apiVersion = instance_version,
    };
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application,
    };
    VkInstance instance;
    VkResult result = vkCreateInstance(&create_info, NULL, &instance);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed: %d\n", (int)result);
        return 1;
    }

    uint32_t device_count = 0;
    result = vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    if (result != VK_SUCCESS || device_count == 0) {
        fprintf(stderr, "No Vulkan physical devices: %d\n", (int)result);
        vkDestroyInstance(instance, NULL);
        return 1;
    }

    VkPhysicalDevice *devices = calloc(device_count, sizeof(*devices));
    if (devices == NULL) {
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    result = vkEnumeratePhysicalDevices(instance, &device_count, devices);
    bool mali_compatible = false;
    if (result == VK_SUCCESS) {
        PFN_vkGetPhysicalDeviceFeatures2 get_features =
            (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr(
                instance, "vkGetPhysicalDeviceFeatures2");
        for (uint32_t i = 0; i < device_count; ++i) {
            mali_compatible |= inspect_device(devices[i], instance_version, get_features);
        }
    } else {
        fprintf(stderr, "Device enumeration failed: %d\n", (int)result);
    }
    free(devices);
    vkDestroyInstance(instance, NULL);
    return result != VK_SUCCESS ? 1 : (mali_compatible ? 0 : 2);
}
