#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "plugin_internal.h"

struct PJRT_Error {
    PJRT_Error_Code code;
    const char *message;
};

static PJRT_Error no_memory = {
    PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not allocate a PJRT error",
};

PJRT_Error *make_error(PJRT_Error_Code code, const char *message) {
    size_t size = strlen(message) + 1;
    PJRT_Error *error = malloc(sizeof(*error) + size);
    if (error == NULL) {
        return &no_memory;
    }
    *error = (PJRT_Error){ .code = code, .message = memcpy(error + 1, message, size) };
    return error;
}

static void error_destroy(PJRT_Error_Destroy_Args *args) {
    if (args->error != &no_memory) {
        free(args->error);
    }
}

static void error_message(PJRT_Error_Message_Args *args) {
    args->message = args->error->message;
    args->message_size = strlen(args->message);
}

static PJRT_Error *error_code(PJRT_Error_GetCode_Args *args) {
    args->code = args->error->code;
    return NULL;
}

static PJRT_Error *plugin_initialize(PJRT_Plugin_Initialize_Args *args) {
    (void)args;
    return NULL;
}

static PJRT_Error *plugin_attributes(PJRT_Plugin_Attributes_Args *args) {
    args->attributes = NULL;
    args->num_attributes = 0;
    return NULL;
}

static PJRT_Error *client_create(PJRT_Client_Create_Args *args) {
    args->client = NULL;
    struct sockaddr_un address;
    socklen_t address_length;
    if (!worker_address(getenv("MALI_PJRT_SOCKET"), &address, &address_length)) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT,
                          "MALI_PJRT_SOCKET must name an abstract Unix socket");
    }

    int worker = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (worker < 0) {
        return make_error(PJRT_Error_Code_UNAVAILABLE, "Could not open the Mali worker socket");
    }

    struct timeval timeout = { .tv_sec = 3 };
    bool connected =
        setsockopt(worker, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
        setsockopt(worker, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0 &&
        connect(worker, (const struct sockaddr *)&address, address_length) == 0;
    struct mali_device_info info;
    bool ready = false;
    if (connected &&
        send(worker, MALI_PJRT_HELLO, sizeof(MALI_PJRT_HELLO) - 1, MSG_NOSIGNAL) ==
            sizeof(MALI_PJRT_HELLO) - 1) {
        ssize_t length = recv(worker, &info, sizeof(info), MSG_TRUNC);
        ready = length == sizeof(info) && info.protocol_version == MALI_PJRT_PROTOCOL_VERSION &&
                info.vendor_id == MALI_VULKAN_VENDOR_ID && info.name[0] != '\0' &&
                memchr(info.name, '\0', sizeof(info.name)) != NULL;
    }

    if (!ready) {
        close(worker);
        return make_error(PJRT_Error_Code_UNAVAILABLE, "Could not handshake with Mali worker");
    }
    // Compiled workloads may take longer than the startup handshake.
    const struct timeval no_receive_timeout = {0};
    if (setsockopt(worker, SOL_SOCKET, SO_RCVTIMEO,
                   &no_receive_timeout, sizeof(no_receive_timeout)) != 0) {
        close(worker);
        return make_error(PJRT_Error_Code_INTERNAL, "Could not configure Mali worker execution");
    }

    PJRT_Client *client = calloc(1, sizeof(*client));
    if (client == NULL) {
        close(worker);
        return &no_memory;
    }
    if (pthread_mutex_init(&client->worker_lock, NULL) != 0) {
        close(worker);
        free(client);
        return make_error(PJRT_Error_Code_INTERNAL, "Could not initialize the worker lock");
    }
    PJRT_Error *error = read_compile_defaults(client, args->create_options, args->num_options);
    if (error != NULL) {
        pthread_mutex_destroy(&client->worker_lock);
        close(worker);
        free(client);
        return error;
    }
    client->worker = worker;
    uint32_t version = info.api_version;
    snprintf(client->platform_version, sizeof(client->platform_version), "Vulkan %u.%u.%u",
             (version >> 22) & 0x7f, (version >> 12) & 0x3ff, version & 0xfff);
    client->device.client = client;
    client->device.description.info = info;
    PJRT_NamedValue *attributes = client->device.description.attributes;
    const char *names[] = {"vendor_id", "device_id", "queue_family"};
    const uint32_t values[] = {info.vendor_id, info.device_id, info.queue_family};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        attributes[i] = (PJRT_NamedValue){
            .struct_size = PJRT_NamedValue_STRUCT_SIZE,
            .name = names[i],
            .name_size = strlen(names[i]),
            .type = PJRT_NamedValue_kInt64,
            .int64_value = values[i],
            .value_size = 1,
        };
    }
    client->devices[0] = &client->device;
    client->memory.client = client;
    client->memories[0] = &client->memory;
    args->client = client;
    return NULL;
}

static PJRT_Error *client_destroy(PJRT_Client_Destroy_Args *args) {
    if (args->client->worker >= 0) {
        close(args->client->worker);
    }
    pthread_mutex_destroy(&args->client->worker_lock);
    free(args->client->default_debug_options);
    free(args->client);
    return NULL;
}

static PJRT_Error *client_platform_name(PJRT_Client_PlatformName_Args *args) {
    args->platform_name = "mali";
    args->platform_name_size = sizeof("mali") - 1;
    return NULL;
}

static PJRT_Error *client_process_index(PJRT_Client_ProcessIndex_Args *args) {
    args->process_index = 0;
    return NULL;
}

static PJRT_Error *client_platform_version(PJRT_Client_PlatformVersion_Args *args) {
    args->platform_version = args->client->platform_version;
    args->platform_version_size = strlen(args->platform_version);
    return NULL;
}

static PJRT_Error *client_topology(PJRT_Client_TopologyDescription_Args *args) {
    args->topology = NULL;
    return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Mali topology description is unavailable");
}

static PJRT_Error *client_devices(PJRT_Client_Devices_Args *args) {
    args->devices = args->client->devices;
    args->num_devices = 1;
    return NULL;
}

static PJRT_Error *client_addressable_devices(PJRT_Client_AddressableDevices_Args *args) {
    args->addressable_devices = args->client->devices;
    args->num_addressable_devices = 1;
    return NULL;
}

static PJRT_Error *client_lookup_device(PJRT_Client_LookupDevice_Args *args) {
    if (args->id != 0) {
        return make_error(PJRT_Error_Code_NOT_FOUND, "Unknown Mali device ID");
    }
    args->device = &args->client->device;
    return NULL;
}

static PJRT_Error *client_lookup_addressable_device(PJRT_Client_LookupAddressableDevice_Args *args) {
    if (args->local_hardware_id != 0) {
        return make_error(PJRT_Error_Code_NOT_FOUND, "Unknown Mali local device ID");
    }
    args->addressable_device = &args->client->device;
    return NULL;
}

static PJRT_Error *client_addressable_memories(PJRT_Client_AddressableMemories_Args *args) {
    args->addressable_memories = args->client->memories;
    args->num_addressable_memories = 1;
    return NULL;
}

static PJRT_Error *device_description(PJRT_Device_GetDescription_Args *args) {
    args->device_description = &args->device->description;
    return NULL;
}

static PJRT_Error *device_id(PJRT_DeviceDescription_Id_Args *args) {
    args->id = 0;
    return NULL;
}

static PJRT_Error *device_process_index(PJRT_DeviceDescription_ProcessIndex_Args *args) {
    args->process_index = 0;
    return NULL;
}

static PJRT_Error *device_attributes(PJRT_DeviceDescription_Attributes_Args *args) {
    args->attributes = args->device_description->attributes;
    args->num_attributes = sizeof(args->device_description->attributes) / sizeof(PJRT_NamedValue);
    return NULL;
}

static PJRT_Error *device_kind(PJRT_DeviceDescription_Kind_Args *args) {
    args->device_kind = args->device_description->info.name;
    args->device_kind_size = strlen(args->device_kind);
    return NULL;
}

static PJRT_Error *device_debug_string(PJRT_DeviceDescription_DebugString_Args *args) {
    args->debug_string = args->device_description->info.name;
    args->debug_string_size = strlen(args->debug_string);
    return NULL;
}

static PJRT_Error *device_to_string(PJRT_DeviceDescription_ToString_Args *args) {
    args->to_string = args->device_description->info.name;
    args->to_string_size = strlen(args->to_string);
    return NULL;
}

static PJRT_Error *device_is_addressable(PJRT_Device_IsAddressable_Args *args) {
    args->is_addressable = true;
    return NULL;
}

static PJRT_Error *device_local_hardware_id(PJRT_Device_LocalHardwareId_Args *args) {
    args->local_hardware_id = 0;
    return NULL;
}

static PJRT_Error *device_addressable_memories(PJRT_Device_AddressableMemories_Args *args) {
    args->memories = args->device->client->memories;
    args->num_memories = 1;
    return NULL;
}

static PJRT_Error *device_default_memory(PJRT_Device_DefaultMemory_Args *args) {
    args->memory = &args->device->client->memory;
    return NULL;
}

static PJRT_Error *device_memory_stats(PJRT_Device_MemoryStats_Args *args) {
    (void)args;
    return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Mali memory statistics are unavailable");
}

static PJRT_Error *memory_id(PJRT_Memory_Id_Args *args) {
    args->id = 0;
    return NULL;
}

static PJRT_Error *memory_kind_id(PJRT_Memory_Kind_Id_Args *args) {
    args->kind_id = 0;
    return NULL;
}

static PJRT_Error *memory_kind(PJRT_Memory_Kind_Args *args) {
    args->kind = "device";
    args->kind_size = sizeof("device") - 1;
    return NULL;
}

static PJRT_Error *memory_debug_string(PJRT_Memory_DebugString_Args *args) {
    args->debug_string = "Mali device memory";
    args->debug_string_size = sizeof("Mali device memory") - 1;
    return NULL;
}

static PJRT_Error *memory_to_string(PJRT_Memory_ToString_Args *args) {
    args->to_string = "Mali device memory";
    args->to_string_size = sizeof("Mali device memory") - 1;
    return NULL;
}

static PJRT_Error *memory_addressable_by_devices(PJRT_Memory_AddressableByDevices_Args *args) {
    args->devices = args->memory->client->devices;
    args->num_devices = 1;
    return NULL;
}

static PJRT_Error *client_default_device_assignment(PJRT_Client_DefaultDeviceAssignment_Args *args) {
    if (args->num_replicas != 1 || args->num_partitions != 1 || args->default_assignment_size < 1) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Mali supports one device");
    }
    args->default_assignment[0] = 0;
    return NULL;
}

static PJRT_Api api = {
        .struct_size = PJRT_Api_STRUCT_SIZE,
        .pjrt_api_version = {
            .struct_size = PJRT_Api_Version_STRUCT_SIZE,
            .major_version = PJRT_API_MAJOR,
            .minor_version = PJRT_API_MINOR,
        },
        .PJRT_Error_Destroy = error_destroy,
        .PJRT_Error_Message = error_message,
        .PJRT_Error_GetCode = error_code,
        .PJRT_Plugin_Initialize = plugin_initialize,
        .PJRT_Plugin_Attributes = plugin_attributes,
        .PJRT_Client_Create = client_create,
        .PJRT_Client_Destroy = client_destroy,
        .PJRT_Client_PlatformName = client_platform_name,
        .PJRT_Client_PlatformVersion = client_platform_version,
        .PJRT_Client_ProcessIndex = client_process_index,
        .PJRT_Client_TopologyDescription = client_topology,
        .PJRT_Client_Devices = client_devices,
        .PJRT_Client_AddressableDevices = client_addressable_devices,
        .PJRT_Client_LookupDevice = client_lookup_device,
        .PJRT_Client_LookupAddressableDevice = client_lookup_addressable_device,
        .PJRT_Client_AddressableMemories = client_addressable_memories,
        .PJRT_Device_GetDescription = device_description,
        .PJRT_DeviceDescription_Id = device_id,
        .PJRT_DeviceDescription_ProcessIndex = device_process_index,
        .PJRT_DeviceDescription_Attributes = device_attributes,
        .PJRT_DeviceDescription_Kind = device_kind,
        .PJRT_DeviceDescription_DebugString = device_debug_string,
        .PJRT_DeviceDescription_ToString = device_to_string,
        .PJRT_Device_IsAddressable = device_is_addressable,
        .PJRT_Device_LocalHardwareId = device_local_hardware_id,
        .PJRT_Device_AddressableMemories = device_addressable_memories,
        .PJRT_Device_DefaultMemory = device_default_memory,
        .PJRT_Device_MemoryStats = device_memory_stats,
        .PJRT_Memory_Id = memory_id,
        .PJRT_Memory_Kind = memory_kind,
        .PJRT_Memory_Kind_Id = memory_kind_id,
        .PJRT_Memory_DebugString = memory_debug_string,
        .PJRT_Memory_ToString = memory_to_string,
        .PJRT_Memory_AddressableByDevices = memory_addressable_by_devices,
        .PJRT_Client_DefaultDeviceAssignment = client_default_device_assignment,
};

static void initialize_api(void) {
    register_buffer_api(&api);
    register_executable_api(&api);
}

__attribute__((visibility("default")))
const PJRT_Api *GetPjrtApi(void) {
    static pthread_once_t initialized = PTHREAD_ONCE_INIT;
    pthread_once(&initialized, initialize_api);
    return &api;
}
