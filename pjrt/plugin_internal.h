#ifndef MALI_PJRT_PLUGIN_INTERNAL_H_
#define MALI_PJRT_PLUGIN_INTERNAL_H_

#include <pthread.h>

#include "pjrt_c_api.h"
#include "protocol.h"

struct PJRT_DeviceDescription {
    struct mali_device_info info;
    PJRT_NamedValue attributes[3];
};

struct PJRT_Device {
    PJRT_Client *client;
    PJRT_DeviceDescription description;
};

struct PJRT_Memory {
    PJRT_Client *client;
};

struct PJRT_Client {
    int worker;
    pthread_mutex_t worker_lock;
    char platform_version[64];
    PJRT_Device device;
    PJRT_Device *devices[1];
    PJRT_Memory memory;
    PJRT_Memory *memories[1];
    unsigned char *default_debug_options;
    size_t default_debug_options_size;
};

struct PJRT_Buffer {
    PJRT_Client *client;
    uint64_t handle;
    size_t size;
    size_t rank;
    PJRT_Buffer_Type type;
    int64_t dims[];
};

PJRT_Error *make_error(PJRT_Error_Code code, const char *message);
int exchange(PJRT_Client *client, const struct mali_request *request,
             const void *input, size_t input_size, void *output, size_t output_size,
             struct mali_reply *reply);
PJRT_Error *transfer_error(enum mali_operation operation, int status);
PJRT_Buffer *create_buffer(PJRT_Client *client, const struct mali_tensor *tensor, uint64_t handle);
void release_handle(PJRT_Client *client, uint64_t handle);
PJRT_Event *buffer_ready(void);
void register_buffer_api(PJRT_Api *api);
void register_executable_api(PJRT_Api *api);
PJRT_Error *read_compile_defaults(PJRT_Client *client, const PJRT_NamedValue *options,
                                  size_t count);
PJRT_Error *validate_compile_options(const PJRT_Client *client, const char *bytes, size_t size);

#endif
