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
};

PJRT_Error *make_error(PJRT_Error_Code code, const char *message);
void register_buffer_api(PJRT_Api *api);

#endif
