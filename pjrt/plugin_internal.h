#ifndef MALI_PJRT_PLUGIN_INTERNAL_H_
#define MALI_PJRT_PLUGIN_INTERNAL_H_

#include <pthread.h>
#include <stdatomic.h>

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
    // Protects the queues and each public handle's resource pointer. Never
    // hold this lock across a socket exchange or while waiting for an event.
    pthread_mutex_t dispatch_lock;
    pthread_cond_t work_available;
    pthread_cond_t callback_available;
    pthread_t dispatch_thread;
    pthread_t callback_thread;
    struct mali_execution *work_head;
    struct mali_execution *work_tail;
    struct mali_callback *callback_head;
    struct mali_callback *callback_tail;
    bool stop_dispatch;
    bool stop_callbacks;
    bool destroy_on_callback_exit;
    char platform_version[64];
    PJRT_Device device;
    PJRT_Device *devices[1];
    PJRT_Memory memory;
    PJRT_Memory *memories[1];
    unsigned char *default_debug_options;
    size_t default_debug_options_size;
};

struct mali_resource {
    PJRT_Client *client;
    PJRT_Event *ready;
    enum mali_operation release;
    atomic_size_t references;
    uint64_t handle;
};

struct PJRT_Buffer {
    PJRT_Client *client;
    struct mali_resource *resource;
    size_t size;
    size_t rank;
    PJRT_Buffer_Type type;
    int64_t dims[];
};

struct mali_execution {
    struct mali_execution *next;
    struct mali_resource *executable;
    PJRT_Event *done;
    size_t inputs;
    size_t outputs;
    size_t output_sizes[MALI_MAX_OUTPUTS];
    struct mali_resource *buffers[];  // inputs followed by outputs
};

PJRT_Error *make_error(PJRT_Error_Code code, const char *message);
int exchange(PJRT_Client *client, const struct mali_request *request,
             const void *input, size_t input_size, void *output, size_t output_size,
             struct mali_reply *reply);
PJRT_Error *transfer_error(enum mali_operation operation, int status);
bool tensor_size(const struct mali_tensor *tensor, size_t *size);
PJRT_Buffer *create_buffer(PJRT_Client *client, const struct mali_tensor *tensor, PJRT_Event *ready);
struct mali_resource *create_resource(PJRT_Client *client, enum mali_operation release,
                                      uint64_t handle, PJRT_Event *ready);
void retain_resource(struct mali_resource *resource);
void release_resource(struct mali_resource *resource);
void delete_resource(PJRT_Client *client, struct mali_resource **resource);
bool resource_deleted(PJRT_Client *client, struct mali_resource **resource);
bool start_dispatch(PJRT_Client *client);
void stop_dispatch(PJRT_Client *client);
void free_client(PJRT_Client *client);
void enqueue_execution(PJRT_Client *client, struct mali_execution *execution);
PJRT_Event *ready_event(void);
PJRT_Event *deleted_event(void);
PJRT_Event *create_event(void);
void retain_event(PJRT_Event *event);
void release_event(PJRT_Event *event);
int await_event(PJRT_Event *event);
void register_event_api(PJRT_Api *api);
void register_buffer_api(PJRT_Api *api);
void register_executable_api(PJRT_Api *api);
PJRT_Error *read_compile_defaults(PJRT_Client *client, const PJRT_NamedValue *options,
                                  size_t count);
PJRT_Error *validate_compile_options(const PJRT_Client *client, const char *bytes, size_t size);

#endif
