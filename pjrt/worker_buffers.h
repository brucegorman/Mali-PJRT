#ifndef MALI_PJRT_WORKER_BUFFERS_H_
#define MALI_PJRT_WORKER_BUFFERS_H_

#include "iree/runtime/api.h"

int serve_buffers(int socket, iree_runtime_instance_t *instance, iree_hal_device_t *device);

#endif
