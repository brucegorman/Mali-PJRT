#ifndef MALI_FENCE_PROBE_H_
#define MALI_FENCE_PROBE_H_

#include "iree/runtime/api.h"

int run_fence_probe(iree_runtime_instance_t *instance, iree_hal_device_t *device, const char *path);

#endif
