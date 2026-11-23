#ifndef MALI_PJRT_EXECUTABLE_IMAGE_H_
#define MALI_PJRT_EXECUTABLE_IMAGE_H_

#include "plugin_internal.h"

struct executable_image;

PJRT_Error *read_compiled_image(const char *path, const struct mali_device_info *device,
                                struct executable_image **image);
PJRT_Error *read_serialized_image(const char *data, size_t size,
                                  const struct mali_device_info *device,
                                  struct executable_image **image);
void retain_image(struct executable_image *image);
void release_image(struct executable_image *image);
const char *image_vmfb(const struct executable_image *image);
size_t image_vmfb_size(const struct executable_image *image);
uint64_t image_fingerprint(const struct executable_image *image);
const char *image_serialized(const struct executable_image *image, size_t *size);

#endif
