#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include "executable_image.h"

// Disk representation for little-endian arm64. Bump the format version when
// changing the header, compiler target flags, or executable loading ABI.
static const char image_magic[8] = "MALIVMFB";
static const uint32_t image_version = 1;
static const char iree_revision[40] = "e4a3b0405d7d23554da26403658d0e8c3c5ecf25";

struct image_header {
    char magic[8];
    uint32_t version;
    uint32_t reserved;
    uint64_t vmfb_size;
    uint64_t checksum;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t api_version;
    uint32_t driver_version;
    unsigned char pipeline_cache_uuid[16];
    char compiler_revision[40];
};

struct executable_image {
    atomic_size_t references;
    struct image_header header;
    char vmfb[];
};

_Static_assert(sizeof(struct image_header) == 104, "Unexpected executable header layout");
_Static_assert(offsetof(struct executable_image, vmfb) ==
               offsetof(struct executable_image, header) + sizeof(struct image_header),
               "Unexpected executable image layout");

static uint64_t checksum(const char *data, size_t size) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ (unsigned char)data[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static struct executable_image *allocate_image(size_t size) {
    if (size > SIZE_MAX - sizeof(struct executable_image)) {
        return NULL;
    }
    struct executable_image *image = malloc(sizeof(*image) + size);
    if (image != NULL) {
        atomic_init(&image->references, 1);
    }
    return image;
}

static PJRT_Error *allocation_error(void) {
    return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not retain the Mali executable image");
}

static bool read_all(int fd, char *data, size_t size) {
    while (size != 0) {
        ssize_t count = read(fd, data, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        data += count;
        size -= (size_t)count;
    }
    return true;
}

PJRT_Error *read_compiled_image(const char *path, const struct mali_device_info *device,
                                struct executable_image **out) {
    *out = NULL;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return make_error(PJRT_Error_Code_INTERNAL, "Could not open the compiled Mali module");
    }
    struct stat file;
    if (fstat(fd, &file) != 0 || file.st_size <= 0 ||
        (uint64_t)file.st_size > SIZE_MAX - sizeof(struct executable_image)) {
        close(fd);
        return make_error(PJRT_Error_Code_INTERNAL, "IREE produced an invalid Mali module size");
    }
    size_t size = (size_t)file.st_size;
    struct executable_image *image = allocate_image(size);
    if (image == NULL) {
        close(fd);
        return allocation_error();
    }
    bool read = read_all(fd, image->vmfb, size);
    close(fd);
    if (!read) {
        release_image(image);
        return make_error(PJRT_Error_Code_INTERNAL, "Could not read the compiled Mali module");
    }
    image->header = (struct image_header){
        .version = image_version,
        .vmfb_size = size,
        .checksum = checksum(image->vmfb, size),
        .vendor_id = device->vendor_id,
        .device_id = device->device_id,
        .api_version = device->api_version,
        .driver_version = device->driver_version,
    };
    memcpy(image->header.magic, image_magic, sizeof(image_magic));
    memcpy(image->header.pipeline_cache_uuid, device->pipeline_cache_uuid,
           sizeof(image->header.pipeline_cache_uuid));
    memcpy(image->header.compiler_revision, iree_revision, sizeof(iree_revision));
    *out = image;
    return NULL;
}

PJRT_Error *read_serialized_image(const char *data, size_t size,
                                  const struct mali_device_info *device,
                                  struct executable_image **out) {
    *out = NULL;
    struct image_header header;
    if (data == NULL || size <= sizeof(header)) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Mali executable image is truncated");
    }
    memcpy(&header, data, sizeof(header));
    if (memcmp(header.magic, image_magic, sizeof(image_magic)) != 0 ||
        header.version != image_version || header.reserved != 0 ||
        header.vmfb_size != size - sizeof(header)) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid Mali executable image header");
    }
    if (header.vendor_id != device->vendor_id || header.device_id != device->device_id ||
        header.api_version != device->api_version || header.driver_version != device->driver_version ||
        memcmp(header.pipeline_cache_uuid, device->pipeline_cache_uuid,
               sizeof(header.pipeline_cache_uuid)) != 0 ||
        memcmp(header.compiler_revision, iree_revision, sizeof(iree_revision)) != 0) {
        return make_error(PJRT_Error_Code_FAILED_PRECONDITION,
                          "Mali executable image targets a different device, driver, or compiler");
    }
    const char *vmfb = data + sizeof(header);
    if (checksum(vmfb, header.vmfb_size) != header.checksum) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Mali executable image checksum mismatch");
    }
    struct executable_image *image = allocate_image(header.vmfb_size);
    if (image == NULL) {
        return allocation_error();
    }
    image->header = header;
    memcpy(image->vmfb, vmfb, header.vmfb_size);
    *out = image;
    return NULL;
}

void retain_image(struct executable_image *image) {
    atomic_fetch_add_explicit(&image->references, 1, memory_order_relaxed);
}

void release_image(struct executable_image *image) {
    if (atomic_fetch_sub_explicit(&image->references, 1, memory_order_acq_rel) == 1) {
        free(image);
    }
}

const char *image_vmfb(const struct executable_image *image) {
    return image->vmfb;
}

size_t image_vmfb_size(const struct executable_image *image) {
    return image->header.vmfb_size;
}

uint64_t image_fingerprint(const struct executable_image *image) {
    return image->header.checksum;
}

const char *image_serialized(const struct executable_image *image, size_t *size) {
    *size = sizeof(image->header) + image->header.vmfb_size;
    return (const char *)&image->header;
}
