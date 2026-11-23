#ifndef MALI_PJRT_PROTOCOL_H_
#define MALI_PJRT_PROTOCOL_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>

#define MALI_PJRT_HELLO "MALI-PJRT/2 HELLO"
#define MALI_PJRT_PROTOCOL_VERSION 2
#define MALI_VULKAN_VENDOR_ID 0x13b5u

// Both peers use little-endian arm64. This packet contains no process-local pointers.
struct mali_device_info {
    uint32_t protocol_version;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t api_version;
    uint32_t queue_family;
    char name[256];
};

_Static_assert(sizeof(struct mali_device_info) == 276, "Unexpected device packet layout");

static inline bool worker_address(const char *name, struct sockaddr_un *address,
                                  socklen_t *address_length) {
    if (name == NULL || name[0] == '\0') {
        return false;
    }
    size_t name_length = strlen(name);
    if (name_length >= sizeof(address->sun_path)) {
        return false;
    }
    *address = (struct sockaddr_un){ .sun_family = AF_UNIX };
    // A leading zero selects Linux's abstract Unix socket namespace.
    memcpy(address->sun_path + 1, name, name_length);
    *address_length = (socklen_t)(offsetof(struct sockaddr_un, sun_path) + 1 + name_length);
    return true;
}

#endif
