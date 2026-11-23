#ifndef MALI_PJRT_PROTOCOL_H_
#define MALI_PJRT_PROTOCOL_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>

#define MALI_PJRT_HELLO "MALI-PJRT/5 HELLO"
#define MALI_PJRT_PROTOCOL_VERSION 5
#define MALI_VULKAN_VENDOR_ID 0x13b5u
#define MALI_TRANSFER_CHUNK (64 * 1024u)
#define MALI_MAX_RANK 64
#define MALI_MAX_OUTPUTS 32

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

enum mali_operation {
    MALI_ALLOCATE = 1,
    MALI_WRITE,
    MALI_READ,
    MALI_RELEASE,
    MALI_LOAD,
    MALI_EXECUTE,
    MALI_UNLOAD,
};

enum mali_status {
    MALI_OK = 0,
    MALI_INVALID,
    MALI_NOT_FOUND,
    MALI_OUT_OF_MEMORY,
    MALI_UNSUPPORTED_MEMORY,
    MALI_RUNTIME_ERROR,
    MALI_UNSUPPORTED_SIGNATURE,
};

struct mali_tensor {
    uint64_t size;
    uint32_t type;
    uint32_t rank;
    int64_t dims[MALI_MAX_RANK];
};

struct mali_result {
    uint64_t handle;
    struct mali_tensor tensor;
};

struct mali_executable_info {
    uint32_t inputs;
    uint32_t outputs;
    struct mali_tensor output_tensors[MALI_MAX_OUTPUTS];
};

struct mali_request {
    uint64_t handle;
    uint64_t offset;
    uint64_t length;
    uint32_t operation;
    uint32_t reserved;
};

struct mali_reply {
    uint64_t handle;
    uint32_t status;
    uint32_t reserved;
};

_Static_assert(sizeof(struct mali_request) == 32 && sizeof(struct mali_reply) == 16,
               "Unexpected buffer packet layout");
_Static_assert(sizeof(struct mali_tensor) == 528 && sizeof(struct mali_result) == 536,
               "Unexpected tensor packet layout");

static inline bool send_packet(int socket, const void *header, size_t header_size,
                               const void *data, size_t data_size) {
    struct iovec iov[] = {{(void *)header, header_size}, {(void *)data, data_size}};
    struct msghdr message = {.msg_iov = iov, .msg_iovlen = data_size ? 2 : 1};
    ssize_t count;
    do {
        count = sendmsg(socket, &message, MSG_NOSIGNAL);
    } while (count < 0 && errno == EINTR);
    return count == (ssize_t)(header_size + data_size);
}

static inline ssize_t receive_packet(int socket, void *header, size_t header_size,
                                     void *data, size_t data_size) {
    struct iovec iov[] = {{header, header_size}, {data, data_size}};
    struct msghdr message = {.msg_iov = iov, .msg_iovlen = data_size ? 2 : 1};
    ssize_t count;
    do {
        count = recvmsg(socket, &message, MSG_TRUNC);
    } while (count < 0 && errno == EINTR);
    return count;
}

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
