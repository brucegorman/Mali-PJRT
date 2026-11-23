#include <stdlib.h>
#include <unistd.h>

#include "plugin_internal.h"

struct PJRT_Buffer {
    PJRT_Client *client;
    uint64_t handle;
    size_t size;
    size_t rank;
    PJRT_Buffer_Type type;
    int64_t dims[];
};

struct PJRT_Event {
    bool deleted_buffer;
};

// Transfers are synchronous, so all events can share immutable terminal states.
static PJRT_Event ready_event;
static PJRT_Event deleted_event = {.deleted_buffer = true};

static PJRT_Error *event_status(PJRT_Event *event) {
    return event->deleted_buffer
        ? make_error(PJRT_Error_Code_FAILED_PRECONDITION, "Mali buffer was deleted")
        : NULL;
}

static PJRT_Error *event_destroy(PJRT_Event_Destroy_Args *args) {
    (void)args;
    return NULL;
}

static PJRT_Error *event_is_ready(PJRT_Event_IsReady_Args *args) {
    args->is_ready = true;
    return NULL;
}

static PJRT_Error *event_error(PJRT_Event_Error_Args *args) {
    return event_status(args->event);
}

static PJRT_Error *event_await(PJRT_Event_Await_Args *args) {
    return event_status(args->event);
}

static PJRT_Error *event_on_ready(PJRT_Event_OnReady_Args *args) {
    args->callback(event_status(args->event), args->user_arg);
    return NULL;
}

// The caller holds worker_lock through the whole transfer. A failed transport
// closes the connection so the worker can release all remaining allocations.
static int exchange(PJRT_Client *client, const struct mali_request *request,
                    const void *input, void *output, struct mali_reply *reply) {
    if (client->worker < 0) {
        return -1;
    }
    size_t write_size = request->operation == MALI_WRITE ? (size_t)request->length : 0;
    size_t read_size = request->operation == MALI_READ ? (size_t)request->length : 0;
    if (send_packet(client->worker, request, sizeof(*request), input, write_size)) {
        ssize_t count = receive_packet(client->worker, reply, sizeof(*reply), output, read_size);
        if (count >= (ssize_t)sizeof(*reply) &&
            count == (ssize_t)(sizeof(*reply) + (reply->status == MALI_OK ? read_size : 0))) {
            return (int)reply->status;
        }
    }
    close(client->worker);
    client->worker = -1;
    return -1;
}

static PJRT_Error *transfer_error(int status) {
    switch (status) {
        case MALI_OK: return NULL;
        case MALI_OUT_OF_MEMORY:
            return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not allocate a Vulkan buffer");
        case MALI_UNSUPPORTED_MEMORY:
            return make_error(PJRT_Error_Code_UNIMPLEMENTED, "No coherent Vulkan memory for this buffer");
        case MALI_NOT_FOUND:
            return make_error(PJRT_Error_Code_FAILED_PRECONDITION, "Vulkan buffer is gone");
        case -1:
            return make_error(PJRT_Error_Code_UNAVAILABLE, "Lost connection to Mali worker");
        default:
            return make_error(PJRT_Error_Code_INTERNAL, "Mali worker rejected the buffer request");
    }
}

static size_t element_size(PJRT_Buffer_Type type) {
    switch (type) {
        case PJRT_Buffer_Type_PRED:
        case PJRT_Buffer_Type_S8:
        case PJRT_Buffer_Type_U8:
            return 1;
        case PJRT_Buffer_Type_S16:
        case PJRT_Buffer_Type_U16:
        case PJRT_Buffer_Type_F16:
        case PJRT_Buffer_Type_BF16:
            return 2;
        case PJRT_Buffer_Type_S32:
        case PJRT_Buffer_Type_U32:
        case PJRT_Buffer_Type_F32:
            return 4;
        case PJRT_Buffer_Type_S64:
        case PJRT_Buffer_Type_U64:
        case PJRT_Buffer_Type_F64:
        case PJRT_Buffer_Type_C64:
            return 8;
        case PJRT_Buffer_Type_C128:
            return 16;
        default:
            return 0;
    }
}

static bool descending_layout(const PJRT_Buffer_MemoryLayout *layout, size_t rank) {
    if (layout == NULL) {
        return true;
    }
    if (layout->type != PJRT_Buffer_MemoryLayout_Type_Tiled ||
        layout->tiled.num_tiles != 0 || layout->tiled.minor_to_major_size != rank ||
        (rank != 0 && layout->tiled.minor_to_major == NULL)) {
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (layout->tiled.minor_to_major[i] != (int64_t)(rank - 1 - i)) {
            return false;
        }
    }
    return true;
}

static bool array_size(const int64_t *dims, size_t rank, size_t item_size, size_t *size) {
    bool empty = false;
    for (size_t i = 0; i < rank; ++i) {
        if (dims[i] < 0) {
            return false;
        }
        empty |= dims[i] == 0;
    }
    *size = empty ? 0 : item_size;
    for (size_t i = 0; !empty && i < rank; ++i) {
        if ((size_t)dims[i] > (size_t)INT64_MAX / *size) {
            return false;
        }
        *size *= (size_t)dims[i];
    }
    return true;
}

static bool check_strides(const PJRT_Client_BufferFromHostBuffer_Args *args,
                          size_t item_size, bool *dense) {
    *dense = true;
    if (args->num_byte_strides == 0) {
        return true;
    }
    if (args->num_byte_strides != args->num_dims || args->byte_strides == NULL) {
        return false;
    }
    int64_t positive = 0, negative = 0;
    size_t expected = item_size;
    for (size_t i = args->num_dims; i-- > 0;) {
        int64_t stride = args->byte_strides[i];
        int64_t dim = args->dims[i];
        if (dim > 1) {
            *dense &= stride == (int64_t)expected;
            int64_t axis;
            if (__builtin_mul_overflow(stride, dim - 1, &axis)) {
                return false;
            }
            int64_t *bound = axis < 0 ? &negative : &positive;
            if (__builtin_add_overflow(*bound, axis, bound)) {
                return false;
            }
        }
        expected *= (size_t)dim;
    }
    return true;
}

static void pack_host_array(const PJRT_Client_BufferFromHostBuffer_Args *args,
                            size_t item_size, size_t size, unsigned char *packed) {
    const unsigned char *source = args->data;
    for (size_t i = 0; i < size / item_size; ++i) {
        size_t index = i;
        int64_t offset = 0;
        for (size_t axis = args->num_dims; axis-- > 0;) {
            size_t dim = (size_t)args->dims[axis];
            offset += (int64_t)(index % dim) * args->byte_strides[axis];
            index /= dim;
        }
        memcpy(packed + i * item_size, source + offset, item_size);
    }
}

static void release_handle(PJRT_Client *client, uint64_t handle) {
    struct mali_request request = {.operation = MALI_RELEASE, .handle = handle};
    struct mali_reply reply;
    exchange(client, &request, NULL, NULL, &reply);
}

static int transfer(PJRT_Buffer *buffer, enum mali_operation operation,
                    size_t offset, size_t size, const void *input, void *output) {
    struct mali_reply reply;
    while (size > 0) {
        const size_t chunk = size < MALI_TRANSFER_CHUNK ? size : MALI_TRANSFER_CHUNK;
        struct mali_request request = {
            .operation = operation, .handle = buffer->handle, .offset = offset, .length = chunk,
        };
        int status = exchange(buffer->client, &request, input, output, &reply);
        if (status != MALI_OK) {
            return status;
        }
        if (input != NULL) {
            input = (const unsigned char *)input + chunk;
        }
        if (output != NULL) {
            output = (unsigned char *)output + chunk;
        }
        offset += chunk;
        size -= chunk;
    }
    return MALI_OK;
}

static PJRT_Error *client_buffer_from_host(PJRT_Client_BufferFromHostBuffer_Args *args) {
    args->buffer = NULL;
    args->done_with_host_buffer = NULL;
    if ((args->device != NULL && args->device != &args->client->device) ||
        (args->memory != NULL && args->memory != &args->client->memory) ||
        (args->num_dims != 0 && args->dims == NULL) ||
        args->num_dims > (SIZE_MAX - sizeof(PJRT_Buffer)) / (2 * sizeof(int64_t))) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid Mali buffer destination or rank");
    }
    size_t item_size = element_size(args->type);
    if (item_size == 0 || !descending_layout(args->device_layout, args->num_dims)) {
        return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Unsupported Mali buffer type or layout");
    }
    size_t size;
    if (!array_size(args->dims, args->num_dims, item_size, &size) ||
        (size != 0 && args->data == NULL) ||
        (args->num_byte_strides != 0 && args->num_byte_strides != args->num_dims)) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid Mali buffer shape or data");
    }
    bool dense = true;
    if (size != 0 && !check_strides(args, item_size, &dense)) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid Mali buffer strides");
    }
    PJRT_Buffer *buffer = calloc(1, sizeof(*buffer) + args->num_dims * 2 * sizeof(int64_t));
    if (buffer == NULL) {
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not record Mali buffer");
    }
    buffer->client = args->client;
    buffer->size = size;
    buffer->rank = args->num_dims;
    buffer->type = args->type;
    for (size_t i = 0; i < buffer->rank; ++i) {
        buffer->dims[i] = args->dims[i];
        buffer->dims[buffer->rank + i] = (int64_t)(buffer->rank - 1 - i);
    }
    unsigned char *packed = NULL;
    if (!dense) {
        packed = malloc(size);
        if (packed == NULL) {
            free(buffer);
            return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not pack strided host data");
        }
        pack_host_array(args, item_size, size, packed);
    }

    struct mali_request request = {.operation = MALI_ALLOCATE, .length = size};
    struct mali_reply reply;
    pthread_mutex_lock(&args->client->worker_lock);
    int status = exchange(args->client, &request, NULL, NULL, &reply);
    if (status == MALI_OK) {
        buffer->handle = reply.handle;
        status = transfer(buffer, MALI_WRITE, 0, size, packed != NULL ? packed : args->data, NULL);
        if (status != MALI_OK) {
            release_handle(args->client, buffer->handle);
        }
    }
    pthread_mutex_unlock(&args->client->worker_lock);
    free(packed);
    if (status != MALI_OK) {
        free(buffer);
        return transfer_error(status);
    }
    args->buffer = buffer;
    args->done_with_host_buffer = &ready_event;
    return NULL;
}

static PJRT_Error *download(PJRT_Buffer *buffer, size_t offset, size_t size, void *output,
                            PJRT_Event **event) {
    *event = NULL;
    pthread_mutex_lock(&buffer->client->worker_lock);
    int status = buffer->handle != 0
        ? transfer(buffer, MALI_READ, offset, size, NULL, output)
        : MALI_NOT_FOUND;
    pthread_mutex_unlock(&buffer->client->worker_lock);
    if (status == MALI_OK) {
        *event = &ready_event;
    }
    return transfer_error(status);
}

static PJRT_Error *buffer_to_host(PJRT_Buffer_ToHostBuffer_Args *args) {
    args->event = NULL;
    if (!descending_layout(args->host_layout, args->src->rank)) {
        return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Unsupported Mali readback layout");
    }
    if (args->dst == NULL) {
        args->dst_size = args->src->size;
        args->event = &ready_event;
        return NULL;
    }
    if (args->dst_size < args->src->size) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Mali readback destination is too small");
    }
    return download(args->src, 0, args->src->size, args->dst, &args->event);
}

static PJRT_Error *buffer_copy_raw_to_host(PJRT_Buffer_CopyRawToHost_Args *args) {
    args->event = NULL;
    if (args->offset < 0 || args->transfer_size < 0 || (size_t)args->offset > args->buffer->size ||
        (size_t)args->transfer_size > args->buffer->size - (size_t)args->offset ||
        (args->transfer_size != 0 && args->dst == NULL)) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid Mali readback byte range");
    }
    return download(args->buffer, (size_t)args->offset, (size_t)args->transfer_size,
                    args->dst, &args->event);
}

static void delete_buffer(PJRT_Buffer *buffer) {
    pthread_mutex_lock(&buffer->client->worker_lock);
    if (buffer->handle != 0) {
        release_handle(buffer->client, buffer->handle);
        buffer->handle = 0;
    }
    pthread_mutex_unlock(&buffer->client->worker_lock);
}

static PJRT_Error *buffer_delete(PJRT_Buffer_Delete_Args *args) {
    delete_buffer(args->buffer);
    return NULL;
}

static PJRT_Error *buffer_destroy(PJRT_Buffer_Destroy_Args *args) {
    if (args->buffer != NULL) {
        delete_buffer(args->buffer);
        free(args->buffer);
    }
    return NULL;
}

static bool is_deleted(PJRT_Buffer *buffer) {
    pthread_mutex_lock(&buffer->client->worker_lock);
    bool deleted = buffer->handle == 0;
    pthread_mutex_unlock(&buffer->client->worker_lock);
    return deleted;
}

static PJRT_Error *buffer_is_deleted(PJRT_Buffer_IsDeleted_Args *args) {
    args->is_deleted = is_deleted(args->buffer);
    return NULL;
}

static PJRT_Error *buffer_ready_event(PJRT_Buffer_ReadyEvent_Args *args) {
    args->event = is_deleted(args->buffer) ? &deleted_event : &ready_event;
    return NULL;
}

static PJRT_Error *buffer_element_type(PJRT_Buffer_ElementType_Args *args) {
    args->type = args->buffer->type;
    return NULL;
}

static PJRT_Error *buffer_dimensions(PJRT_Buffer_Dimensions_Args *args) {
    args->dims = args->buffer->dims;
    args->num_dims = args->buffer->rank;
    return NULL;
}

static PJRT_Error *buffer_unpadded_dimensions(PJRT_Buffer_UnpaddedDimensions_Args *args) {
    args->unpadded_dims = args->buffer->dims;
    args->num_dims = args->buffer->rank;
    return NULL;
}

static PJRT_Error *buffer_dynamic_dimensions(PJRT_Buffer_DynamicDimensionIndices_Args *args) {
    args->dynamic_dim_indices = NULL;
    args->num_dynamic_dims = 0;
    return NULL;
}

static PJRT_Error *buffer_layout(PJRT_Buffer_GetMemoryLayout_Args *args) {
    args->layout = (PJRT_Buffer_MemoryLayout){
        .struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE,
        .type = PJRT_Buffer_MemoryLayout_Type_Tiled,
        .tiled = {
            .struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE,
            .minor_to_major = args->buffer->dims + args->buffer->rank,
            .minor_to_major_size = args->buffer->rank,
        },
    };
    return NULL;
}

static PJRT_Error *buffer_size(PJRT_Buffer_OnDeviceSizeInBytes_Args *args) {
    args->on_device_size_in_bytes = args->buffer->size;
    return NULL;
}

static PJRT_Error *buffer_is_on_cpu(PJRT_Buffer_IsOnCpu_Args *args) {
    args->is_on_cpu = false;
    return NULL;
}

static PJRT_Error *buffer_device(PJRT_Buffer_Device_Args *args) {
    args->device = &args->buffer->client->device;
    return NULL;
}

static PJRT_Error *buffer_memory(PJRT_Buffer_Memory_Args *args) {
    args->memory = &args->buffer->client->memory;
    return NULL;
}

void register_buffer_api(PJRT_Api *api) {
    api->PJRT_Event_Destroy = event_destroy;
    api->PJRT_Event_IsReady = event_is_ready;
    api->PJRT_Event_Error = event_error;
    api->PJRT_Event_Await = event_await;
    api->PJRT_Event_OnReady = event_on_ready;
    api->PJRT_Client_BufferFromHostBuffer = client_buffer_from_host;
    api->PJRT_Buffer_ToHostBuffer = buffer_to_host;
    api->PJRT_Buffer_CopyRawToHost = buffer_copy_raw_to_host;
    api->PJRT_Buffer_Destroy = buffer_destroy;
    api->PJRT_Buffer_Delete = buffer_delete;
    api->PJRT_Buffer_IsDeleted = buffer_is_deleted;
    api->PJRT_Buffer_ReadyEvent = buffer_ready_event;
    api->PJRT_Buffer_ElementType = buffer_element_type;
    api->PJRT_Buffer_Dimensions = buffer_dimensions;
    api->PJRT_Buffer_UnpaddedDimensions = buffer_unpadded_dimensions;
    api->PJRT_Buffer_DynamicDimensionIndices = buffer_dynamic_dimensions;
    api->PJRT_Buffer_GetMemoryLayout = buffer_layout;
    api->PJRT_Buffer_OnDeviceSizeInBytes = buffer_size;
    api->PJRT_Buffer_IsOnCpu = buffer_is_on_cpu;
    api->PJRT_Buffer_Device = buffer_device;
    api->PJRT_Buffer_Memory = buffer_memory;
}
