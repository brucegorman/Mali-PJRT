#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "plugin_internal.h"

enum transport_status {
    TRANSPORT_CLOSED = -4,
    TRANSPORT_SEND_FAILED,
    TRANSPORT_RECEIVE_FAILED,
    TRANSPORT_INVALID_REPLY,
};

// The caller holds worker_lock through the whole transfer. A failed transport
// closes the connection so the worker can release all remaining allocations.
int exchange(PJRT_Client *client, const struct mali_request *request,
             const void *input, size_t input_size, void *output, size_t output_size,
             struct mali_reply *reply) {
    if (client->worker < 0) {
        return TRANSPORT_CLOSED;
    }
    int status = TRANSPORT_SEND_FAILED;
    if (send_packet(client->worker, request, sizeof(*request), input, input_size)) {
        ssize_t count = receive_packet(client->worker, reply, sizeof(*reply), output, output_size);
        if (count >= (ssize_t)sizeof(*reply) &&
            reply->status <= MALI_UNSUPPORTED_SIGNATURE && reply->reserved == 0 &&
            count == (ssize_t)(sizeof(*reply) + (reply->status == MALI_OK ? output_size : 0))) {
            return (int)reply->status;
        }
        status = count <= 0 ? TRANSPORT_RECEIVE_FAILED : TRANSPORT_INVALID_REPLY;
    }
    close(client->worker);
    client->worker = -1;
    return status;
}

PJRT_Error *transfer_error(enum mali_operation operation, int status) {
    static const char *const names[] = {
        [MALI_ALLOCATE] = "allocate", [MALI_WRITE] = "write", [MALI_READ] = "read",
        [MALI_LOAD] = "load", [MALI_EXECUTE] = "execute",
    };
    PJRT_Error_Code code = PJRT_Error_Code_INTERNAL;
    const char *detail;
    switch (status) {
        case MALI_OK: return NULL;
        case MALI_OUT_OF_MEMORY:
            code = PJRT_Error_Code_RESOURCE_EXHAUSTED;
            detail = "Vulkan runtime ran out of resources";
            break;
        case MALI_UNSUPPORTED_MEMORY:
            code = PJRT_Error_Code_UNIMPLEMENTED;
            detail = "No coherent Vulkan memory for this buffer";
            break;
        case MALI_NOT_FOUND:
            code = PJRT_Error_Code_FAILED_PRECONDITION;
            detail = "Mali handle is gone";
            break;
        case MALI_RUNTIME_ERROR:
            detail = "IREE Vulkan runtime failed (see worker stderr)";
            break;
        case MALI_UNSUPPORTED_SIGNATURE:
            code = PJRT_Error_Code_UNIMPLEMENTED;
            detail = "Mali requires a supported static tensor output signature";
            break;
        case TRANSPORT_CLOSED:
            code = PJRT_Error_Code_UNAVAILABLE;
            detail = "worker connection is closed";
            break;
        case TRANSPORT_SEND_FAILED:
            code = PJRT_Error_Code_UNAVAILABLE;
            detail = "could not send request";
            break;
        case TRANSPORT_RECEIVE_FAILED:
            code = PJRT_Error_Code_UNAVAILABLE;
            detail = "could not receive reply";
            break;
        case TRANSPORT_INVALID_REPLY:
            detail = "invalid worker reply";
            break;
        default:
            detail = "worker rejected the request";
    }
    char message[160];
    snprintf(message, sizeof(message), "Mali worker %s: %s", names[operation], detail);
    return make_error(code, message);
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

bool tensor_size(const struct mali_tensor *tensor, size_t *size) {
    size_t item_size = element_size(tensor->type);
    return tensor->rank <= MALI_MAX_RANK && item_size != 0 &&
           array_size(tensor->dims, tensor->rank, item_size, size);
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

static int transfer(struct mali_resource *resource, enum mali_operation operation,
                    size_t offset, size_t size, const void *input, void *output) {
    struct mali_reply reply;
    while (size > 0) {
        const size_t chunk = size < MALI_TRANSFER_CHUNK ? size : MALI_TRANSFER_CHUNK;
        struct mali_request request = {
            .operation = operation, .handle = resource->handle, .offset = offset, .length = chunk,
        };
        int status = exchange(resource->client, &request, input, input ? chunk : 0,
                              output, output ? chunk : 0, &reply);
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

PJRT_Buffer *create_buffer(PJRT_Client *client, const struct mali_tensor *tensor, PJRT_Event *ready) {
    if (tensor->rank > MALI_MAX_RANK) {
        return NULL;
    }
    PJRT_Buffer *buffer = calloc(1, sizeof(*buffer) + tensor->rank * 2 * sizeof(int64_t));
    if (buffer != NULL) {
        buffer->resource = create_resource(client, MALI_RELEASE, 0, ready);
        if (buffer->resource == NULL) {
            free(buffer);
            return NULL;
        }
        buffer->client = client;
        buffer->size = tensor->size;
        buffer->rank = tensor->rank;
        buffer->type = tensor->type;
        for (size_t i = 0; i < buffer->rank; ++i) {
            buffer->dims[i] = tensor->dims[i];
            buffer->dims[buffer->rank + i] = (int64_t)(buffer->rank - 1 - i);
        }
    }
    return buffer;
}

static PJRT_Error *client_buffer_from_host(PJRT_Client_BufferFromHostBuffer_Args *args) {
    args->buffer = NULL;
    args->done_with_host_buffer = NULL;
    if ((args->device != NULL && args->device != &args->client->device) ||
        (args->memory != NULL && args->memory != &args->client->memory) ||
        (args->num_dims != 0 && args->dims == NULL) ||
        args->num_dims > MALI_MAX_RANK) {
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
    struct mali_tensor tensor = {.size = size, .type = args->type, .rank = args->num_dims};
    for (size_t i = 0; i < args->num_dims; ++i) {
        tensor.dims[i] = args->dims[i];
    }
    PJRT_Buffer *buffer = create_buffer(args->client, &tensor, ready_event());
    if (buffer == NULL) {
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not record Mali buffer");
    }
    unsigned char *packed = NULL;
    if (!dense) {
        packed = malloc(size);
        if (packed == NULL) {
            release_resource(buffer->resource);
            free(buffer);
            return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not pack strided host data");
        }
        pack_host_array(args, item_size, size, packed);
    }

    struct mali_request request = {.operation = MALI_ALLOCATE, .length = size};
    enum mali_operation operation = MALI_ALLOCATE;
    struct mali_reply reply;
    pthread_mutex_lock(&args->client->worker_lock);
    int status = exchange(args->client, &request, &tensor, sizeof(tensor), NULL, 0, &reply);
    if (status == MALI_OK) {
        buffer->resource->handle = reply.handle;
        operation = MALI_WRITE;
        status = transfer(buffer->resource, operation, 0, size,
                          packed != NULL ? packed : args->data, NULL);
    }
    pthread_mutex_unlock(&args->client->worker_lock);
    free(packed);
    if (status != MALI_OK) {
        release_resource(buffer->resource);
        free(buffer);
        return transfer_error(operation, status);
    }
    args->buffer = buffer;
    args->done_with_host_buffer = ready_event();
    return NULL;
}

static PJRT_Error *download(PJRT_Buffer *buffer, size_t offset, size_t size, void *output,
                            PJRT_Event **event) {
    *event = NULL;
    PJRT_Client *client = buffer->client;
    pthread_mutex_lock(&client->dispatch_lock);
    struct mali_resource *resource = buffer->resource;
    if (resource != NULL) {
        retain_resource(resource);
    }
    pthread_mutex_unlock(&client->dispatch_lock);
    if (resource == NULL) {
        return transfer_error(MALI_READ, MALI_NOT_FOUND);
    }
    int status = await_event(resource->ready);
    if (status != MALI_OK) {
        release_resource(resource);
        return transfer_error(MALI_EXECUTE, status);
    }
    pthread_mutex_lock(&client->worker_lock);
    status = transfer(resource, MALI_READ, offset, size, NULL, output);
    pthread_mutex_unlock(&client->worker_lock);
    release_resource(resource);
    if (status == MALI_OK) {
        *event = ready_event();
    }
    return transfer_error(MALI_READ, status);
}

static PJRT_Error *buffer_to_host(PJRT_Buffer_ToHostBuffer_Args *args) {
    args->event = NULL;
    if (!descending_layout(args->host_layout, args->src->rank)) {
        return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Unsupported Mali readback layout");
    }
    if (args->dst == NULL) {
        args->dst_size = args->src->size;
        args->event = ready_event();
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

static PJRT_Error *buffer_delete(PJRT_Buffer_Delete_Args *args) {
    delete_resource(args->buffer->client, &args->buffer->resource);
    return NULL;
}

static PJRT_Error *buffer_destroy(PJRT_Buffer_Destroy_Args *args) {
    if (args->buffer != NULL) {
        delete_resource(args->buffer->client, &args->buffer->resource);
        free(args->buffer);
    }
    return NULL;
}

static PJRT_Error *buffer_is_deleted(PJRT_Buffer_IsDeleted_Args *args) {
    args->is_deleted = resource_deleted(args->buffer->client, &args->buffer->resource);
    return NULL;
}

static PJRT_Error *buffer_ready_event(PJRT_Buffer_ReadyEvent_Args *args) {
    PJRT_Buffer *buffer = args->buffer;
    pthread_mutex_lock(&buffer->client->dispatch_lock);
    args->event = buffer->resource != NULL ? buffer->resource->ready : deleted_event();
    retain_event(args->event);
    pthread_mutex_unlock(&buffer->client->dispatch_lock);
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
