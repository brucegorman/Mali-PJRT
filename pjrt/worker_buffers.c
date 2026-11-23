#include <stdio.h>
#include <stdlib.h>

#include "iree/io/file_contents.h"
#include "iree/vm/bytecode/module.h"
#include "output_metadata.h"
#include "pjrt_c_api.h"
#include "protocol.h"
#include "worker_buffers.h"

struct device_buffer {
    struct device_buffer *next;
    uint64_t handle;
    iree_hal_buffer_view_t *view;
};

struct executable {
    struct executable *next;
    uint64_t handle;
    iree_runtime_session_t *session;
    iree_vm_function_t function;
    struct mali_executable_info info;
};

struct worker {
    iree_runtime_instance_t *instance;
    iree_hal_device_t *device;
    struct device_buffer *buffers;
    struct executable *executables;
    uint64_t next_handle;
};

static const iree_hal_element_type_t element_types[] = {
    [PJRT_Buffer_Type_PRED] = IREE_HAL_ELEMENT_TYPE_BOOL_8,
    [PJRT_Buffer_Type_S8] = IREE_HAL_ELEMENT_TYPE_SINT_8,
    [PJRT_Buffer_Type_S16] = IREE_HAL_ELEMENT_TYPE_SINT_16,
    [PJRT_Buffer_Type_S32] = IREE_HAL_ELEMENT_TYPE_SINT_32,
    [PJRT_Buffer_Type_S64] = IREE_HAL_ELEMENT_TYPE_SINT_64,
    [PJRT_Buffer_Type_U8] = IREE_HAL_ELEMENT_TYPE_UINT_8,
    [PJRT_Buffer_Type_U16] = IREE_HAL_ELEMENT_TYPE_UINT_16,
    [PJRT_Buffer_Type_U32] = IREE_HAL_ELEMENT_TYPE_UINT_32,
    [PJRT_Buffer_Type_U64] = IREE_HAL_ELEMENT_TYPE_UINT_64,
    [PJRT_Buffer_Type_F16] = IREE_HAL_ELEMENT_TYPE_FLOAT_16,
    [PJRT_Buffer_Type_F32] = IREE_HAL_ELEMENT_TYPE_FLOAT_32,
    [PJRT_Buffer_Type_F64] = IREE_HAL_ELEMENT_TYPE_FLOAT_64,
    [PJRT_Buffer_Type_BF16] = IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
    [PJRT_Buffer_Type_C64] = IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64,
    [PJRT_Buffer_Type_C128] = IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128,
};

static uint32_t pjrt_type(iree_hal_element_type_t type) {
    switch (type) {
        case IREE_HAL_ELEMENT_TYPE_INT_8: return PJRT_Buffer_Type_S8;
        case IREE_HAL_ELEMENT_TYPE_INT_16: return PJRT_Buffer_Type_S16;
        case IREE_HAL_ELEMENT_TYPE_INT_32: return PJRT_Buffer_Type_S32;
        case IREE_HAL_ELEMENT_TYPE_INT_64: return PJRT_Buffer_Type_S64;
        default: break;
    }
    for (uint32_t i = 1; i < sizeof(element_types) / sizeof(element_types[0]); ++i) {
        if (element_types[i] == type) {
            return i;
        }
    }
    return PJRT_Buffer_Type_INVALID;
}

static enum mali_status runtime_status(iree_status_t status) {
    if (iree_status_is_ok(status)) {
        return MALI_OK;
    }
    bool exhausted = iree_status_is_resource_exhausted(status);
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return exhausted ? MALI_OUT_OF_MEMORY : MALI_RUNTIME_ERROR;
}

static struct device_buffer **find_buffer(struct worker *worker, uint64_t handle) {
    struct device_buffer **link = &worker->buffers;
    while (*link != NULL && (*link)->handle != handle) {
        link = &(*link)->next;
    }
    return link;
}

static struct executable **find_executable(struct worker *worker, uint64_t handle) {
    struct executable **link = &worker->executables;
    while (*link != NULL && (*link)->handle != handle) {
        link = &(*link)->next;
    }
    return link;
}

static void release_buffer(struct device_buffer **link) {
    struct device_buffer *buffer = *link;
    *link = buffer->next;
    iree_hal_buffer_view_release(buffer->view);
    free(buffer);
}

static void release_executable(struct executable **link) {
    struct executable *executable = *link;
    *link = executable->next;
    iree_runtime_session_release(executable->session);
    free(executable);
}

static uint64_t retain_view(struct worker *worker, iree_hal_buffer_view_t *view) {
    struct device_buffer *buffer = malloc(sizeof(*buffer));
    if (buffer == NULL) {
        return 0;
    }
    *buffer = (struct device_buffer){
        .next = worker->buffers, .handle = worker->next_handle++, .view = view,
    };
    worker->buffers = buffer;
    return buffer->handle;
}

static enum mali_status allocate_buffer(struct worker *worker, const struct mali_tensor *tensor,
                                        uint64_t *handle) {
    if (tensor->rank > MALI_MAX_RANK ||
        tensor->type >= sizeof(element_types) / sizeof(element_types[0]) ||
        element_types[tensor->type] == 0) {
        return MALI_INVALID;
    }
    iree_hal_dim_t dims[MALI_MAX_RANK];
    for (uint32_t i = 0; i < tensor->rank; ++i) {
        if (tensor->dims[i] < 0) {
            return MALI_INVALID;
        }
        dims[i] = (iree_hal_dim_t)tensor->dims[i];
    }
    iree_hal_element_type_t type = element_types[tensor->type];
    iree_device_size_t expected_size = 0;
    enum mali_status status = runtime_status(iree_hal_buffer_compute_view_size(
        tensor->rank, dims, type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &expected_size));
    if (status != MALI_OK || expected_size != tensor->size) {
        return MALI_INVALID;
    }
    iree_hal_buffer_t *storage = NULL;
    iree_hal_buffer_params_t params = {
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
    };
    status = runtime_status(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(worker->device), params, tensor->size ? tensor->size : 1,
        &storage));
    if (status != MALI_OK) {
        return status;
    }
    iree_hal_buffer_view_t *view = NULL;
    status = runtime_status(iree_hal_buffer_view_create(
        storage, tensor->rank, dims, type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        iree_allocator_system(), &view));
    iree_hal_buffer_release(storage);
    if (status == MALI_OK) {
        *handle = retain_view(worker, view);
        if (*handle == 0) {
            iree_hal_buffer_view_release(view);
            status = MALI_OUT_OF_MEMORY;
        }
    }
    return status;
}

static enum mali_status load_module(struct worker *worker, const char *path, uint64_t *handle,
                                    struct mali_executable_info *info) {
    struct executable *executable = calloc(1, sizeof(*executable));
    if (executable == NULL) {
        return MALI_OUT_OF_MEMORY;
    }
    iree_runtime_session_options_t options;
    iree_runtime_session_options_initialize(&options);
    iree_allocator_t allocator = iree_allocator_system();
    iree_status_t status = iree_runtime_session_create_with_device(
        worker->instance, &options, worker->device, allocator, &executable->session);
    iree_vm_module_t *module = NULL;
    if (iree_status_is_ok(status)) {
        iree_io_file_contents_t *contents = NULL;
        status = iree_io_file_contents_read(iree_make_cstring_view(path), allocator, &contents);
        if (iree_status_is_ok(status)) {
            status = iree_vm_bytecode_module_create(
                iree_runtime_instance_vm_instance(worker->instance),
                IREE_VM_BYTECODE_MODULE_FLAG_NONE, contents->const_buffer,
                iree_io_file_contents_deallocator(contents), allocator, &module);
            if (!iree_status_is_ok(status)) {
                iree_io_file_contents_free(contents);
            }
        }
    }
    if (iree_status_is_ok(status)) {
        status = iree_runtime_session_append_module(executable->session, module);
    }
    if (iree_status_is_ok(status)) {
        status = iree_vm_module_lookup_function_by_name(
            module, IREE_VM_FUNCTION_LINKAGE_EXPORT, IREE_SV("main"), &executable->function);
    }
    iree_host_size_t inputs = 0, outputs = 0;
    if (iree_status_is_ok(status)) {
        iree_vm_function_signature_t signature = iree_vm_function_signature(&executable->function);
        status = iree_vm_function_call_count_arguments_and_results(&signature, &inputs, &outputs);
    }
    iree_vm_module_release(module);
    enum mali_status result = runtime_status(status);
    if (result != MALI_OK || inputs > MALI_TRANSFER_CHUNK / sizeof(uint64_t) ||
        outputs > MALI_MAX_OUTPUTS) {
        iree_runtime_session_release(executable->session);
        free(executable);
        return result == MALI_OK ? MALI_INVALID : result;
    }
    executable->handle = worker->next_handle++;
    executable->info.inputs = (uint32_t)inputs;
    executable->info.outputs = (uint32_t)outputs;
    iree_string_view_t declaration = iree_vm_function_lookup_attr_by_name(
        &executable->function, IREE_SV("iree.abi.declaration"));
    if (!parse_output_metadata(declaration, &executable->info)) {
        fputs("Unsupported IREE output signature for Mali\n", stderr);
        iree_runtime_session_release(executable->session);
        free(executable);
        return MALI_UNSUPPORTED_SIGNATURE;
    }
    executable->next = worker->executables;
    worker->executables = executable;
    *handle = executable->handle;
    *info = executable->info;
    return MALI_OK;
}

static enum mali_status execute_module(struct worker *worker, struct executable *executable,
                                       const unsigned char *payload, struct mali_result *results) {
    iree_runtime_call_t call;
    iree_status_t status =
        iree_runtime_call_initialize(executable->session, executable->function, &call);
    if (!iree_status_is_ok(status)) {
        return runtime_status(status);
    }
    for (uint32_t i = 0; i < executable->info.inputs && iree_status_is_ok(status); ++i) {
        uint64_t handle;
        memcpy(&handle, payload + i * sizeof(handle), sizeof(handle));
        struct device_buffer *buffer = *find_buffer(worker, handle);
        status = buffer != NULL ? iree_runtime_call_inputs_push_back_buffer_view(&call, buffer->view)
                                : iree_make_status(IREE_STATUS_NOT_FOUND, "Unknown Mali input");
    }
    if (iree_status_is_ok(status)) {
        status = iree_runtime_call_invoke(&call, 0);
    }
    uint32_t produced = 0;
    while (produced < executable->info.outputs && iree_status_is_ok(status)) {
        iree_hal_buffer_view_t *view = NULL;
        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &view);
        if (!iree_status_is_ok(status)) {
            break;
        }
        struct mali_result *result = &results[produced];
        const struct mali_tensor *expected = &executable->info.output_tensors[produced];
        result->tensor = *expected;
        iree_hal_element_type_t actual_type = iree_hal_buffer_view_element_type(view);
        iree_hal_element_type_t expected_type = element_types[expected->type];
        bool type_matches = pjrt_type(actual_type) == expected->type ||
            (iree_hal_element_numerical_type(actual_type) == IREE_HAL_NUMERICAL_TYPE_INTEGER &&
             (iree_hal_element_numerical_type(expected_type) == IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED ||
              iree_hal_element_numerical_type(expected_type) == IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED) &&
             iree_hal_element_bit_count(actual_type) == iree_hal_element_bit_count(expected_type));
        bool shape_matches = iree_hal_buffer_view_shape_rank(view) == expected->rank;
        const iree_hal_dim_t *dims = iree_hal_buffer_view_shape_dims(view);
        for (uint32_t i = 0; shape_matches && i < expected->rank; ++i) {
            shape_matches = dims[i] == (iree_hal_dim_t)expected->dims[i];
        }
        result->tensor.size = iree_hal_buffer_view_byte_length(view);
        if (!type_matches || !shape_matches ||
            iree_hal_buffer_view_encoding_type(view) != IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
            status = iree_make_status(IREE_STATUS_INTERNAL, "Mali output differs from its compiled signature");
        } else {
            result->handle = retain_view(worker, view);
            if (result->handle != 0) {
                ++produced;
                continue;
            }
            status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "Could not retain Mali output");
        }
        iree_hal_buffer_view_release(view);
    }
    iree_runtime_call_deinitialize(&call);
    if (!iree_status_is_ok(status)) {
        for (uint32_t i = 0; i < produced; ++i) {
            release_buffer(find_buffer(worker, results[i].handle));
        }
    }
    return runtime_status(status);
}

static enum mali_status transfer(struct worker *worker, const struct mali_request *request,
                                  size_t input_size, unsigned char *payload) {
    struct device_buffer **link = find_buffer(worker, request->handle);
    if (*link == NULL) {
        return MALI_NOT_FOUND;
    }
    if (request->operation == MALI_RELEASE && input_size == 0) {
        release_buffer(link);
        return MALI_OK;
    }
    iree_hal_buffer_view_t *view = (*link)->view;
    iree_device_size_t size = iree_hal_buffer_view_byte_length(view);
    if (request->offset > size || request->length > size - request->offset ||
        request->length > MALI_TRANSFER_CHUNK) {
        return MALI_INVALID;
    }
    iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(view);
    if (request->operation == MALI_WRITE && input_size == request->length) {
        return runtime_status(iree_hal_device_transfer_h2d(
            worker->device, payload, buffer, request->offset, request->length, 0,
            iree_infinite_timeout()));
    }
    if (request->operation == MALI_READ && input_size == 0) {
        return runtime_status(iree_hal_device_transfer_d2h(
            worker->device, buffer, request->offset, payload, request->length, 0,
            iree_infinite_timeout()));
    }
    return MALI_INVALID;
}

int serve_buffers(int socket, iree_runtime_instance_t *instance, iree_hal_device_t *device) {
    struct worker worker = {.instance = instance, .device = device, .next_handle = 1};
    unsigned char payload[MALI_TRANSFER_CHUNK];
    int result = 1;
    for (;;) {
        struct mali_request request;
        ssize_t count = receive_packet(socket, &request, sizeof(request), payload, sizeof(payload));
        if (count == 0) {
            result = 0;
            break;
        }
        if (count < (ssize_t)sizeof(request) ||
            count > (ssize_t)(sizeof(request) + sizeof(payload))) {
            fputs("Invalid Mali packet length\n", stderr);
            break;
        }

        struct mali_reply reply = {.status = MALI_INVALID};
        size_t input_size = (size_t)count - sizeof(request);
        const void *output = NULL;
        size_t output_size = 0;
        struct mali_executable_info info;
        struct mali_result results[MALI_MAX_OUTPUTS] = {0};
        if (request.operation == MALI_ALLOCATE && input_size == sizeof(struct mali_tensor)) {
            struct mali_tensor tensor;
            memcpy(&tensor, payload, sizeof(tensor));
            reply.status = allocate_buffer(&worker, &tensor, &reply.handle);
        } else if (request.operation == MALI_LOAD && input_size > 1 &&
                   payload[input_size - 1] == '\0') {
            reply.status = load_module(&worker, (const char *)payload, &reply.handle, &info);
            output = &info;
            output_size = sizeof(info);
        } else if (request.operation == MALI_EXECUTE || request.operation == MALI_UNLOAD) {
            struct executable **link = find_executable(&worker, request.handle);
            if (*link == NULL) {
                reply.status = MALI_NOT_FOUND;
            } else if (request.operation == MALI_UNLOAD && input_size == 0) {
                release_executable(link);
                reply.status = MALI_OK;
            } else if (request.operation == MALI_EXECUTE &&
                       input_size == (*link)->info.inputs * sizeof(uint64_t)) {
                reply.status = execute_module(&worker, *link, payload, results);
                output = results;
                output_size = (*link)->info.outputs * sizeof(results[0]);
            }
        } else {
            reply.status = transfer(&worker, &request, input_size, payload);
            if (request.operation == MALI_READ && reply.status == MALI_OK) {
                output = payload;
                output_size = request.length;
            }
        }
        if (!send_packet(socket, &reply, sizeof(reply), output,
                         reply.status == MALI_OK ? output_size : 0)) {
            fputs("Could not reply to Mali request\n", stderr);
            break;
        }
    }
    while (worker.executables != NULL) {
        release_executable(&worker.executables);
    }
    while (worker.buffers != NULL) {
        release_buffer(&worker.buffers);
    }
    return result;
}
